import bz2
from itertools import batched
import sqlite3
import zlib
import numpy as np
from httputils import WikipediaHTMLParser, get_wikipedia_page
from llmutils import embed_one, embed_multiple
from ftsutils import sanitize_fts_query
import faiss


def store_projects(connection, projects):
    cursor = connection.cursor()
    cursor.executemany(
        "INSERT OR IGNORE INTO projects (name) VALUES (?)",
        [(project_name,) for project_name in projects],
    )
    connection.commit()


def load_projects(connection):
    projects = {}
    cursor = connection.cursor()
    for project_id, project_name in cursor.execute("SELECT * FROM projects", []):
        projects[project_name] = project_id
    connection.commit()
    return projects


def store_pages(connection, projects, pages):
    cursor = connection.cursor()
    cursor.executemany(
        "INSERT OR IGNORE INTO pages (project_id, name, views) VALUES (?, ?, ?)",
        [
            (projects[project_name], page_name, views)
            for (project_name, page_name), views in pages
        ],
    )
    connection.commit()


def update_page_status_html(cursor, page_id, status, html):
    cursor.execute(
        "UPDATE pages SET status = ?, html = ? WHERE id = ?",
        [status, html, page_id],
    )


def update_page_markdown(cursor, page_id, markdown):
    cursor.execute(
        "UPDATE pages SET markdown = ? WHERE id = ?",
        [markdown, page_id],
    )


def store_page(connection, project_id, page_name, markdown):
    cursor = connection.cursor()
    cursor.execute(
        "INSERT OR IGNORE INTO pages (project_id, name, views, markdown) VALUES (?,?, ?, ?)",
        [project_id, page_name, 0, markdown],
    )
    connection.commit()


def load_page(connection, project_id, page_name):
    cursor = connection.cursor()
    markdown = list(cursor.execute(
        "SELECT markdown FROM pages WHERE project_id = ? AND name = ?",
        [project_id, page_name],
    ))
    connection.commit()
    assert len(markdown) <= 1
    return markdown[0][0] if markdown else None


def load_chunks(connection, page_id=None):
    chunk_ids = []
    embeddings = []
    cursor = connection.cursor()
    if not page_id:
        cursor.execute("SELECT id, embedding FROM chunks WHERE status = 200", [])
    else:
        cursor.execute(
            "SELECT id, embedding FROM chunks WHERE page_id = ? AND status = 200",
            [page_id],
        )
    for chunk_id, embedding in cursor:
        chunk_ids.append(chunk_id)
        embeddings.append(np.frombuffer(embedding, dtype="float32"))
    return chunk_ids, embeddings


def load_wikipedia_pageviews(path):
    projects = dict()
    pages = dict()
    with bz2.open(path, "rt") as file:
        for index, line in enumerate(file):
            if index % 1_000_000 == 0:
                print(index, len(pages))
            try:
                project_name, page, size, access, accumulated_views, detailed_views = (
                    line.strip().split()
                )
            except ValueError:
                continue
            project_name = project_name.lower()
            projects[project_name] = None
            if project_name != "en.wikipedia" or size == "null":
                continue
            if (project_name, page) not in pages:
                pages[(project_name, page)] = int(accumulated_views)
            else:
                pages[(project_name, page)] += int(accumulated_views)
    with sqlite3.connect("data/rag.db") as connection:
        store_projects(connection, projects)
        projects = load_projects(connection)
        for index, batch in enumerate(batched(pages.items(), n=1_000)):
            if index % 1_000 == 0:
                print(index * 1_000, len(pages))
            store_pages(connection, projects, batch)


def update_faiss(connection, index, page_id=None):
    chunk_ids, embeddings = load_chunks(connection, page_id)
    if chunk_ids and embeddings:
        X = np.vstack(embeddings)
        faiss.normalize_L2(X)
        I = np.array(chunk_ids, dtype="int64")
        index.add_with_ids(X, I)
        # test_query = X[0].reshape(1, -1)
        # test_D, test_I = index.search(test_query, 3)
    return index


def load_faiss():
    with sqlite3.connect("data/rag.db") as connection:
        # index = faiss.IndexIDMap(faiss.IndexFlatIP(768))
        index = faiss.IndexIDMap(faiss.IndexFlatIP(1024))
        # index = faiss.IndexIDMap(faiss.IndexFlatIP(4096))
        return update_faiss(connection, index)


def query_faiss(index, prompt, k=5):
    texts = []
    with sqlite3.connect("data/rag.db") as connection:
        # status, embedding = embed_one("search_query: ", prompt)
        event = embed_one("", prompt)
        if event["status"] == 200:
            query = np.array(event["data"], dtype="float32").reshape(1, -1)
            faiss.normalize_L2(query)
            D, I = index.search(query, k=k)
            cursor = connection.cursor()
            for distance, id in zip(D[0], I[0]):
                for (text,) in cursor.execute(
                    "SELECT text FROM chunks WHERE id = ?",
                    [int(id)],
                ):
                    if (not texts and distance >= 0.6) or distance >= 0.65:
                        if text not in texts:
                            texts.append(text)
    return texts


def query_fts(term, k=5):
    texts = []
    with sqlite3.connect("data/rag.db") as connection:
        cursor = connection.cursor()
        for id, text, rank in cursor.execute(
            "SELECT chunks.id as chunk_id, chunks.text, chunks_fts.rank "
            "FROM chunks "
            "INNER JOIN chunks_fts ON chunks_fts.rowid = chunks.id "
            "WHERE chunks_fts.text MATCH ? "
            "ORDER BY chunks_fts.rank "
            "LIMIT ?",
            [sanitize_fts_query(term), k],
        ):
            if text not in texts:
                texts.append(text)
    return texts


def search_wikipedia_term(term, min_views=1_000, k=5):
    pages = []
    with sqlite3.connect("data/rag.db") as connection:
        cursor = connection.cursor()
        for (
            page_id,
            project_name,
            page_name,
            views,
            status,
        ) in cursor.execute(
            "SELECT pages.id as page_id, projects.name as project_name, pages.name as page_name, pages.views as views, pages.status as status "
            "FROM pages "
            "INNER JOIN projects on pages.project_id = projects.id "
            "INNER JOIN pages_fts ON pages_fts.rowid = pages.id "
            "WHERE pages_fts.name MATCH ? "
            "AND pages.views >= ? "
            "ORDER BY pages_fts.rank, pages.views DESC "
            "LIMIT ?",
            [sanitize_fts_query(term), min_views, k],
        ):
            pages.append(
                {
                    "page_id": page_id,
                    "project_name": project_name,
                    "page_name": page_name,
                    "views": views,
                    "status": status,
                }
            )
        connection.commit()
    return pages


def get_and_update_wikipedia_page(cursor, page_id, project_name, page_name):
    status, html = get_wikipedia_page(project_name, page_name)
    html_compressed = zlib.compress(html.encode("utf-8")) if html else None
    update_page_status_html(cursor, page_id, status, html_compressed)
    return status, html_compressed


def scrape_wikipedia_pages(limit):
    count = 0
    with sqlite3.connect("data/rag.db") as connection:
        cursor1 = connection.cursor()
        cursor2 = connection.cursor()
        for (
            page_id,
            project_name,
            page_name,
            status,
            html_compressed,
        ) in cursor1.execute(
            "SELECT pages.id as page_id, projects.name as project_name, pages.name as page_name, pages.status as status, pages.html as html "
            "FROM pages INNER JOIN projects on pages.project_id = projects.id "
            "ORDER BY views DESC LIMIT ?",
            [limit],
        ):
            if status:
                continue
            status, html_compressed = get_and_update_wikipedia_page(
                cursor2, page_id, project_name, page_name
            )
            count += 1
            if count % 100 == 0:
                connection.commit()
        connection.commit()


def update_wikipedia_sections(cursor, page_id, html):
    parser = WikipediaHTMLParser()
    parser.feed(html)
    # print(parser.markdown)
    markdown = "".join(parser.markdown)
    update_page_markdown(cursor, page_id, markdown)
    # print(parser.sections)
    """
    for section in parser.sections:
        text = "\n".join(section[0]) + "\n" + "\n".join(section[1])
        # print(text)
        # event = embed_one("search_document: ", text)
        event = embed_one("", text)
        if event["status"] == 200:
            embedding = np.array(event["data"]).astype("float32").tobytes()
            assert all(
                np.isclose(document, np.frombuffer(embedding, dtype="float32"))
            )
            cursor.execute(
                "INSERT OR IGNORE INTO chunks (page_id, text, status, embedding) VALUES (?, ?, ?, ?)",
                [
                    page_id,
                    text,
                    event["status"],
                    sqlite3.Binary(embedding),
                ],
            )
    """
    texts = []
    for section in parser.sections:
        text = "\n".join(section[0]) + "\n" + "\n".join(section[1])
        # print(text)
        texts.append(text)
    # event = embed_multiple("search_document: ", texs)
    event = embed_multiple("", texts)
    if event["status"] == 200:
        embeddings = []
        for document in event["data"]:
            embedding = np.array(document).astype("float32").tobytes()
            assert all(np.isclose(document, np.frombuffer(embedding, dtype="float32")))
            embeddings.append(embedding)
        cursor.executemany(
            "INSERT OR IGNORE INTO chunks (page_id, text, status, embedding) VALUES (?, ?, ?, ?)",
            [
                (page_id, text, event["status"], sqlite3.Binary(embedding))
                for text, embedding in zip(texts, embeddings)
            ],
        )


def extract_wikipedia_sections():
    with sqlite3.connect("data/rag.db") as connection:
        cursor1 = connection.cursor()
        cursor2 = connection.cursor()
        for page_id, page_name, html_compressed in cursor1.execute(
            "SELECT id, name, html FROM pages WHERE html IS NOT NULL AND markdown IS NULL",
            [],
        ):
            print(page_name)
            html = zlib.decompress(html_compressed).decode("utf-8")
            update_wikipedia_sections(cursor2, page_id, html)
            connection.commit()


def ingest_wikipedia_page(index, project_name, page_name):
    status = 404
    with sqlite3.connect("data/rag.db") as connection:
        cursor1 = connection.cursor()
        cursor2 = connection.cursor()
        for (
            page_id,
            status,
            html_compressed,
            markdown,
        ) in cursor1.execute(
            "SELECT pages.id as page_id, pages.status as status, pages.html, pages.markdown "
            "FROM pages INNER JOIN projects on pages.project_id = projects.id "
            "WHERE projects.name = ? AND pages.name = ?",
            [project_name, page_name],
        ):
            if not status:
                status, html_compressed = get_and_update_wikipedia_page(
                    cursor2, page_id, project_name, page_name
                )
            if html_compressed and not markdown:
                html = zlib.decompress(html_compressed).decode("utf-8")
                update_wikipedia_sections(connection, page_id, html)
                update_faiss(connection, index, page_id)
            break
        connection.commit()
        return status


def get_sqlite_schema():
    with sqlite3.connect("data/catalog.db") as connection:
        schema = {"tables": {}}
        cursor1 = connection.cursor()
        for name, sql in cursor1.execute(
            "SELECT name, sql "
            "FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'",
            [],
        ):
            schema["tables"][name] = sql
        connection.commit()
        return schema


def get_sqlite_tables():
    with sqlite3.connect("data/catalog.db") as connection:
        cursor = connection.cursor()
        tables = list(
            cursor.execute(
                "SELECT name "
                "FROM sqlite_master "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'",
                [],
            )
        )
        connection.commit()
        return tables


def get_sqlite_table(table):
    with sqlite3.connect("data/catalog.db") as connection:
        cursor = connection.cursor()
        sql = list(
            cursor.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
                [table],
            )
        )
        connection.commit()
        return sql


def query_sqlite(query, parameters):
    parameters = parameters or {}
    with sqlite3.connect("data/catalog.db") as connection:
        rows = []
        cursor = connection.cursor()
        cursor.execute(query, parameters)
        header = [desc[0] for desc in cursor.description]
        for values in cursor.fetchall():
            row = {key: value for key, value in zip(header, values)}
            rows.append(row)
        connection.commit()
        return rows
