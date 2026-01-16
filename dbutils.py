import bz2
from itertools import batched
import sqlite3
import zlib
import numpy as np
from httputils import WikipediaHTMLParser, get_wikipedia_page
from llmutils import embed_one, embed_multiple
import faiss


def load_wikipedia_pageviews(path):
    projects = dict()
    pages = dict()
    with bz2.open(path, "rt") as file:
        for index, line in enumerate(file):
            if index % 1_000_000 == 0:
                print(index, len(pages))
            try:
                project, page, size, access, accumulated_views, detailed_views = (
                    line.strip().split()
                )
            except ValueError:
                continue
            project = project.lower()
            projects[project] = None
            if project != "en.wikipedia" or size == "null":
                continue
            if (project, page) not in pages:
                pages[(project, page)] = int(accumulated_views)
            else:
                pages[(project, page)] += int(accumulated_views)
    with sqlite3.connect("data/rag.db") as connection:
        cursor = connection.cursor()
        cursor.executemany(
            "INSERT OR REPLACE INTO projects (name) VALUES (?)",
            [(project,) for project in projects],
        )
        connection.commit()
        for project_id, project_name in cursor.execute("SELECT * FROM projects", []):
            projects[project_name] = project_id
        for batch in batched(pages.items(), n=1_000):
            cursor.executemany(
                "INSERT OR REPLACE INTO pages (project_id, name, views) VALUES (?, ?, ?)",
                [(projects[project], page, views) for (project, page), views in batch],
            )
            connection.commit()


def query_chunks(connection, page_id=None):
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


def update_faiss(connection, index, page_id=None):
    chunk_ids, embeddings = query_chunks(connection, page_id)
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
                    if (not texts and distance >= 0.5) or distance >= 0.6:
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
            [term.replace("?", ""), k],
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
            [term.replace("?", ""), min_views, k],
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


def get_and_update_wikipedia_page(connection, page_id, project_name, page_name):
    status, html = get_wikipedia_page(project_name, page_name)
    html_compressed = zlib.compress(html.encode("utf-8")) if html else None
    cursor = connection.cursor()
    cursor.execute(
        "UPDATE pages SET status = ?, html = ? WHERE id = ?",
        [status, html_compressed, page_id],
    )
    return status, html_compressed


def scrape_wikipedia_pages(limit):
    count = 0
    with sqlite3.connect("data/rag.db") as connection:
        cursor = connection.cursor()
        for (
            page_id,
            project_name,
            page_name,
            status,
            html_compressed,
        ) in cursor.execute(
            "SELECT pages.id as page_id, projects.name as project_name, pages.name as page_name, pages.status as status, pages.html as html "
            "FROM pages INNER JOIN projects on pages.project_id = projects.id "
            "ORDER BY views DESC LIMIT ?",
            [limit],
        ):
            if status:
                continue
            status, html_compressed = get_and_update_wikipedia_page(
                connection, page_id, project_name, page_name
            )
            count += 1
            if count % 100 == 0:
                connection.commit()
        connection.commit()


def update_wikipedia_sections(connection, page_id, html):
    parser = WikipediaHTMLParser()
    parser.feed(html)
    # print(parser.markdown)
    markdown = "".join(parser.markdown)
    cursor2 = connection.cursor()
    cursor2.execute(
        "UPDATE pages SET markdown = ? WHERE id = ?",
        [markdown, page_id],
    )
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
                cursor2.execute(
                    "INSERT OR REPLACE INTO chunks (page_id, text, status, embedding) VALUES (?, ?, ?, ?)",
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
        cursor2.executemany(
            "INSERT OR REPLACE INTO chunks (page_id, text, status, embedding) VALUES (?, ?, ?, ?)",
            [
                (page_id, text, event["status"], sqlite3.Binary(embedding))
                for text, embedding in zip(texts, embeddings)
            ],
        )


def extract_wikipedia_sections():
    with sqlite3.connect("data/rag.db") as connection:
        cursor = connection.cursor()
        for page_id, page_name, html_compressed in cursor.execute(
            "SELECT id, name, html FROM pages WHERE html IS NOT NULL AND markdown IS NULL",
            [],
        ):
            print(page_name)
            html = zlib.decompress(html_compressed).decode("utf-8")
            update_wikipedia_sections(connection, page_id, html)
            connection.commit()


def ingest_wikipedia_page(index, project_name, page_name):
    status = 404
    with sqlite3.connect("data/rag.db") as connection:
        cursor = connection.cursor()
        for (
            page_id,
            status,
            html_compressed,
            markdown,
        ) in cursor.execute(
            "SELECT pages.id as page_id, pages.status as status, pages.html, pages.markdown "
            "FROM pages INNER JOIN projects on pages.project_id = projects.id "
            "WHERE projects.name = ? AND pages.name = ?",
            [project_name, page_name],
        ):
            if not status:
                status, html_compressed = get_and_update_wikipedia_page(
                    connection, page_id, project_name, page_name
                )
            if html_compressed and not markdown:
                html = zlib.decompress(html_compressed).decode("utf-8")
                update_wikipedia_sections(connection, page_id, html)
                update_faiss(connection, index, page_id)
            break
        connection.commit()
        return status
