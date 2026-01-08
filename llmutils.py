import json
import urllib.request
from urllib.error import HTTPError


def embed_one(tip, document_or_query):
    # model = "nomic-embed-text"
    model = "bge-m3"
    # model = "qwen3-embedding"
    prompt = tip + document_or_query
    try:
        with urllib.request.urlopen(
            "http://localhost:11434/api/embeddings",
            data=json.dumps({"model": model, "prompt": prompt}).encode("utf-8"),
        ) as response:
            status = response.status
            answer = json.loads(response.read().decode("utf-8"))
            embedding = answer["embedding"]
    except HTTPError as e:
        status = e.status
        embedding = None
    return status, embedding


def embed_multiple(tip, documents_or_queries):
    # model = "nomic-embed-text"
    model = "bge-m3"
    # model = "qwen3-embedding"
    inputs = [tip + document_or_query for document_or_query in documents_or_queries]
    try:
        with urllib.request.urlopen(
            "http://localhost:11434/api/embed",
            data=json.dumps({"model": model, "input": inputs}).encode("utf-8"),
        ) as response:
            status = response.status
            answer = json.loads(response.read().decode("utf-8"))
            embeddings = answer["embeddings"]
    except HTTPError as e:
        status = e.status
        embeddings = None
    return status, embeddings
