import bz2
from itertools import batched
import json
import sqlite3
from urllib.error import HTTPError
import urllib.request
from html.parser import HTMLParser
import sys
import numpy as np
from dbutils import (
    search_wikipedia_term,
    ingest_wikipedia_page,
    load_faiss,
    query_faiss,
    query_fts,
)


def chat(prompt, with_tools=False):
    try:
        payload = {
            "model": "qwen3",
            "messages": [
                {
                    "role": "system",
                    "content": "Double check an answer with embeddings, which are loaded lazily."
                    # "At least query the local Wikipedia FTS chunk index or embeddings."
                    "At least query the local Wikipedia chunk embeddings."
                    "Additionally search the local Wikipedia FTS page index to ingest missing pages if appropriate."
                    "Indicate if RAG results are still not available.",
                },
                {"role": "user", "content": prompt},
            ],
        }
        if with_tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": "query_faiss",
                        "description": "Queries the RAG knowledge base (e.g., ingested Wikipedia markdown sections) using semantic retrieval to return relevant context chunks for answering a question.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "prompt": {
                                    "type": "string",
                                    "description": "Natural language question",
                                },
                            },
                            "required": ["prompt"],
                        },
                    },
                },
                # {
                #     "type": "function",
                #     "function": {
                #         "name": "query_fts",
                #         "description": "Queries the RAG knowledge base (e.g., ingested Wikipedia markdown sections) using lexical retrieval to return relevant context chunks for answering a question.",
                #         "parameters": {
                #             "type": "object",
                #             "properties": {
                #                 "term": {
                #                     "type": "string",
                #                     "description": "Search term used in SQLite FTS MATCH query containing logical operators like AND/OR",
                #                 },
                #             },
                #             "required": ["term"],
                #         },
                #     },
                # },
                {
                    "type": "function",
                    "function": {
                        "name": "search_wikipedia_term",
                        "description": "Performs a SQLite FTS5 full-text search over locally indexed Wikipedia page metadata (project names and page titles)."
                        "Returns metadata for the pages including the HTTP status of the page ingestion."
                        "Does not call Wikipedia APIs.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "term": {
                                    "type": "string",
                                    "description": "Search term used in SQLite FTS MATCH query containing logical operators like AND/OR",
                                }
                            },
                        },
                        "required": ["term"],
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "ingest_wikipedia_page",
                        "description": "Fetches a Wikipedia page via HTTP, converts it to markdown, splits it into semantic sections, embeds them, and stores them in FAISS and SQLite. Does NOT return page HTML or markdown content.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "project_name": {
                                    "type": "string",
                                    "description": "Wikipedia project name",
                                },
                                "page_name": {
                                    "type": "string",
                                    "description": "Wikipedia page title",
                                },
                            },
                        },
                        "required": ["project_name", "page_name"],
                    },
                },
            ]
        index = load_faiss()
        status = None
        thinking = ""
        result = ""
        done = False
        while not done:
            with urllib.request.urlopen(
                "http://localhost:11434/api/chat",
                data=json.dumps(payload).encode("utf-8"),
            ) as response:
                done = True
                status = response.status
                for line in response.read().decode("utf-8").splitlines():
                    answer = json.loads(line)["message"]
                    if "thinking" in answer:
                        thinking += answer["thinking"]
                    elif "tool_calls" in answer:
                        payload["messages"].append(answer)
                        for tool_call in answer["tool_calls"]:
                            if tool_call["function"]["name"] == "search_wikipedia_term":
                                pages = search_wikipedia_term(
                                    tool_call["function"]["arguments"]["term"]
                                )
                                payload["messages"].append(
                                    {
                                        "role": "tool",
                                        "tool_call_id": tool_call["id"],
                                        "content": json.dumps(pages),
                                    }
                                )
                                done = False
                            elif (
                                tool_call["function"]["name"] == "ingest_wikipedia_page"
                            ):
                                ingest_wikipedia_page(
                                    index,
                                    tool_call["function"]["arguments"]["project_name"],
                                    tool_call["function"]["arguments"]["page_name"],
                                )
                                payload["messages"].append(
                                    {
                                        "role": "tool",
                                        "tool_call_id": tool_call["id"],
                                        "content": json.dumps(status),
                                    }
                                )
                                done = False
                            elif tool_call["function"]["name"] == "query_faiss":
                                texts = query_faiss(
                                    index, tool_call["function"]["arguments"]["prompt"]
                                )
                                payload["messages"].append(
                                    {
                                        "role": "tool",
                                        "tool_call_id": tool_call["id"],
                                        "content": json.dumps(texts),
                                    }
                                )
                                done = False
                            elif tool_call["function"]["name"] == "query_fts":
                                texts = query_fts(
                                    tool_call["function"]["arguments"]["term"]
                                )
                                payload["messages"].append(
                                    {
                                        "role": "tool",
                                        "tool_call_id": tool_call["id"],
                                        "content": json.dumps(texts),
                                    }
                                )
                                done = False
                    else:
                        result += answer["content"]
    except HTTPError as e:
        status = e.status
        thinking = None
        result = None
    return status, thinking, result


if __name__ == "__main__":
    for basic_prompt in [
        # "Which tools are available?"
        # "Which Wikipedia pages could reference Google Chrome?"
        # "Tell me about Google Chrome."
        # "What year was the Berlin Wall built, and which countries were involved in its construction?",
        # "Who discovered penicillin, and how was the discovery made?",
        # "What is the chemical formula of ozone, and how does it differ from oxygen?",
        # "Who was the first woman to win a Nobel Prize, and in which field?",
        # "What is the capital of Mongolia, and what was its former name?",
        # "What information is known about Cleopatra’s appearance, and what is disputed?",
        # "What caused the Bronze Age collapse, and why is it still debated?",
        # "How accurate are medieval maps compared to modern cartography?",
        "What are the competing theories about the origin of life on Earth?",
        # "What historical facts about King Arthur are supported by evidence, if any?",
    ]:
        if "simple" in sys.argv[1:]:
            print("Basic prompt:", basic_prompt, flush=True)
            status, thinking, result = chat(basic_prompt)
            print("Thinking about basic prompt:", thinking, flush=True)
            print("Answer to basic prompt:", result, flush=True)
        if "rag" in sys.argv[1:]:
            print("Basic prompt:", basic_prompt, flush=True)
            status, thinking, result = chat(basic_prompt, with_tools=True)
            print("Thinking about basic prompt with tools:", thinking, flush=True)
            print("Answer to basic prompt with tools:", result, flush=True)
