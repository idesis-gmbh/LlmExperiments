import sys
from dbutils import (
    search_wikipedia_term,
    ingest_wikipedia_page,
    load_faiss,
    query_faiss,
    query_fts,
)
from llmutils import assemble_messages, chat, chat_stream

SYSTEM_PROMPT = (
    "Double check an answer with embeddings, which are loaded lazily. "
    # "At least query the local Wikipedia FTS chunk index or embeddings. "
    "At least query the local Wikipedia chunk embeddings. "
    "Additionally search the local Wikipedia FTS page index to ingest missing pages if appropriate. "
    "Indicate if RAG results are still not available. "
)
INDEX = load_faiss()
TOOLS = [
    {
        "description": {
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
        "handler": lambda tool_call: query_faiss(
            INDEX, tool_call["function"]["arguments"]["prompt"]
        ),
    },
    {
        "description": {
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
        "handler": lambda tool_call: search_wikipedia_term(
            tool_call["function"]["arguments"]["term"]
        ),
    },
    {
        "description": {
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
        "handler": lambda tool_call: ingest_wikipedia_page(
            INDEX,
            tool_call["function"]["arguments"]["project_name"],
            tool_call["function"]["arguments"]["page_name"],
        ),
    },
]
"""
{
    "description": {
        "type": "function",
        "function": {
            "name": "query_fts",
            "description": "Queries the RAG knowledge base (e.g., ingested Wikipedia markdown sections) using lexical retrieval to return relevant context chunks for answering a question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "term": {
                        "type": "string",
                        "description": "Search term used in SQLite FTS MATCH query containing logical operators like AND/OR",
                    },
                },
                "required": ["term"],
            }
        }
    },
    "handler": lambda tool_call: query_fts(
        tool_call["function"]["arguments"]["term"]
    ),
},
"""


def run_chat(user_prompt, tools=None):
    system_prompt = SYSTEM_PROMPT if tools else None
    print(user_prompt)
    messages = assemble_messages(system_prompt, user_prompt)
    status, thinking, content = chat(messages, tools)
    print(status, thinking, content)


def run_chat_stream(user_prompt, tools=None):
    system_prompt = SYSTEM_PROMPT if tools else None
    print(user_prompt)
    messages = assemble_messages(system_prompt, user_prompt)
    for status, thinking, content in chat_stream(messages, tools):
        assert status == 200
        if thinking:
            print(thinking, end="", flush=True)
        elif content:
            print(content, end="", flush=True)


if __name__ == "__main__":
    for user_prompt in [
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
        if "chat" in sys.argv[1:]:
            run_chat(user_prompt)
        if "chat_stream" in sys.argv[1:]:
            run_chat_stream(user_prompt)
        if "rag_chat" in sys.argv[1:]:
            run_chat(user_prompt, TOOLS)
        if "rag_chat_stream" in sys.argv[1:]:
            run_chat_stream(user_prompt, TOOLS)
