import gradio as gr
from dbutils import (
    search_wikipedia_term,
    ingest_wikipedia_page,
    load_faiss,
    query_faiss,
    query_fts,
)
from llmutils import chat_stream


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


def run_chat_stream(message, history):
    print("*** history", history)
    messages = [
        {
            "role": entry["role"],
            "content": "".join(chunk["text"] for chunk in entry["content"]),
        }
        for entry in history
    ]    
    assert len(messages) == 0 or messages[-1]["role"] != "user"
    messages.append({"role": "user", "content": message})
    # messages.append({"role": "assistant", "content": ""})
    print("*** messages 1", messages)
    response = {"thinking": "", "tooling": "", "content": ""}

# *** history []
# *** event {'type': 'thinking', 'status': 200, 'data': 'Okay'}
# *** response {'thinking': 'Okay', 'tooling': '', 'content': ''}
# *** messages[-1] {'role': 'assistant', 'content': '', 'thinking': 'Okay'}

    for event in chat_stream(messages, TOOLS):
        print("*** event", event)
        assert event["status"] == 200
        response[event["type"]] += event["data"]
        print("*** response", response)
        # print("*** messages 2", messages)
        # if event["type"] == "content":
        #     messages[-1]["content"] = response["content"]
        # print("*** messages 3", messages)
        print("*** yield thinking", response["thinking"])
        print("*** yield tooling", response["tooling"])
        print("*** yield content", messages)
        yield response["thinking"], response["tooling"], messages


with gr.Blocks() as demo:
    gr.Markdown("## RAG Chat PoC")
    thinking_box = gr.Textbox(label="Chatbot thinking", lines=3, interactive=False)
    tooling_box = gr.Textbox(label="Chatbot tooling", lines=3, interactive=False)
    chatbot = gr.Chatbot(label="Chat")
    message = gr.Textbox(label="Message")
    send = gr.Button("Send")
    send.click(
        run_chat_stream,
        inputs=[message, chatbot],
        outputs=[thinking_box, tooling_box, chatbot],
    )
    demo.launch()