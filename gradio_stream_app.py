import gradio as gr
from dbutils import (
    search_wikipedia_term,
    ingest_wikipedia_page,
    load_faiss,
    query_faiss,
    query_fts,
)
from llmutils import chat_stream
from env import SYSTEM_PROMPT, TOOLS

INDEX = load_faiss()
TOOLS[0]["handler"] = lambda tool_call: query_faiss(
    INDEX, tool_call["function"]["arguments"]["prompt"]
)
TOOLS[1]["handler"] = lambda tool_call: search_wikipedia_term(
    tool_call["function"]["arguments"]["term"]
)
TOOLS[2]["handler"] = lambda tool_call: ingest_wikipedia_page(
    INDEX,
    tool_call["function"]["arguments"]["project_name"],
    tool_call["function"]["arguments"]["page_name"],
)
# TOOLS[3]["handler"] = (
#     lambda tool_call: query_fts(tool_call["function"]["arguments"]["term"]),
# )


def run_chat_stream(message, history):
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ] + [
        {
            "role": entry["role"],
            "content": "".join(chunk["text"] for chunk in entry["content"]),
        }
        for entry in history
    ]
    assert len(messages) == 0 or messages[-1]["role"] != "user"
    messages.append({"role": "user", "content": message})
    response = {"thinking": "", "tooling": "", "content": ""}
    for event in chat_stream(messages, TOOLS):
        assert event["status"] == 200
        response[event["type"]] += event["data"]
        yield response["thinking"], response["tooling"], messages, ""


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
        outputs=[thinking_box, tooling_box, chatbot, message],
    )
    demo.launch()
