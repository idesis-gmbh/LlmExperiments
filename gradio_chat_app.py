import gradio as gr
from llmutils import chat


def run_chat(message, history):
    messages = [
        {
            "role": entry["role"],
            "content": "".join(chunk["text"] for chunk in entry["content"]),
        }
        for entry in history
    ]
    assert len(messages) == 0 or messages[-1]["role"] != "user"
    messages.append({"role": "user", "content": message})
    for event in chat(messages):
        assert event["status"] == 200
        if event["type"] == "content":
            return event["data"]
    return ""


gr.ChatInterface(
    fn=run_chat,
).launch()
