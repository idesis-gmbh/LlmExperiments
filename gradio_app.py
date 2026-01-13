from llmutils import chat
import gradio as gr


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
    status, thinking, content = chat(messages)
    assert status == 200
    return f"{thinking}{content}"


gr.ChatInterface(
    fn=run_chat,
).launch()
