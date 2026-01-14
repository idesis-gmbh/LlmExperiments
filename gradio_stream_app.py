import gradio as gr
from llmutils import chat_stream


def run_chat_stream(message, history):
    messages = [
        {
            "role": entry["role"],
            "content": "".join(chunk["text"] for chunk in entry["content"]),
        }
        for entry in history
    ]
    assert len(messages) == 0 or messages[-1]["role"] != "user"
    messages.append({"role": "user", "content": message})
    thinking_response = ""
    content_response = ""
    for status, thinking, content in chat_stream(messages):
        assert status == 200
        if thinking:
            thinking_response += thinking
            yield content_response, thinking_response
        elif content:
            content_response += content
            yield content_response, thinking_response


with gr.Blocks() as demo:
    gr.Markdown("## RAG Chat PoC")
    chat = gr.ChatInterface(
        fn=run_chat_stream,
        additional_outputs=[
            gr.Textbox(
                label="Chatbot thinking",
                interactive=False,
                lines=5,
            )
        ],
    )
    demo.launch()
