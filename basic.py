import sys
from llmutils import chat, chat_stream


def run_chat(user_prompt):
    print(user_prompt)
    status, thinking, content = chat(None, user_prompt)
    print(status, thinking, content)


def run_chat_stream(user_prompt):
    print(user_prompt)
    for status, thinking, content in chat_stream(None, user_prompt):
        assert status == 200
        if thinking:
            print(thinking, end="", flush=True)
        elif content:
            print(content, end="", flush=True)


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        if sys.argv[1] == "chat":
            run_chat("Why sky is blue?")
        elif sys.argv[1] == "chat_stream":
            run_chat_stream("Why sky is blue?")
