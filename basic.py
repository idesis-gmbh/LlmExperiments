import sys
from llmutils import assemble_messages, chat, chat_stream


def run_chat(user_prompt):
    print(user_prompt)
    messages = assemble_messages(None, user_prompt)
    for event in chat(messages):
        assert event["status"] == 200
        print(event["type"], event["data"])


def run_chat_stream(user_prompt):
    print(user_prompt)
    messages = assemble_messages(None, user_prompt)
    event_type = None
    for event in chat_stream(messages):
        assert event["status"] == 200
        if event["type"] != event_type:
            print(f"{event['type']} ", end="")
            event_type = event["type"]
        print(event["data"], end="")


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        if sys.argv[1] == "chat":
            run_chat("Why sky is blue?")
        elif sys.argv[1] == "chat_stream":
            run_chat_stream("Why sky is blue?")
