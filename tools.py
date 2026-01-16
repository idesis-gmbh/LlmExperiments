import sys
from dbutils import (
    search_wikipedia_term,
    ingest_wikipedia_page,
    load_faiss,
    query_faiss,
    query_fts,
)
from llmutils import assemble_messages, chat, chat_stream
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


def run_chat(user_prompt, tools=None):
    system_prompt = SYSTEM_PROMPT if tools else None
    print(user_prompt)
    messages = assemble_messages(system_prompt, user_prompt)
    for event in chat(messages, tools):
        assert event["status"] == 200
        print(event["type"], event["data"])


def run_chat_stream(user_prompt, tools=None):
    system_prompt = SYSTEM_PROMPT if tools else None
    print(user_prompt)
    messages = assemble_messages(system_prompt, user_prompt)
    event_type = None
    for event in chat_stream(messages, tools):
        assert event["status"] == 200
        if event["type"] != event_type:
            print(f"{event['type']} ", end="")
            event_type = event["type"]
        print(event["data"], end="")


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
