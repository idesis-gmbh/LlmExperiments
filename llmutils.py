import json
import urllib.request
from urllib.error import HTTPError


def embed_one(tip, document_or_query):
    # model = "nomic-embed-text"
    model = "bge-m3"
    # model = "qwen3-embedding"
    prompt = tip + document_or_query
    try:
        with urllib.request.urlopen(
            "http://localhost:11434/api/embeddings",
            data=json.dumps({"model": model, "prompt": prompt}).encode("utf-8"),
        ) as response:
            status = response.status
            answer = json.loads(response.read().decode("utf-8"))
            embedding = answer["embedding"]
    except HTTPError as e:
        status = e.status
        embedding = None
    return status, embedding


def embed_multiple(tip, documents_or_queries):
    # model = "nomic-embed-text"
    model = "bge-m3"
    # model = "qwen3-embedding"
    inputs = [tip + document_or_query for document_or_query in documents_or_queries]
    try:
        with urllib.request.urlopen(
            "http://localhost:11434/api/embed",
            data=json.dumps({"model": model, "input": inputs}).encode("utf-8"),
        ) as response:
            status = response.status
            answer = json.loads(response.read().decode("utf-8"))
            embeddings = answer["embeddings"]
    except HTTPError as e:
        status = e.status
        embeddings = None
    return status, embeddings


def chat_stream(system_prompt, user_prompt, tools=None):
    try:
        payload = {
            "model": "qwen3",
            "messages": [],
            "stream": True,
        }
        if system_prompt:
            payload["messages"].append(
                {
                    "role": "system",
                    "content": system_prompt,
                }
            )
        if user_prompt:
            payload["messages"].append(
                {
                    "role": "user",
                    "content": user_prompt,
                }
            )
        if tools:
            payload["tools"] = [tool["description"] for tool in tools]
        pending = False
        status = None
        done = False
        while pending or not done:
            with urllib.request.urlopen(
                "http://localhost:11434/api/chat",
                data=json.dumps(payload).encode("utf-8"),
            ) as response:
                pending = False
                status = response.status
                for line in response:
                    answer = json.loads(line)
                    message = answer["message"]
                    payload["messages"].append(message)
                    done = answer["done"]
                    if "thinking" in message:
                        assert not done and not message["content"]
                        yield status, message["thinking"], None
                    elif "content" in message:
                        assert "thinking" not in message
                        yield status, None, message["content"]
                    if "tool_calls" in message:
                        assert not done
                        for tool_call in message["tool_calls"]:
                            for tool in tools:
                                if (
                                    tool_call["function"]["name"]
                                    == tool["description"]["function"]["name"]
                                ):
                                    tools_return = tool["handler"](tool_call)
                                    payload["messages"].append(
                                        {
                                            "role": "tool",
                                            "tool_call_id": tool_call["id"],
                                            "content": json.dumps(tools_return),
                                        }
                                    )
                                    pending = True
                                    break
                print(pending, done)    
    except HTTPError as e:
        yield e.status, None, None


def chat(system_prompt, user_prompt, tools=None):
    try:
        payload = {
            "model": "qwen3",
            "messages": [],
            "stream": False,
        }
        if system_prompt:
            payload["messages"].append(
                {
                    "role": "system",
                    "content": system_prompt,
                }
            )
        if user_prompt:
            payload["messages"].append(
                {
                    "role": "user",
                    "content": user_prompt,
                }
            )
        if tools:
            payload["tools"] = [tool["description"] for tool in tools]
        pending = False
        status = None
        thinking = ""
        content = ""
        done = False
        while pending or not done:
            with urllib.request.urlopen(
                "http://localhost:11434/api/chat",
                data=json.dumps(payload).encode("utf-8"),
            ) as response:
                pending = False
                status = response.status
                # for line in response.read().decode("utf-8").splitlines():
                for line in response:
                    answer = json.loads(line)
                    message = answer["message"]
                    payload["messages"].append(message)
                    done = answer["done"]
                    if "thinking" in message:
                        thinking += message["thinking"]
                    if "content" in message:
                        content += message["content"]
                    if "tool_calls" in message:
                        for tool_call in message["tool_calls"]:
                            for tool in tools:
                                if (
                                    tool_call["function"]["name"]
                                    == tool["description"]["function"]["name"]
                                ):
                                    tools_return = tool["handler"](tool_call)
                                    payload["messages"].append(
                                        {
                                            "role": "tool",
                                            "tool_call_id": tool_call["id"],
                                            "content": json.dumps(tools_return),
                                        }
                                    )
                                    pending = True
                                    break
    except HTTPError as e:
        return e.status, None, None
    return status, thinking, content
