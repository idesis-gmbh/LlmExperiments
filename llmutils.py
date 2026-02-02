import json
import urllib.request
from urllib.error import HTTPError


def embed_one(tip, document_or_query, model="bge-m3"):
    # model = "nomic-embed-text"
    # model = "bge-m3"
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
    return {"type": "embedding", "status": status, "data": embedding}


def embed_multiple(tip, documents_or_queries, model="bge-m3"):
    # model = "nomic-embed-text"
    # model = "bge-m3"
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
    return {"type": "embeddings", "status": status, "data": embeddings}


def generate_stream(prompt, model="qwen3"):
    # model = "magistral"
    # model = "qwen3"
    # model = "gpt-oss"
    try:
        with urllib.request.urlopen(
            "http://localhost:11434/api/generate",
            data=json.dumps({"model": model, "prompt": prompt, "stream": True}).encode(
                "utf-8"
            ),
        ) as response:
            status = response.status
            for line in response:
                answer = json.loads(line)
                if "thinking" in answer:
                    assert not answer["response"]
                    yield {
                        "type": "thinking",
                        "status": status,
                        "data": answer["thinking"],
                    }
                elif "response" in answer:
                    assert "thinking" not in answer
                    yield {
                        "type": "response",
                        "status": status,
                        "data": answer["response"],
                    }
    except HTTPError as e:
        yield {"type": "error", "status": status, "data": None}


def generate(prompt, model="qwen3"):
    # model = "magistral"
    # model = "qwen3"
    # model = "gpt-oss"
    try:
        with urllib.request.urlopen(
            "http://localhost:11434/api/generate",
            data=json.dumps({"model": model, "prompt": prompt, "stream": False}).encode(
                "utf-8"
            ),
        ) as response:
            status = response.status
            thinking = ""
            content = ""
            for line in response:
                answer = json.loads(line)
                if "thinking" in answer:
                    thinking += answer["thinking"]
                if "response" in answer:
                    content += answer["response"]
    except HTTPError as e:
        return [{"status": status, "type": "error", "data": None}]
    return [
        {"type": "thinking", "status": status, "data": thinking},
        {"type": "content", "status": status, "data": content},
    ]


def assemble_messages(system_prompt, user_prompt):
    messages = []
    if system_prompt:
        messages.append(
            {
                "role": "system",
                "content": system_prompt,
            }
        )
    if user_prompt:
        messages.append(
            {
                "role": "user",
                "content": user_prompt,
            }
        )
    return messages


def chat_stream(messages, model="qwen3", think=None, format=None, tools=None):
    # model = "magistral"
    # model = "qwen3"
    # model = "gpt-oss"
    try:
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        if think:
            payload["think"] = think
        if format:
            payload["format"] = format
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
                event_type = None
                for line in response:
                    answer = json.loads(line)
                    message = answer["message"]
                    done = answer["done"]
                    # if done:
                    #    print(answer)
                    if "thinking" in message:
                        assert not done and not message["content"]
                        # if event_type != "thinking":
                        #     payload["messages"].append(message)
                        #     event_type = "thinking"
                        # else:
                        #     payload["messages"][-1]["thinking"] += message["thinking"]
                        yield {
                            "type": "thinking",
                            "status": status,
                            "data": message["thinking"],
                        }
                    elif "content" in message:
                        assert "thinking" not in message
                        if event_type != "content":
                            payload["messages"].append(message)
                            event_type = "content"
                        else:
                            payload["messages"][-1]["content"] += message["content"]
                        yield {
                            "type": "content",
                            "status": status,
                            "data": message["content"],
                        }
                    if "tool_calls" in message:
                        # assert not done
                        if event_type != "tooling":
                            payload["messages"].append(message)
                            event_type = "tooling"
                        else:
                            payload["messages"][-1]["tool_calls"].append(
                                message["tool_calls"]
                            )
                        tooling = []
                        for tool_call in message["tool_calls"]:
                            for tool in tools:
                                if (
                                    tool_call["function"]["name"]
                                    == tool["description"]["function"]["name"]
                                ):
                                    tool_return = tool["handler"](tool_call)
                                    tooling.append(
                                        {"call": tool_call, "return": tool_return}
                                    )
                                    payload["messages"].append(
                                        {
                                            "role": "tool",
                                            "tool_call_id": tool_call["id"],
                                            "content": json.dumps(tool_return),
                                        }
                                    )
                                    pending = True
                                    break
                        yield {
                            "type": "tooling",
                            "status": status,
                            "data": json.dumps(tooling),
                        }
    except HTTPError as e:
        print("HTTPError", e)
        yield {"type": "error", "status": status, "data": None}


def chat(messages, model="qwen3", think=None, format=None, tools=None):
    # model = "magistral"
    # model = "qwen3"
    # model = "gpt-oss"
    try:
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        if think:
            payload["think"] = think
        if format:
            payload["format"] = format
        if tools:
            payload["tools"] = [tool["description"] for tool in tools]
        pending = False
        status = None
        thinking = ""
        content = ""
        tooling = []
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
                    done = answer["done"]
                    payload["messages"].append(message)
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
                                    tool_return = tool["handler"](tool_call)
                                    tooling.append(
                                        {
                                            "call": tool_call,
                                            "return": tool_return,
                                        }
                                    )
                                    payload["messages"].append(
                                        {
                                            "role": "tool",
                                            "tool_call_id": tool_call["id"],
                                            "content": json.dumps(tool_return),
                                        }
                                    )
                                    pending = True
                                    break
    except HTTPError as e:
        return [{"status": status, "type": "error", "data": None}]
    return [
        {"type": "thinking", "status": status, "data": thinking},
        {"type": "tooling", "status": status, "data": json.dumps(tooling)},
        {"type": "content", "status": status, "data": content},
    ]


def run_chat(user_prompt, model, think, format, debug):
    if debug:
        print(user_prompt)
    messages = assemble_messages(None, user_prompt)
    content = ""
    for event in chat(messages, model=model, think=think, format=format):
        assert event["status"] == 200
        if event["type"] == "content":
            content += event["data"]
        if debug:
            print(event["type"], event["data"])
    return content


def run_chat_stream(user_prompt, model, think, format, debug):
    if debug:
        print(user_prompt)
    messages = assemble_messages(None, user_prompt)
    event_type = None
    content = ""
    for event in chat_stream(messages, model=model, think=think, format=format):
        assert event["status"] == 200
        if event["type"] != event_type:
            if debug:
                print(f"{event['type']} ", end="")
            event_type = event["type"]
        if debug:
            print(event["data"], end="")
        if event["type"] == "content":
            content += event["data"]
    return content


def load_json_response(response):
    if response.startswith("```"):
        response = response[3:]
        if response.startswith("json"):
            response = response[4:]
        if response.endswith("```"):
            response = response[:-3]
    return json.loads(response)
