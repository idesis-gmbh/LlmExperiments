import json
from urllib.error import HTTPError
import urllib.request
import sys
from dbutils import (
    load_wikipedia_pageviews,
    scrape_wikipedia_pages,
    extract_wikipedia_sections,
    load_faiss,
    query_rag,
)
from llmutils import embed_one, embed_multiple


def generate(prompt):
    try:
        with urllib.request.urlopen(
            "http://localhost:11434/api/generate",
            data=json.dumps({"model": "qwen3", "prompt": prompt}).encode("utf-8"),
        ) as response:
            status = response.status
            thinking = ""
            result = ""
            for line in response.read().decode("utf-8").splitlines():
                answer = json.loads(line)
                if "thinking" in answer:
                    thinking += answer["thinking"]
                else:
                    result += answer["response"]
    except HTTPError as e:
        status = e.status
        thinking = None
        result = None
    return status, thinking, result


if __name__ == "__main__":
    if "load_wikipedia_pageviews" in sys.argv[1:]:
        load_wikipedia_pageviews("data/pageviews-202511-user.bz2")
        # load_wikipedia_pageviews("data/pageviews-20251201-user.bz2")
    if "scrape_wikipedia_pages" in sys.argv[1:]:
        scrape_wikipedia_pages(100)
    if "extract_wikipedia_sections" in sys.argv[1:]:
        extract_wikipedia_sections()
    for basic_prompt in [
        "Tell me about Google Chrome."
        # "What year was the Berlin Wall built, and which countries were involved in its construction?",
        # "Who discovered penicillin, and how was the discovery made?",
        # "What is the chemical formula of ozone, and how does it differ from oxygen?",
        # "Who was the first woman to win a Nobel Prize, and in which field?",
        # "What is the capital of Mongolia, and what was its former name?",
    ]:
        if "simple" in sys.argv[1]:
            print("Basic prompt:", basic_prompt, flush=True)
            status, thinking, result = generate(basic_prompt)
            print("Thinking about basic prompt:", thinking, flush=True)
            print("Answer to basic prompt:", result, flush=True)
        if "lookup" in sys.argv[1:]:
            print("Basic prompt:", basic_prompt, flush=True)
            index = load_faiss()
            texts = query_rag(index, basic_prompt)
            print(texts, flush=True)
        if "rag" in sys.argv[1:]:
            print("Basic prompt:", basic_prompt, flush=True)
            index = load_faiss()
            result = query_rag(index, basic_prompt, k=5)
            print(
                "RAG distance:",
                [result[index][0] for index in range(len(result))],
                flush=True,
            )
            rag_prompt = basic_prompt + "\nAdditional information from Wikipedia:\n"
            for text in texts:
                rag_prompt += text + "\n"
            print("RAG prompt:", rag_prompt, flush=True)
            status, thinking, result = generate(rag_prompt)
            print("Thinking about RAG prompt:", thinking, flush=True)
            print("Answer to RAG prompt:", result, flush=True)
