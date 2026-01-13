import sys
from dbutils import (
    load_wikipedia_pageviews,
    scrape_wikipedia_pages,
    extract_wikipedia_sections,
    load_faiss,
    query_faiss,
    query_fts,
)
from llmutils import generate, generate_stream


def run_generate(prompt):
    print(prompt)
    status, thinking, content = generate(prompt)
    print(status, thinking, content)


def run_generate_stream(prompt):
    print(prompt)
    for status, thinking, content in generate_stream(prompt):
        assert status == 200
        if thinking:
            print(thinking, end="", flush=True)
        elif content:
            print(content, end="", flush=True)


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
        if "generate" in sys.argv[1]:
            run_generate(basic_prompt)
        if "generate_stream" in sys.argv[1]:
            run_generate_stream(basic_prompt)
        if "lookup" in sys.argv[1:]:
            print(basic_prompt)
            index = load_faiss()
            texts = query_faiss(index, basic_prompt)
            print(texts, flush=True)
        if "rag_generate" in sys.argv[1:]:
            index = load_faiss()
            texts = query_faiss(index, basic_prompt, k=5)
            print([texts[index][0] for index in range(len(texts))])
            rag_prompt = basic_prompt + "\nAdditional information from Wikipedia:\n"
            for text in texts:
                rag_prompt += text + "\n"
            run_generate(rag_prompt)
        if "rag_generate_stream" in sys.argv[1:]:
            index = load_faiss()
            texts = query_faiss(index, basic_prompt, k=5)
            print([texts[index][0] for index in range(len(texts))])
            rag_prompt = basic_prompt + "\nAdditional information from Wikipedia:\n"
            for text in texts:
                rag_prompt += text + "\n"
            run_generate_stream(rag_prompt)
