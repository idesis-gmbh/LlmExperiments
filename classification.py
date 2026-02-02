import json
import sqlite3
from dbutils import load_projects, load_pages, store_page
from proartutils import (
    assemble_classification_prompt_en,
    evaluate_classification_response_en,
    evaluate_classification_error_en,
    classify_serial,
    classify_parallel,
)


if __name__ == "__main__":
    # think = None
    # model = "phi4-mini"  # OKish
    # model = "phi4-mini-reasoning" # Thinking not correctly configured/implemented by Ollama?
    # model = "gemma3n" # Suspect tokenizer bugs
    # model = "gemma3" # Suspect tokenizer bugs
    # model = "ministral-3"
    # model = "magistral"
    # model = "deepseek-r1"
    model = "qwen3"
    think = True
    with sqlite3.connect("data/rag.db") as connection:
        projects = load_projects(connection)
        items = [
            json.loads(page) for page in load_pages(connection, projects["proart"])
        ]
        """
        classify_serial(
            assemble_classification_prompt_en,
            evaluate_classification_response_en,
            connection,
            projects["proart"],
            items,
            model,
            think,
        )
        """
        classify_parallel(
            assemble_classification_prompt_en,
            evaluate_classification_response_en,
            evaluate_classification_error_en,
            connection,
            projects["proart"],
            items,
            model,
            think,
        )
