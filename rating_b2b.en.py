import concurrent.futures
import json
import sqlite3
from dbutils import store_projects, load_projects, store_page, load_page
from llmutils import embed_one, run_chat, run_chat_stream, load_json_response


def rate(item, model, debug=False):
    result_template = {
        "completeness_score": "<float between 0.0 and 1.0>",
        "quality_rating": "<excellent|good|adequate|poor>",
        "missing_critical_attributes": "<list of strings>",
        "missing_optional_attributes": "<list of strings>",
        "inconsistencies": "<list of strings or empty array>",
        "data_quality_issues": "<list of strings or empty array>",
        "reasoning": "<brief explanation of the assessment>",
        "recommendations": "<list of prioritized improvement suggestions>",
    }
    prompt = f"""
Rate a product description responding **only** with a JSON object in this exact format: 
```{json.dumps(result_template)}```

Check for the presence and quality of these attributes:

COMMERCIAL DATA:
- Product code/SKU/EAN
- Packaging unit specification
- Minimum order quantity
- Lead time/availability information

LOGISTICS DATA:
- Package dimensions (L x W x H)
- Package weight
- Pallet configuration
- Storage requirements
- Hazmat classification (if applicable)
- Shelf life

CLASSIFICATION:
- Product category/hierarchy
- HS/customs code
- Brand name

TECHNICAL DATA:
- Core product specifications
- Regulatory compliance/certifications
- Reference to safety/technical datasheets

Rate the following product description: 
```{json.dumps(item)}```
"""
    try:
        response = run_chat_stream(prompt, model, debug)
        result = load_json_response(response)
        if result.get("quality_rating") not in [
            "excellent",
            "good",
            "adequate",
            "poor",
        ]:
            return {
                "completeness_score": 0.0,
                "quality_rating": "UNKNOWN",
                "reasoning": "LLM returned no or invalid category",
                "raw_response": response,
            }
        return result
    except json.JSONDecodeError:
        return {
            "completeness_score": 0.0,
            "quality_rating": "UNKNOWN",
            "reasoning": "Failed to parse LLM response",
            "raw_response": response,
        }


def prepare(project, item):
    if item["ist_nicht_mehr_lieferbar"]:
        return None
    markdown = load_page(connection, project, item["ean"])
    if markdown:
        item = json.loads(markdown)
        if "bewertung_b2b" in item:
            if item["bewertung_b2b"]["quality_rating"] not in ["UNKNOWN"]:
                return None
            item.pop("bewertung_b2b")
    return item


def process_serial(project, items, model):
    for item in items:
        item = prepare(project, item)
        if item:
            item["bewertung_b2b"] = rate(item, model, debug=True)
            print(item["ean"])
            print(item["bewertung_b2b"])
            print()
            store_page(connection, project, item["ean"], json.dumps(item))


def process_parallel(project, items, model, max_workers=4):
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for item in items:
            item = prepare(project, item)
            if item:
                futures[executor.submit(rate, item, model)] = item
        for future in concurrent.futures.as_completed(futures):
            item = futures[future]
            try:
                item["bewertung_b2b"] = future.result()
            except Exception as e:
                item["bewertung_b2b"] = {
                    "completeness_score": 0.0,
                    "quality_rating": "UNKNOWN",
                    "reasoning": "An error occurred during rating",
                    "raw_response": str(e),
                }
            print(item["ean"])
            print(item["bewertung_b2b"])
            print()
            store_page(connection, project, item["ean"], json.dumps(item))


if __name__ == "__main__":
    model = "phi4-mini"
    # model = "phi4-mini-reasoning"
    # model = "gemma3n"
    # model = "gemma3"
    # model = "qwen3"
    with sqlite3.connect("data/rag.db") as connection:
        store_projects(connection, ["proart"])
        projects = load_projects(connection)
    # intro = "Check the following product description for incomplete or inconsistent information:"
    with open("data/proartsubscriber.json", "r") as file:
        items = [
            item for item in json.load(file) if item["hersteller"]["name"] == "Ralston"
        ]
        # process_serial(projects["proart"], items, model)
        process_parallel(projects["proart"], items, model)
