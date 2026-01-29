import concurrent.futures
import json
import sqlite3
from dbutils import store_projects, load_projects, store_page, load_page
from llmutils import embed_one, assemble_messages, chat, chat_stream


def run_chat(user_prompt, debug):
    if debug:
        print(user_prompt)
    messages = assemble_messages(None, user_prompt)
    content = ""
    for event in chat(messages):
        assert event["status"] == 200
        if event["type"] == "content":
            content += event["data"]
        if debug:
            print(event["type"], event["data"])
    return content


def run_chat_stream(user_prompt, debug):
    if debug:
        print(user_prompt)
    messages = assemble_messages(None, user_prompt)
    event_type = None
    content = ""
    for event in chat_stream(messages):
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


def classify(item, debug=False):
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
        response = run_chat_stream(prompt, debug)
        result = json.loads(response)
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


def prepare(item):
    if item["ist_nicht_mehr_lieferbar"]:
        return None
    markdown = load_page(connection, projects["proart"], item["ean"])
    if markdown:
        item = json.loads(markdown)
        if "bewertung_b2b" in item:
            if item["bewertung_b2b"]["quality_rating"] not in ["UNKNOWN"]:
                return None
            item.pop("bewertung_b2b")
    return item


def process_serial(items):
    for item in items:
        item = prepare(item)
        if item:
            item["bewertung_b2b"] = classify(item)
            print(item["bewertung_b2b"])
            store_page(connection, projects["proart"], item["ean"], json.dumps(item))


def process_parallel(items, max_workers=4):
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for item in items:
            item = prepare(item)
            if item:
                futures[executor.submit(classify, item)] = item
        for future in concurrent.futures.as_completed(futures):
            item = futures[future]
            try:
                item["bewertung_b2b"] = classify(item)
            except Exception as e:
                item["bewertung_b2b"] = {
                    "completeness_score": 0.0,
                    "quality_rating": "UNKNOWN",
                    "reasoning": "An error occurred during rating",
                    "raw_response": str(e),
                }
            print(item["bewertung_b2b"])
            store_page(connection, projects["proart"], item["ean"], json.dumps(item))


if __name__ == "__main__":
    with sqlite3.connect("data/rag.db") as connection:
        store_projects(connection, ["proart"])
        projects = load_projects(connection)
    # intro = "Check the following product description for incomplete or inconsistent information:"
    with open("data/proartsubscriber.json", "r") as file:
        items = json.load(file)
        # process_serial(items)
        process_parallel(items)
