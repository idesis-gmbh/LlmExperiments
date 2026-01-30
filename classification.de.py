import concurrent.futures
import json
import sqlite3
from dbutils import store_projects, load_projects, store_page, load_page
from llmutils import embed_one, run_chat, run_chat_stream, load_json_response


def classify(item, model, debug=False):
    product_categories = [
        "Farben, Lacke & Lasuren",
        "Grundierungen & Oberflächenvorbereitung",
        "Malerwerkzeuge & Geräte",
        "Lösemittel & Zusatzstoffe",
        "Persönliche Schutzausrüstung",
        "Malerzubehör",
        "Wandbeläge & Tapeten",
        "Isolierung & Wetterschutz",
        "Klebstoffe & Verbindungsmaterialien",
        "Profile, Leisten & Zierleisten",
    ]
    result_template = {
        "category": "<Kategoriename>",
        "confidence": "<ein Wert zwischen 0.0 und 1.0>",
        "reasoning": "<kurze Erklärung>",
        "proposed_new_category": "<Name der neuen Kategorie falls nötig, ansonsten null>",
    }
    prompt = f"""
Gegeben sind die folgenden Produktkategorien:
`````{json.dumps(product_categories)}```
Klassifiziere eine Produktbeschreibung und antworte **ausschließlich** mit einem JSON-Objekt in genau diesem Format:
````{json.dumps(result_template)}```

Bitte beachte:
- Wenn das Produkt wirklich in **keine** bestehende Kategorie passt, setze category auf "PROPOSED_NEW_CATEGORY"
- Beim Vorschlagen einer neuen Kategorie gib einen klaren, beschreibenden Namen an, der auf mehrere ähnliche Produkte anwendbar ist
- Halte reasoning kurz auf einen einzigen Satz

Klassifiziere die folgende Produktbeschreibung:
```{json.dumps(item)}```
"""
    try:
        response = run_chat_stream(prompt, model, debug)
        result = load_json_response(response)
        if result.get("category") not in product_categories + [
            "PROPOSED_NEW_CATEGORY",
            "UNKNOWN",
        ]:
            return {
                "category": "UNKNOWN",
                "confidence": 0.0,
                "reasoning": "LLM hat keine oder eine ungültige Kategorie zurückgegeben",
                "proposed_new_category": None,
                "raw_response": response,
            }
        return result
    except json.JSONDecodeError:
        return {
            "category": "UNKNOWN",
            "confidence": 0.0,
            "reasoning": "Fehler beim Parsen der LLM-Antwort",
            "proposed_new_category": None,
            "raw_response": response,
        }


def prepare(project, item):
    if item["ist_nicht_mehr_lieferbar"]:
        return None
    markdown = load_page(connection, project, item["ean"])
    if markdown:
        item = json.loads(markdown)
        if "klassifikation" in item:
            if item["klassifikation"]["category"] not in [
                "PROPOSED_NEW_CATEGORY",
                "UNKNOWN",
            ]:
                return None
            item.pop("klassifikation")
    return item


def process_serial(project, items, model):
    for item in items:
        item = prepare(project, item)
        if item:
            item["klassifikation"] = classify(item, model, debug=True)
            print(item["ean"])
            print(item["klassifikation"])
            print()
            store_page(connection, project, item["ean"], json.dumps(item))


def process_parallel(project, items, model, max_workers=4):
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for item in items:
            item = prepare(project, item)
            if item:
                futures[executor.submit(classify, item, model)] = item
        for future in concurrent.futures.as_completed(futures):
            item = futures[future]
            try:
                item["klassifikation"] = future.result()
            except Exception as e:
                item["klassifikation"] = {
                    "category": "UNKNOWN",
                    "confidence": 0.0,
                    "reasoning": "An error occurred during classification",
                    "proposed_new_category": None,
                    "raw_response": str(e),
                }
            print(item["ean"])
            print(item["klassifikation"])
            print()
            store_page(connection, project, item["ean"], json.dumps(item))


if __name__ == "__main__":
    model = "phi4-mini"  # OKish, doesn't respect PROPOSED_NEW_CATEGORY?
    # model = "phi4-mini-reasoning" # Thinking not correctly configured/implemented by Ollama?
    # model = "gemma3n" # Suspect tokenizer bugs
    # model = "gemma3" # Suspect tokenizer bugs
    # model = "qwen3"
    with sqlite3.connect("data/rag.db") as connection:
        store_projects(connection, ["proart"])
        projects = load_projects(connection)
    with open("data/proartsubscriber.json", "r") as file:
        items = [
            item for item in json.load(file) if item["hersteller"]["name"] == "Ralston"
        ]
        # process_serial(projects["proart"], items, model)
        process_parallel(projects["proart"], items, model)
