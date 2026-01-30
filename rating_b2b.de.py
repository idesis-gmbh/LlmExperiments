import concurrent.futures
import json
import sqlite3
from dbutils import store_projects, load_projects, store_page, load_page
from llmutils import embed_one, run_chat, run_chat_stream, load_json_response


def rate(item, model, debug=False):
    result_template = {
        "completeness_score": "<Gleitkommazahl zwischen 0.0 und 1.0>",
        "quality_rating": "<ausgezeichnet|gut|ausreichend|mangelhaft>",
        "missing_critical_attributes": "<Liste von Strings>",
        "missing_optional_attributes": "<Liste von Strings>",
        "inconsistencies": "<Liste von Strings oder leeres Array>",
        "data_quality_issues": "<Liste von Strings oder leeres Array>",
        "reasoning": "<kurze Erklärung der Bewertung>",
        "recommendations": "<Liste priorisierter Verbesserungsvorschläge>",
    }
    prompt = f"""
Bewerte eine Produktbeschreibung und antworte **ausschließlich** mit einem JSON-Objekt in genau diesem Format:
````{json.dumps(result_template)}```

Prüfe das Vorhandensein und die Qualität dieser Attribute:

KOMMERZIELLE DATEN:
- Produktcode/SKU/EAN
- Verpackungseinheit
- Mindestbestellmenge
- Lieferzeit/Verfügbarkeitsinformationen

LOGISTIKDATEN:
- Paketabmessungen (L x B x H)
- Paketgewicht
- Palettenkonfiguration
- Lageranforderungen
- Gefahrgutklassifizierung (falls zutreffend)
- Haltbarkeit

KLASSIFIZIERUNG:
- Produktkategorie/Hierarchie
- HS-/Zolltarifnummer
- Markenname

TECHNISCHE DATEN:
- Kerntechnische Produktspezifikationen
- Einhaltung von Vorschriften/Zertifizierungen
- Verweis auf Sicherheits-/technische Datenblätter

Bewerte die folgende Produktbeschreibung:
```{json.dumps(item)}```
"""
    try:
        response = run_chat_stream(prompt, model, debug)
        result = load_json_response(response)
        if result.get("quality_rating") not in [
            "ausgezeichnet",
            "gut",
            "ausreichend",
            "mangelhaft",
        ]:
            return {
                "completeness_score": 0.0,
                "quality_rating": "UNBEKANNT",
                "reasoning": "LLM hat keine oder eine ungültige Kategorie zurückgegeben",
                "raw_response": response,
            }
        return result
    except json.JSONDecodeError:
        return {
            "completeness_score": 0.0,
            "quality_rating": "UNBEKANNT",
            "reasoning": "Fehler beim Parsen der LLM-Antwort",
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
    with open("data/proartsubscriber.json", "r") as file:
        items = [
            item for item in json.load(file) if item["hersteller"]["name"] == "Ralston"
        ]
        # process_serial(projects["proart"], items, model)
        process_parallel(projects["proart"], items, model)
