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

Universal Attributes (all products):

- Product name and manufacturer/brand
- Primary use/application
- Coverage area or yield (e.g., m² per liter/kg)
- Package sizes available
- Drying/curing time
- Application temperature range
- Storage conditions and shelf life
- Color/finish options where applicable
- VOC content and environmental certifications
- Safety classifications and hazard warnings

Category-Specific Attributes:
Paints, Varnishes & Stains:

- Finish type (matt, satin, gloss) with description of look
- Number of coats needed
- Washability/durability for everyday life
- Odor level (low-odor, odorless)
- What surfaces it works on (wood, metal, plaster, etc.)
- Special properties (mold-resistant, child-safe, pet-friendly)
- Color matching/tinting availability

Primers & Surface Preparation:

- What problems it solves (covers stains, fills cracks, improves adhesion)
- Which surfaces it prepares
- What can be applied on top
- How easy to sand/work with
- When you need it vs. when you can skip it

Painting Tools & Equipment:

- Tool type and size
- What it's best used for
- Skill level required
- Durability/how many uses
- Cleaning instructions
- Compatible with which paint types

Solvents & Additives:

- What it does in simple terms
- Which products it works with
- How much to add/mixing instructions
- Safety precautions for home use
- Storage after opening

Personal Protective Equipment:

- What it protects against (fumes, splashes, dust)
- Comfort level for extended use
- Size guide
- Reusable or disposable
- Certification in plain language

Painting Accessories:

- What problem it solves
- How it makes the job easier
- Compatibility with other products
- Reusability

Wall Coverings & Wallpapers:

- Pattern/design style and dimensions
- Room suitability (kitchen, bathroom, bedroom)
- Installation difficulty
- Pre-pasted or requires adhesive
- Removability/strippable
- Washability/maintenance
- Pattern repeat and wastage guidance

Insulation & Weatherproofing:

- Energy savings potential
- Where to use it (attic, walls, pipes)
- Installation method (DIY-able or professional needed)
- Performance ratings in understandable terms
- Fire safety information
- Moisture resistance

Adhesives & Bonding Materials:

- What materials it bonds (wood to wood, plastic to metal, etc.)
- Setting time and final cure time
- Strength in practical terms ("holds up to X kg")
- Temperature and water resistance
- Gap-filling ability
- Repositionable or instant bond

Profiles, Trim & Molding:

- Dimensions and lengths available
- Material and finish
- Style/design aesthetic
- Pre-finished or needs painting
- Installation method and hardware needed
- Cutting/working instructions
- Room/application examples

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
        if "bewertung_b2c" in item:
            if item["bewertung_b2c"]["quality_rating"] not in ["UNKNOWN"]:
                return None
            item.pop("bewertung_b2c")
    return item


def process_serial(project, items, model):
    for item in items:
        item = prepare(project, item)
        if item:
            item["bewertung_b2c"] = rate(item, model, debug=True)
            print(item["bewertung_b2c"])
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
                item["bewertung_b2c"] = future.result()
            except Exception as e:
                item["bewertung_b2c"] = {
                    "completeness_score": 0.0,
                    "quality_rating": "UNKNOWN",
                    "reasoning": "An error occurred during rating",
                    "raw_response": str(e),
                }
            print(item["bewertung_b2c"])
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
