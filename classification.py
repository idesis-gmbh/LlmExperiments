import json
import sqlite3
from dbutils import store_projects, load_projects, store_page, load_page
from llmutils import embed_one, assemble_messages, chat, chat_stream


def run_chat(user_prompt):
    print(user_prompt)
    messages = assemble_messages(None, user_prompt)
    for event in chat(messages):
        assert event["status"] == 200
        print(event["type"], event["data"])


def run_chat_stream(user_prompt):
    print(user_prompt)
    messages = assemble_messages(None, user_prompt)
    event_type = None
    content = ""
    for event in chat_stream(messages):
        assert event["status"] == 200
        if event["type"] != event_type:
            print(f"{event['type']} ", end="")
            event_type = event["type"]
        print(event["data"], end="")
        if event["type"] == "content":
            content += event["data"]
    return content


def classify(item):
    product_groups = [
        "Coatings & Finishes",
        "Surface Preparation",
        "Application Tools",
        "Thinners & Additives",
        "Protective Equipment",
        "Ancillary Products",
        "Wall Coverings & Decorative Materials",
        "Insulation & Weatherproofing",         
        "Adhesives & Bonding Materials",        
        "Structural Profiles & Components",     
    ]
    result_template = {
        "category": "<category name>",
        "confidence": "<a value between 0.0 to 1.0>",
        "reasoning": "<brief explanation>",
        "proposed_new_group": "<name of new group if needed, otherwise null>",
    }
    prompt = f"""Classify the following product into ONE of these categories:
{json.dumps(product_groups)}

Product to classify:
{json.dumps(item)}

Respond ONLY with a JSON object in this exact format:
{json.dumps(result_template)}

Rules:
- Try to use an existing category if reasonably appropriate
- If the product genuinely doesn't fit ANY existing category, set category to "PROPOSED_NEW_GROUP"
- When proposing a new group, provide a clear, descriptive name that could apply to multiple similar products
- Only propose new groups when truly necessary (not just for slight variations)
- Set confidence below 0.7 if uncertain
- Keep reasoning concise (one sentence)

Examples:
- Acrylic paint → {{"category": "Coatings & Finishes", "confidence": 0.95, "reasoning": "Interior coating", "proposed_new_group": null}}
- Sandpaper → {{"category": "Surface Preparation", "confidence": 0.95, "reasoning": "Abrasive material", "proposed_new_group": null}}
- Paint roller → {{"category": "Application Tools", "confidence": 0.95, "reasoning": "Paint application tool", "proposed_new_group": null}}
- White spirit → {{"category": "Thinners & Additives", "confidence": 0.95, "reasoning": "Solvent", "proposed_new_group": null}}
- Safety goggles → {{"category": "Protective Equipment", "confidence": 0.95, "reasoning": "PPE", "proposed_new_group": null}}
- Masking tape → {{"category": "Ancillary Products", "confidence": 0.95, "reasoning": "Painting accessory", "proposed_new_group": null}}
- Vinyl wallpaper → {{"category": "Wall Coverings & Decorative Materials", "confidence": 0.95, "reasoning": "Wall covering", "proposed_new_group": null}}
- Mineral wool → {{"category": "Insulation & Weatherproofing", "confidence": 0.95, "reasoning": "Insulation material", "proposed_new_group": null}}
- Construction adhesive → {{"category": "Adhesives & Bonding Materials", "confidence": 0.95, "reasoning": "Bonding agent", "proposed_new_group": null}}
- Edge profile → {{"category": "Structural Profiles & Components", "confidence": 0.95, "reasoning": "Structural component", "proposed_new_group": null}}
- Power drill → {{"category": "PROPOSED_NEW_GROUP", "confidence": 0.85, "reasoning": "Power tool", "proposed_new_group": "Power Tools & Equipment"}}
"""
    try:
        response = run_chat_stream(prompt)
        result = json.loads(response)
        if result.get("category") not in product_groups + [
            "PROPOSED_NEW_GROUP",
            "UNKNOWN",
        ]:
            return {
                "category": "UNKNOWN",
                "confidence": 0.0,
                "reasoning": "LLM returned no or invalid category",
                "proposed_new_group": None,
                "raw_response": response,
            }
        return result
    except json.JSONDecodeError:
        return {
            "category": "UNKNOWN",
            "confidence": 0.0,
            "reasoning": "Failed to parse LLM response",
            "proposed_new_group": None,
            "raw_response": response,
        }


def needs_review(classification_result):
    """Determine if classification needs human review"""
    return (
        classification_result["category"] in ["PROPOSED_NEW_GROUP", "UNKNOWN"]
        or classification_result["confidence"] < 0.7
    )


def get_proposed_groups_summary(classifications):
    """Analyze proposed new groups across all classifications"""
    from collections import Counter

    proposals = [
        c["proposed_new_group"] for c in classifications if c.get("proposed_new_group")
    ]

    return Counter(proposals)


if __name__ == "__main__":
    with sqlite3.connect("data/rag.db") as connection:
        store_projects(connection, ["proart"])
        projects = load_projects(connection)
    # intro = "Check the following product description for incomplete or inconsistent information:"
    with open("data/proartsubscriber.json", "r") as file:
        data = json.load(file)
        for index, item in enumerate(data):
            if item["ist_nicht_mehr_lieferbar"]:
                continue
            markdown = load_page(connection, projects["proart"], item["ean"])
            if markdown:
                item = json.loads(markdown)
                if item["klassifikation"]["category"] != "PROPOSED_NEW_GROUP":                    
                    continue
                item.pop["klassifikation"]
            item["klassifikation"] = classify(item)
            store_page(connection, projects["proart"], item["ean"], json.dumps(item))

