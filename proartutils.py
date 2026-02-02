import concurrent.futures
import json
from dbutils import load_projects, load_pages, store_page
from llmutils import embed_one, run_chat, run_chat_stream, load_json_response


def assemble_classification_prompt_de(item):
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
        "proposed_category": "<Name der neuen Kategorie falls nötig, ansonsten null>",
    }
    prompt = f"""
Gegeben sind die folgenden Produktkategorien:
`````{json.dumps(product_categories)}```
Klassifiziere eine Produktbeschreibung und antworte **ausschließlich** mit einem JSON-Objekt in genau diesem Format:
````{json.dumps(result_template)}```

Bitte beachte:
- Wenn das Produkt wirklich in **keine** bestehende Kategorie passt, setze category auf "null" und schlage eine neue Kategorie in "proposed_category" vor 
- Beim Vorschlagen einer neuen Kategorie gib einen klaren, beschreibenden Namen an, der auf mehrere ähnliche Produkte anwendbar ist
- Halte reasoning kurz auf einen einzigen Satz

Klassifiziere die folgende Produktbeschreibung:
```{json.dumps(item)}```
"""
    return product_categories, prompt


def evaluate_classification_response_de(product_categories, response):
    try:
        result = load_json_response(response)
        if result.get("category") not in product_categories + [
            None,
            "UNKNOWN",
        ]:
            return {
                "category": "UNKNOWN",
                "confidence": 0.0,
                "reasoning": "LLM hat keine oder eine ungültige Kategorie zurückgegeben",
                "proposed_category": None,
                "raw_response": response,
            }
        return result
    except json.JSONDecodeError as error:
        return {
            "category": "UNKNOWN",
            "confidence": 0.0,
            "reasoning": f"Fehler beim Parsen der LLM-Antwort: {error}",
            "proposed_category": None,
            "raw_response": response,
        }


def evaluate_classification_error_de(error):
    return {
        "category": "UNKNOWN",
        "confidence": 0.0,
        "reasoning": "Ein Fehler ist während der Klassifizierung aufgetreten",
        "proposed_category": None,
        "raw_response": str(error),
    }


def assemble_classification_prompt_en(item):
    product_categories = [
        "Paints, Varnishes & Stains",
        "Primers & Surface Preparation",
        "Painting Tools & Equipment",
        "Solvents & Additives",
        "Personal Protective Equipment",
        "Painting Accessories",
        "Wall Coverings & Wallpapers",
        "Insulation & Weatherproofing",
        "Adhesives & Bonding Materials",
        "Profiles, Trim & Molding",
    ]
    result_template = {
        "category": "<category name>",
        "confidence": "<a value between 0.0 to 1.0>",
        "reasoning": "<brief explanation>",
        "proposed_category": "<name of new category if needed, otherwise null>",
    }
    prompt = f"""
Given the following product categories 
```{json.dumps(product_categories)}```
classify a product description responding **only** with a JSON object in this exact format: 
```{json.dumps(result_template)}```

Please note:
- If the product genuinely doesn't fit **any** existing category, set category to "null" and propose a new category in "proposed_category"
- When proposing a new category, provide a clear, descriptive name that could apply to multiple similar products
- Keep reasoning concise to a single sentence

Classify the following product description: 
```{json.dumps(item)}```
"""
    return product_categories, prompt


def evaluate_classification_response_en(product_categories, response):
    try:
        result = load_json_response(response)
        if result.get("category") not in product_categories + [
            None,
            "UNKNOWN",
        ]:
            return {
                "category": "UNKNOWN",
                "confidence": 0.0,
                "reasoning": "LLM returned no or invalid category",
                "proposed_category": None,
                "raw_response": response,
            }
        return result
    except json.JSONDecodeError as error:
        return {
            "category": "UNKNOWN",
            "confidence": 0.0,
            "reasoning": f"Failed to parse LLM response: {error}",
            "proposed_category": None,
            "raw_response": response,
        }


def evaluate_classification_error_en(error):
    return {
        "category": "UNKNOWN",
        "confidence": 0.0,
        "reasoning": "An error occurred during classification",
        "proposed_category": None,
        "raw_response": str(error),
    }


def classify(
    assemble_prompt,
    evaluate_response,
    item,
    model,
    think,
    debug=False,
):
    product_categories, prompt = assemble_prompt(item)
    response = run_chat_stream(prompt, model, think, "json", debug)
    return evaluate_response(product_categories, response)


def assemble_b2c_rating_prompt_de(item):
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

Universelle Attribute (alle Produkte):

- Produktname und Hersteller/Marke
- Hauptverwendung/Anwendung
- Reichweite oder Ergiebigkeit (z.B. m² pro Liter/kg)
- Verfügbare Verpackungsgrößen
- Trocknungs-/Aushärtezeit
- Anwendungstemperaturbereich
- Lagerbedingungen und Haltbarkeit
- Farb-/Oberflächenoptionen, falls zutreffend
- VOC-Gehalt und Umweltzertifizierungen
- Sicherheitsklassifizierungen und Gefahrenhinweise

Kategoriespezifische Attribute:
Farben, Lacke & Lasuren:

- Oberflächentyp (matt, seidenmatt, glänzend) mit Beschreibung des Aussehens
- Anzahl benötigter Anstriche
- Waschbarkeit/Strapazierfähigkeit für den Alltag
- Geruchsstärke (geruchsarm, geruchlos)
- Geeignete Untergründe (Holz, Metall, Putz, etc.)
- Besondere Eigenschaften (schimmelresistent, kindersicher, haustierfreundlich)
- Farbmischung/Abtönbarkeit

Grundierungen & Oberflächenvorbereitung:

- Welche Probleme es löst (überdeckt Flecken, füllt Risse, verbessert Haftung)
- Welche Oberflächen es vorbereitet
- Was darauf aufgetragen werden kann
- Wie leicht zu schleifen/zu verarbeiten
- Wann man es braucht vs. wann man darauf verzichten kann

Malerwerkzeuge & -geräte:

- Werkzeugtyp und Größe
- Wofür es am besten geeignet ist
- Erforderliches Fertigkeitsniveau
- Haltbarkeit/Anzahl der Verwendungen
- Reinigungsanleitung
- Kompatibel mit welchen Farbtypen

Lösemittel & Zusatzstoffe:

- Was es in einfachen Worten bewirkt
- Mit welchen Produkten es funktioniert
- Wie viel hinzuzufügen ist/Mischanleitung
- Sicherheitsvorkehrungen für den Heimgebrauch
- Lagerung nach dem Öffnen

Persönliche Schutzausrüstung:

- Wogegen es schützt (Dämpfe, Spritzer, Staub)
- Tragekomfort bei längerem Gebrauch
- Größentabelle
- Wiederverwendbar oder Einweg
- Zertifizierung in einfacher Sprache

Malerzubehör:

- Welches Problem es löst
- Wie es die Arbeit erleichtert
- Kompatibilität mit anderen Produkten
- Wiederverwendbarkeit

Wandbeläge & Tapeten:

- Muster-/Designstil und Abmessungen
- Raumeignung (Küche, Bad, Schlafzimmer)
- Schwierigkeitsgrad bei der Anbringung
- Vorkleister oder Kleister erforderlich
- Abziehbarkeit/ablösbar
- Waschbarkeit/Pflege
- Musterrapport und Verschnitt-Hinweise

Isolierung & Wetterschutz:

- Energieeinsparpotenzial
- Wo es verwendet wird (Dachboden, Wände, Rohre)
- Installationsmethode (selbst machbar oder Fachmann erforderlich)
- Leistungskennzahlen in verständlichen Begriffen
- Brandschutzinformationen
- Feuchtigkeitsbeständigkeit

Klebstoffe & Verbindungsmaterialien:

- Welche Materialien es verbindet (Holz zu Holz, Kunststoff zu Metall, etc.)
- Abbindezeit und endgültige Aushärtezeit
- Festigkeit in praktischen Begriffen („hält bis zu X kg")
- Temperatur- und Wasserbeständigkeit
- Spaltfüllvermögen
- Repositionierbar oder Sofortbindung

Profile, Leisten & Zierleisten:

- Verfügbare Abmessungen und Längen
- Material und Oberfläche
- Stil/Designästhetik
- Fertig beschichtet oder lackierbedürftig
- Installationsmethode und benötigte Befestigung
- Zuschnitt-/Verarbeitungshinweise
- Raum-/Anwendungsbeispiele

Bitte beachte:
- Prüfe, ob Beschreibung, Bild, technisches und Sicherheitsdatenblatt verfügbar sind
- Wesentliche Informationen (was es bewirkt, wie zu verwenden, Reichweite/Menge): fehlend = große Abwertung
- Wichtige Informationen (Trocknungszeit, geeignete Untergründe, Schwierigkeitsgrad): fehlend = mittlere Abwertung
- Wünschenswerte Informationen (Tipps, ergänzende Produkte): fehlend = geringe Abwertung
- Mengenangaben sollten Einheiten haben
- Prüfe auf Widersprüche

Bewerte die folgende Produktbeschreibung:
```{json.dumps(item)}```
"""
    return prompt


def evaluate_b2c_rating_response_de(response):
    try:
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
                "reasoning": "LLM hat keine oder eine ungültige Bewertung zurückgegeben",
                "raw_response": response,
            }
        return result
    except json.JSONDecodeError as error:
        return {
            "completeness_score": 0.0,
            "quality_rating": "UNBEKANNT",
            "reasoning": f"Fehler beim Parsen der LLM-Antwort: {error}",
            "raw_response": response,
        }


def evaluate_b2c_rating_error_de(error):
    return {
        "completeness_score": 0.0,
        "quality_rating": "UNBEKANNT",
        "reasoning": "Ein Fehler ist während der Bewertung aufgetreten",
        "raw_response": str(error),
    }


def assemble_b2c_rating_prompt_en(item):
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

Check for the presence, quality and consistency of these attributes:

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

Please note:
- Assess if description, picture, technical and safety datasheet are available
- Essential info (what it does, how to use, coverage/quantity): missing = major penalty
- Important info (drying time, suitable surfaces, difficulty level): missing = moderate penalty
- Nice-to-have (tips, complementary products): missing = minor penalty
- Quantities should have units
- Check for contradictions

Rate the following product description: 
```{json.dumps(item)}```
"""
    return prompt


def evaluate_b2c_rating_response_en(response):
    try:
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
                "reasoning": "LLM returned no or invalid quality rating",
                "raw_response": response,
            }
        return result
    except json.JSONDecodeError as error:
        return {
            "completeness_score": 0.0,
            "quality_rating": "UNKNOWN",
            "reasoning": f"Failed to parse LLM response: {error}",
            "raw_response": response,
        }


def evaluate_b2c_rating_error_en(error):
    return {
        "completeness_score": 0.0,
        "quality_rating": "UNKNOWN",
        "reasoning": "An error occurred during the rating",
        "raw_response": str(error),
    }


def rate_b2c(assemble_prompt, evaluate_response, item, model, think, debug=False):
    prompt = assemble_prompt(item)
    response = run_chat_stream(prompt, model, think, "json", debug)
    return evaluate_response(response)


def assemble_b2b_rating_prompt_de(item):
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

Bitte beachte:
- Prüfe, ob Beschreibung, Bild, technisches und Sicherheitsdatenblatt verfügbar sind
- Wesentliche Informationen (was es bewirkt, wie zu verwenden, Reichweite/Menge): fehlend = große Abwertung
- Wichtige Informationen (Trocknungszeit, geeignete Untergründe, Schwierigkeitsgrad): fehlend = mittlere Abwertung
- Wünschenswerte Informationen (Tipps, ergänzende Produkte): fehlend = geringe Abwertung
- Mengenangaben sollten Einheiten haben
- Prüfe auf Widersprüche

Bewerte die folgende Produktbeschreibung:
```{json.dumps(item)}```
"""
    return prompt


def evaluate_b2b_rating_response_de(response):
    try:
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
                "reasoning": "LLM hat keine oder eine ungültige Bewertung zurückgegeben",
                "raw_response": response,
            }
        return result
    except json.JSONDecodeError as error:
        return {
            "completeness_score": 0.0,
            "quality_rating": "UNBEKANNT",
            "reasoning": f"Fehler beim Parsen der LLM-Antwort: {error}",
            "raw_response": response,
        }


def evaluate_b2b_rating_error_de(error):
    return {
        "completeness_score": 0.0,
        "quality_rating": "UNBEKANNT",
        "reasoning": "Ein Fehler ist während der Bewertung aufgetreten",
        "raw_response": str(error),
    }


def assemble_b2b_rating_prompt_en(item):
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

Check for the presence, quality and consistency of these attributes:

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

Please note:
- Assess if description, picture, technical and safety datasheet are available
- Essential info (product code/SKU/EAN, packaging unit, package dimensions (L×W×H), package weight): missing = major penalty
- Important info (pallet configuration, storage requirements, shelf life, HS/customs code, hazmat classification): missing = moderate penalty
- Nice-to-have (MOQ, lead time, seasonal demand notes): missing = minor penalty
- Quantities should have units
- Check for contradictions

Rate the following product description: 
```{json.dumps(item)}```
"""
    return prompt


def evaluate_b2b_rating_response_en(response):
    try:
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
                "reasoning": "LLM returned no or invalid quality rating",
                "raw_response": response,
            }
        return result
    except json.JSONDecodeError as error:
        return {
            "completeness_score": 0.0,
            "quality_rating": "UNKNOWN",
            "reasoning": f"Failed to parse LLM response: {error}",
            "raw_response": response,
        }


def evaluate_b2b_rating_error_en(error):
    return {
        "completeness_score": 0.0,
        "quality_rating": "UNKNOWN",
        "reasoning": "An error occurred during the rating",
        "raw_response": str(error),
    }


def rate_b2b(assemble_prompt, evaluate_response, item, model, think, debug=False):
    prompt = assemble_prompt(item)
    response = run_chat_stream(prompt, model, think, "json", debug)
    return evaluate_response(response)


def prepare(item, model):
    # if item["ist_nicht_mehr_lieferbar"]:
    #     return None
    if "klassifikation" not in item:
        item["klassifikation"] = {}
    if model not in item["klassifikation"]:
        item["klassifikation"][model] = []
    if "bewertung_b2c" not in item:
        item["bewertung_b2c"] = {}
    if model not in item["bewertung_b2c"]:
        item["bewertung_b2c"][model] = []
    if "bewertung_b2b" not in item:
        item["bewertung_b2b"] = {}
    if model not in item["bewertung_b2b"]:
        item["bewertung_b2b"][model] = []
    return item


def follow_up(connection, project, item, model, key, value):
    print(item["ean"])
    print(value)
    print()
    item[key][model].append(value)
    store_page(connection, project, item["ean"], json.dumps(item))


def classify_serial(
    assemble_prompt,
    evaluate_response,
    connection,
    project,
    items,
    model,
    think,
):
    for item in items:
        item = prepare(item, model)
        if item:
            for _ in range(3):
                classification = classify(
                    assemble_prompt,
                    evaluate_response,
                    item,
                    model,
                    think,
                    debug=True,
                )
                follow_up(
                    connection, project, item, model, "klassifikation", classification
                )


def classify_parallel(
    assemble_prompt,
    evaluate_response,
    evaluate_error,
    connection,
    project,
    items,
    model,
    think,
    max_workers=4,
):
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for item in items:
            item = prepare(item, model)
            if item:
                for _ in range(3):
                    futures[
                        executor.submit(
                            classify,
                            assemble_prompt,
                            evaluate_response,
                            item,
                            model,
                            think,
                        )
                    ] = item
        for future in concurrent.futures.as_completed(futures):
            item = futures[future]
            try:
                classification = future.result()
            except Exception as error:
                classification = evaluate_error(error)
            follow_up(
                connection, project, item, model, "klassifikation", classification
            )


def rate_b2c_serial(
    assemble_prompt, evaluate_response, connection, project, items, model, think
):
    for item in items:
        item = prepare(item, model)
        if item:
            for _ in range(3):
                rating = rate_b2c(
                    assemble_prompt,
                    evaluate_response,
                    item,
                    model,
                    think,
                    debug=True,
                )
                follow_up(connection, project, item, model, "bewertung_b2c", rating)


def rate_b2c_parallel(
    assemble_prompt,
    evaluate_response,
    evaluate_error,
    connection,
    project,
    items,
    model,
    think,
    max_workers=4,
):
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for item in items:
            item = prepare(item, model)
            if item:
                for _ in range(3):
                    futures[
                        executor.submit(
                            rate_b2c,
                            assemble_prompt,
                            evaluate_response,
                            item,
                            model,
                            think,
                        )
                    ] = item
        for future in concurrent.futures.as_completed(futures):
            item = futures[future]
            try:
                rating = future.result()
            except Exception as error:
                rating = evaluate_error(error)
            follow_up(connection, project, item, model, "bewertung_b2c", rating)
