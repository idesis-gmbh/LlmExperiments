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
