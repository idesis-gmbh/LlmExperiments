import json
import sqlite3
import gradio as gr


def load_data(project_name="proart"):
    pages = []
    with sqlite3.connect("data/rag.db") as connection:
        cursor = connection.cursor()
        for (
            page_name,
            markdown,
        ) in cursor.execute(
            "SELECT pages.name as page_name, pages.markdown as markdown "
            "FROM pages "
            "INNER JOIN projects on pages.project_id = projects.id "
            "WHERE projects.name = ? AND pages.markdown IS NOT NULL "
            "ORDER BY pages.name ",
            [project_name],
        ):
            pages.append(
                {
                    "key": page_name,
                    "value": markdown,
                }
            )
        connection.commit()
    return pages


data = load_data()


def format_for_table(records):
    if not records:
        return []
    content = []
    for record in records:
        item = json.loads(record["value"])
        content.append([record["key"], item["artikelname"]])
    return content


def display_json(filtered_data, event: gr.SelectData):
    row_index = event.index[0]
    json_string = filtered_data[row_index]["value"]
    try:
        json_object = json.loads(json_string)
        return json_object
    except Exception as e:
        return {"error": f"Invalid JSON: {str(e)}"}


def search_table(query):
    if not query:
        filtered = data
    else:
        filtered = [
            record
            for record in data
            if query.lower() in record["key"].lower()
            or query.lower() in record["value"].lower()
        ]
    return format_for_table(filtered), filtered


def reload_data():
    global data
    data = load_data()
    return format_for_table(data), data, ""


with gr.Blocks(title="JSON Table Browser") as demo:
    gr.Markdown("# JSON Table Browser")
    filtered_state = gr.State(value=data)
    with gr.Row():
        search_box = gr.Textbox(
            label="Search", placeholder="Filter by key or JSON content..."
        )
        reload_button = gr.Button("Reload", scale=0)
    with gr.Row():
        with gr.Column(scale=1):
            table = gr.Dataframe(
                value=format_for_table(data),
                headers=["GTIN", "Name"],
                interactive=False,
                label="Data Table",
                wrap=True,
            )
        with gr.Column(scale=1):
            json_viewer = gr.JSON(label="Selected JSON", value=None)
    search_box.submit(
        search_table, inputs=[search_box], outputs=[table, filtered_state]
    )
    table.select(display_json, inputs=[filtered_state], outputs=[json_viewer])
    reload_button.click(reload_data, outputs=[table, filtered_state, search_box])


if __name__ == "__main__":
    demo.launch()
