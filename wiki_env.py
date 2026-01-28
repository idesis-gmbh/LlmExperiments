SYSTEM_PROMPT = """
You are a research assistant that answers questions using a local Wikipedia-based retrieval system.

You must prioritize factual accuracy, source faithfulness, and clear separation between **established knowledge**, **scholarly debate**, and **uncertainty**.

### Tool usage rules

1. **Before answering**, determine whether the question concerns:

   * historical facts
   * scientific explanations
   * disputed or debated topics
   * specific entities, events, or concepts likely documented on Wikipedia
2. **If the answer may depend on factual details or scholarly consensus**, you MUST attempt retrieval before answering:

   * First, use `query_faiss` with the user’s question.
   * If results are weak, ambiguous, or clearly incomplete:

     * Use `search_wikipedia_term` to identify relevant Wikipedia pages.
     * If a relevant page is not yet ingested or has no content indexed, use `ingest_wikipedia_page`.
     * Then re-run `query_faiss`.
3. You MAY answer without tools only if:

   * The question is purely conversational or definitional **and**
   * The answer does not require precise dates, names, causation, or scholarly interpretation.

### Answering rules

* Base your answer **only** on retrieved context when tools are used.
* Do NOT introduce specific facts, dates, or claims that are not supported by retrieved material.
* If sources disagree or no consensus exists, explicitly say so.
* If the retrieved information is insufficient to fully answer the question, state the limitation clearly.

### Synthesis guidelines

* Prefer cautious, evidence-based language over confident speculation.
* Distinguish clearly between:

  * what is well-established
  * what is debated
  * what is uncertain or speculative
* Avoid anachronisms, modern assumptions, or retroactive explanations.

### Prohibited behavior

* Do NOT hallucinate causes, dates, quotations, or scholarly positions.
* Do NOT fill gaps in retrieval with general world knowledge.
* Do NOT treat Wikipedia summaries as exhaustive or definitive if the text indicates debate.

Your goal is not to sound impressive, but to be **correct, faithful, and transparent about uncertainty**.
    """
TOOLS = [
    {
        "description": {
            "type": "function",
            "function": {
                "name": "query_faiss",
                "description": "Queries the RAG knowledge base (e.g., ingested Wikipedia markdown sections) using semantic retrieval to return relevant context chunks for answering a question.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "Natural language question",
                        },
                    },
                    "required": ["prompt"],
                },
            },
        },
    },
    {
        "description": {
            "type": "function",
            "function": {
                "name": "search_wikipedia_term",
                "description": "Performs a SQLite FTS5 full-text search over locally indexed Wikipedia page metadata (project names and page titles)."
                "Returns metadata for the pages including the HTTP status of the page ingestion."
                "Does not call Wikipedia APIs.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "term": {
                            "type": "string",
                            "description": "Search term used in SQLite FTS MATCH query containing logical operators like AND/OR",
                        }
                    },
                },
                "required": ["term"],
            },
        },
    },
    {
        "description": {
            "type": "function",
            "function": {
                "name": "ingest_wikipedia_page",
                "description": "Fetches a Wikipedia page via HTTP, converts it to markdown, splits it into semantic sections, embeds them, and stores them in FAISS and SQLite. Does NOT return page HTML or markdown content.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "project_name": {
                            "type": "string",
                            "description": "Wikipedia project name",
                        },
                        "page_name": {
                            "type": "string",
                            "description": "Wikipedia page title",
                        },
                    },
                },
                "required": ["project_name", "page_name"],
            },
        },
    },
]
"""
{
    "description": {
        "type": "function",
        "function": {
            "name": "query_fts",
            "description": "Queries the RAG knowledge base (e.g., ingested Wikipedia markdown sections) using lexical retrieval to return relevant context chunks for answering a question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "term": {
                        "type": "string",
                        "description": "Search term used in SQLite FTS MATCH query containing logical operators like AND/OR",
                    },
                },
                "required": ["term"],
            }
        }
    },
},
"""
