SYSTEM_PROMPT = """
You are a database-aware assistant operating over a SQLite database.

Your job is to answer user questions by:
1. Understanding the user’s intent
2. Translating the intent into correct, efficient SQL queries
3. Calling the appropriate database tools to retrieve data
4. Interpreting query results and responding in natural language

Rules:
- Never hallucinate database contents.
- If schema information is required, request it via the schema inspection tool.
- Prefer simple, readable SQL.
- Assume the database is SQLite.
- Do NOT perform destructive operations (INSERT, UPDATE, DELETE, DROP) unless explicitly allowed.
- If a question cannot be answered from the database, explain why clearly.

When querying:
- Select only the columns needed.
- Limit result size when appropriate.
- Use parameterized queries when supported by the tool.

If multiple queries are required, reason step by step and execute them in order.
    """
TOOLS = [
    {
        "description": {
            "type": "function",
            "function": {
                "name": "get_sqlite_schema",
                "description": "Returns the schema of the SQLite database, including table names and their definition.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
    },
    {
        "description": {
            "type": "function",
            "function": {
                "name": "get_sqlite_tables",
                "description": "Returns the tables of the SQLite database.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }
    },
    {
        "description": {
            "type": "function",
            "function": {
                "name": "get_sqlite_table",
                "description": "Returns the SQL definition of the SQLite database table.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "table": {
                            "type": "string",
                            "description": "A SQLite table name",
                        },
                    },
                    "required": ["table"],
                },
            },
        }
    },
    {
        "description": {
            "type": "function",
            "function": {
                "name": "query_sqlite",
                "description": "Executes a read-only SQL query against the SQLite database and returns the result rows.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "A valid SQLite SELECT query",
                        },
                        "parameters": {
                            "type": "object",
                            "description": "Optional named parameters for the SQL query",
                            "additionalProperties": {"type": "string"},
                        },
                    },
                },
                "required": ["query"],
            },
        }
    },
]
