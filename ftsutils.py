import re

_FTS_SYNTAX_RE = re.compile(r'[()"*.:]')
_FTS_KEYWORDS = {"and", "or", "not", "near"}


def sanitize_fts_query(user_input: str) -> str:
    if not user_input:
        return ""
    text = user_input.lower()
    text = _FTS_SYNTAX_RE.sub(" ", text)
    text = text.replace("-", " ")
    raw_tokens = text.split()
    tokens = [token for token in raw_tokens if token not in _FTS_KEYWORDS]
    if not tokens:
        return ""
    return " AND ".join(f'"{t}"' for t in tokens)
