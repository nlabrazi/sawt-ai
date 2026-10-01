"""Conservative keyword matching against visible French source fields."""

import re
import unicodedata
from dataclasses import dataclass

from app.services.hadith_query import normalize_hadith_query

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
_SHORT_PREFIX = re.compile(r"^(?:(?:un|le|les|des)\s+)?hadiths?\s+", re.IGNORECASE)
_STOP_WORDS = frozenset("le la les de du des un une l d au aux à sur concernant et en pour".split())
_NEGATIONS = frozenset({"ne", "n", "pas", "jamais", "sans"})


@dataclass(frozen=True)
class SearchQuery:
    text: str
    # None means a sentence; an empty tuple means a keyword query with no subject.
    terms: tuple[str, ...] | None


def words(text: str) -> set[str]:
    # Preserve accents: the noun 'couronne' must not match 'couronné de succès'.
    return set(_WORD.findall(unicodedata.normalize("NFC", text).casefold()))


def prepare_search_query(query: str) -> SearchQuery:
    text = normalize_hadith_query(query)
    subject = _SHORT_PREFIX.sub("", text).strip()
    tokens = _WORD.findall(unicodedata.normalize("NFC", subject).casefold())
    # A short negated sentence still needs semantic retrieval, not bag-of-words.
    if len(tokens) > 3 or _NEGATIONS.intersection(tokens):
        return SearchQuery(text, None)
    terms = tuple(dict.fromkeys(token for token in tokens if token not in _STOP_WORDS and token not in {"hadith", "hadiths"}))
    return SearchQuery(subject, terms)


def term_variants(term: str) -> set[str]:
    variants = {term, term + "s"}
    if len(term) > 3 and term.endswith("s"):
        variants.add(term[:-1])
    return variants


def matches_keywords(text: str, terms: tuple[str, ...]) -> bool:
    tokens = words(text)
    return bool(terms) and all(tokens.intersection(term_variants(term)) for term in terms)


def source_search_text(payload: dict) -> str:
    parts = [payload["title"], payload["hadeeth"]]
    if payload.get("explanation"):
        parts.append(payload["explanation"])
    if any(not isinstance(part, str) or not part.strip() for part in parts):
        raise ValueError("Invalid French source text")
    return "\n\n".join(parts)


def source_search_documents(records: list[dict]) -> dict[str, str]:
    documents = {}
    for record in records:
        payload = record["payload"]
        hid = str(payload["id"])
        if not hid.isascii() or not hid.isdigit():
            raise ValueError("Invalid source ID")
        if hid in documents:
            raise ValueError("Duplicate source ID")
        documents[hid] = source_search_text(payload)
    return documents
