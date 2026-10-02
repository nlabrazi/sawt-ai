"""Remove recognized French search requests without rewriting their subject."""

import re

# These expressions only match at the beginning and must include "hadith" plus
# an explicit subject connector. Negated requests and unknown forms stay intact.
_COURTESY = r"(?:s['’]il\s+(?:vous|te)\s+pla[îi]t[ ,:]+)?"
_REQUEST = (
    r"(?:je\s+(?:cherche|recherche|veux|voudrais|souhaite|aimerais)(?:\s+(?:trouver|retrouver))?"
    r"|j['’]aimerais(?:\s+(?:trouver|retrouver))?"
    r"|(?:donne[z]?|trouve[z]?|montre[z]?|cherche[z]?|rappelle[z]?)[\s-]+moi"
    r"|(?:peux[\s-]+tu|pouvez[\s-]+vous)\s+(?:me\s+)?(?:donner|trouver|montrer|retrouver))\s+"
)
_CONNECTOR = (
    r"(?:(?:sur|concernant|où)\s+"
    r"|(?:(?:au\s+sujet|à\s+propos)|qui\s+(?:parle|parlent|traite|traitent))"
    r"\s+(?:de\s+|d['’]|(?=du\s+|des\s+))"
    r"|qui\s+(?:dit|disent|explique|expliquent|enseigne|enseignent|rappelle|rappellent)"
    r"\s+qu(?:e\s+|['’]))"
)
_PREFIX = re.compile(
    r"^" + _COURTESY + r"(?:" + _REQUEST + r")?"
    r"(?:(?:le\s+ou\s+les|un\s+ou\s+plusieurs|un|le|les|des)\s+)?hadiths?\s+" + _CONNECTOR,
    re.IGNORECASE,
)


def normalize_hadith_query(query: str) -> str:
    original = query.strip()
    match = _PREFIX.match(original)
    if match is None:
        return original
    subject = original[match.end():].strip()
    # Never send an empty subject or repeatedly strip nested requests/quotations.
    if (not re.search(r"[^\W\d_]", subject) or _PREFIX.match(subject)
            or subject.casefold() in {"le", "la", "les", "un", "une", "des", "du", "de"}):
        return original
    return subject
