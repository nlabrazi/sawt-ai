"""Search-only document layouts; official source payloads remain untouched."""

from dataclasses import dataclass

PASSAGE_PREFIX = "passage: "
QUERY_PREFIX = "query: "


@dataclass(frozen=True)
class SearchDocument:
    hadeethenc_id: str
    kind: str
    text: str
    sections: tuple[dict, ...]


def source_sections(payload: dict, categories: dict[str, str]) -> dict[str, list[str]]:
    return {
        "title": [payload["title"]],
        "categories": [categories[str(cid)] for cid in payload.get("categories", []) if str(cid) in categories],
        "hints": [hint for hint in payload.get("hints", []) if isinstance(hint, str)],
        "hadith": [payload["hadeeth"]],
        "explanation": [payload.get("explanation") or ""],
    }


def join_document(payload: dict, categories: dict[str, str], order: tuple[str, ...], kind: str) -> SearchDocument:
    parts = source_sections(payload, categories)
    text = PASSAGE_PREFIX
    spans = []
    for section in order:
        for part in parts[section]:
            if not part.strip():
                continue
            if spans:
                text += "\n\n"
            start = len(text)
            text += part
            spans.append({"section": section, "start": start, "end": len(text)})
    return SearchDocument(str(payload["id"]), kind, text, tuple(spans))


def inspect_document(document: SearchDocument, tokenizer, max_tokens: int = 512) -> dict:
    full = tokenizer(document.text, return_offsets_mapping=True, truncation=False)
    retained = tokenizer(document.text, return_offsets_mapping=True, truncation=True, max_length=max_tokens)
    def count(offsets, span):
        return sum(end > start and start < span["end"] and end > span["start"] for start, end in offsets)
    sections = []
    for span in document.sections:
        total = count(full["offset_mapping"], span)
        kept = count(retained["offset_mapping"], span)
        sections.append({"section": span["section"], "tokens": total, "retained_tokens": kept, "status": "retained" if kept == total else "lost" if kept == 0 else "truncated"})
    return {"kind": document.kind, "search_text": document.text, "token_count": len(full["input_ids"]), "retained_token_count": len(retained["input_ids"]), "truncated": len(full["input_ids"]) > max_tokens, "sections": sections}


def split_document(document: SearchDocument, tokenizer, max_tokens: int = 512) -> list[SearchDocument]:
    if len(tokenizer(document.text)["input_ids"]) <= max_tokens:
        return [document]
    body = document.text[len(PASSAGE_PREFIX):]
    offsets = tokenizer(body, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    # Reserve prefix, special tokens, and boundary re-tokenization room.
    budget = max_tokens - len(tokenizer(PASSAGE_PREFIX)["input_ids"]) - 4
    if budget <= 0:
        raise ValueError("Token budget too small")
    chunks = []
    token_start = 0
    char_start = 0
    while token_start < len(offsets):
        token_end = min(token_start + budget, len(offsets))
        char_end = offsets[token_end][0] if token_end < len(offsets) else len(body)
        text = PASSAGE_PREFIX + body[char_start:char_end]
        while len(tokenizer(text)["input_ids"]) > max_tokens:
            token_end -= 1
            if token_end <= token_start:
                raise ValueError("Unable to split search document")
            char_end = offsets[token_end][0]
            text = PASSAGE_PREFIX + body[char_start:char_end]
        section_spans = []
        for span in document.sections:
            start = max(span["start"] - len(PASSAGE_PREFIX), char_start)
            end = min(span["end"] - len(PASSAGE_PREFIX), char_end)
            if end > start:
                section_spans.append({"section": span["section"], "start": len(PASSAGE_PREFIX) + start - char_start, "end": len(PASSAGE_PREFIX) + end - char_start})
        chunks.append(SearchDocument(document.hadeethenc_id, f"{document.kind}:{len(chunks) + 1}", text, tuple(section_spans)))
        char_start, token_start = char_end, token_end
    return chunks



def split_document_with_context(document: SearchDocument, tokenizer, max_tokens: int = 512) -> list[SearchDocument]:
    """Rebalance the last two chunks if the tail has less than a quarter-window.

    This preserves every character, including attributions and punctuation, while
    preventing an orphan closing quote or source credit from ranking on its own.
    The original splitter remains available to reproduce the frozen A/B reports.
    """
    chunks = split_document(document, tokenizer, max_tokens)
    if len(chunks) < 2:
        return chunks
    tail = chunks[-1].text[len(PASSAGE_PREFIX):]
    if len(tokenizer(tail, add_special_tokens=False)["input_ids"]) >= max_tokens // 4:
        return chunks
    body = document.text[len(PASSAGE_PREFIX):]
    start = sum(len(chunk.text) - len(PASSAGE_PREFIX) for chunk in chunks[:-2])
    remaining = body[start:]
    offsets = tokenizer(remaining, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    midpoint = len(offsets) // 2
    candidates = sorted(range(1, len(offsets)), key=lambda i: (
        not (remaining[offsets[i][0] - 1].isspace() or remaining[offsets[i][0]].isspace()),
        abs(i - midpoint),
    ))
    for token_cut in candidates:
        cut = offsets[token_cut][0]
        if all(len(tokenizer(PASSAGE_PREFIX + part)["input_ids"]) <= max_tokens
               for part in (remaining[:cut], remaining[cut:])):
            break
    else:
        raise ValueError("Unable to retain context within the token budget")
    rebuilt = chunks[:-2]
    for char_start, char_end in ((start, start + cut), (start + cut, len(body))):
        spans = []
        for span in document.sections:
            left = max(span["start"] - len(PASSAGE_PREFIX), char_start)
            right = min(span["end"] - len(PASSAGE_PREFIX), char_end)
            if right > left:
                spans.append({"section": span["section"], "start": len(PASSAGE_PREFIX) + left - char_start,
                              "end": len(PASSAGE_PREFIX) + right - char_start})
        rebuilt.append(SearchDocument(document.hadeethenc_id, f"{document.kind}:{len(rebuilt) + 1}",
                                      PASSAGE_PREFIX + body[char_start:char_end], tuple(spans)))
    return rebuilt


def build_documents(payload: dict, categories: dict[str, str], strategy: str, tokenizer=None, max_tokens: int = 512) -> list[SearchDocument]:
    if strategy == "original":
        return [join_document(payload, categories, ("title", "hadith", "explanation", "hints", "categories"), "original")]
    if strategy == "semantic_first":
        return [join_document(payload, categories, ("title", "categories", "hints", "hadith", "explanation"), "semantic_first")]
    if strategy not in ("multi", "multi_context") or tokenizer is None:
        raise ValueError("Unknown strategy or missing tokenizer")
    documents = []
    for kind, order in (("title_categories", ("title", "categories")), ("hints", ("hints",)), ("hadith_explanation", ("hadith", "explanation"))):
        document = join_document(payload, categories, order, kind)
        if document.sections:
            splitter = split_document_with_context if strategy == "multi_context" else split_document
            documents.extend(splitter(document, tokenizer, max_tokens))
    return documents
