#!/usr/bin/env python3
"""Compare contextual tail splitting with the frozen E5-base experiment."""

import json
from pathlib import Path
import re
import sys
from time import perf_counter

import numpy as np
from transformers import AutoTokenizer

API_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(API_DIR))

from app.services.hadith_documents import PASSAGE_PREFIX, build_documents
from app.services.hadith_query import normalize_hadith_query
from scripts.diagnose_hadith_retrieval import fingerprint, group_documents, markdown_report, rank_documents, run_diagnostics
from scripts.hadith_experiment import HadithExperiment


def main():
    cache = API_DIR / ".cache" / "hadith_retrieval"
    reports = API_DIR / "evaluation" / "hadith_retrieval"
    output = API_DIR / "evaluation" / "hadith_context"
    reference = json.loads((reports / "multilingual-e5-base_multi.json").read_text())
    snapshot = json.loads((cache / "snapshot.json").read_text())
    index = HadithExperiment()
    old_matrix, ids, groups, model = index.load()
    tokenizer = AutoTokenizer.from_pretrained(reference["document_tokenizer"], revision=reference["document_tokenizer_revision"])
    tokenizer.model_max_length = 10**9
    old_documents, documents = [], []
    for record in snapshot["records"]:
        payload = record["payload"]
        old = build_documents(payload, snapshot["categories"], "multi", tokenizer, 512)
        new = build_documents(payload, snapshot["categories"], "multi_context", tokenizer, 512)
        for kind in ("title_categories", "hints", "hadith_explanation"):
            before = "".join(doc.text[len(PASSAGE_PREFIX):] for doc in old if doc.kind.split(":")[0] == kind)
            after = "".join(doc.text[len(PASSAGE_PREFIX):] for doc in new if doc.kind.split(":")[0] == kind)
            if before != after:
                raise ValueError("Source text changed during splitting")
        old_documents.extend(old)
        documents.extend(new)
    if fingerprint([{"id": d.hadeethenc_id, "kind": d.kind, "text": d.text} for d in old_documents]) != reference["documents_sha256"]:
        raise ValueError("Old documents differ from the frozen benchmark")
    if any(len(model.tokenizer(doc.text)["input_ids"]) > 512 for doc in documents):
        raise ValueError("New passages exceed E5-base's token limit")
    if group_documents(documents)[0] != ids:
        raise ValueError("Hadith IDs changed")

    old_rows = {(doc.hadeethenc_id, doc.text): i for i, doc in enumerate(old_documents)}
    changed = [i for i, doc in enumerate(documents) if (doc.hadeethenc_id, doc.text) not in old_rows]
    affected_ids = sorted({documents[i].hadeethenc_id for i in changed})
    key = fingerprint({"model": reference["model"], "revision": reference["model_revision"],
                       "documents": [{"id": d.hadeethenc_id, "text": d.text} for d in documents]})
    matrix_path = cache / "multilingual-e5-base_multi_context.npz"
    started = perf_counter()
    encoded = not matrix_path.is_file()
    if not encoded:
        with np.load(matrix_path, allow_pickle=False) as archive:
            if archive["key"].item() != key:
                raise ValueError("Existing contextual cache differs; preserve it before rebuilding")
            matrix = archive["embeddings"]
    else:
        matrix = np.empty((len(documents), old_matrix.shape[1]), dtype=np.float32)
        for i, doc in enumerate(documents):
            row = old_rows.get((doc.hadeethenc_id, doc.text))
            if row is not None:
                matrix[i] = old_matrix[row]
        print(f"Re-encoding {len(changed)} changed passages for {len(affected_ids)} hadiths; reusing {len(documents) - len(changed)} unchanged embeddings.", flush=True)
        if changed:
            matrix[changed] = model.encode([documents[i].text for i in changed], batch_size=16,
                                           normalize_embeddings=True, show_progress_bar=True)
        np.savez_compressed(matrix_path, embeddings=matrix, key=np.asarray(key))
    embedding_ms = (perf_counter() - started) * 1000
    cases = [{"query": normalize_hadith_query(row["query"]), "expected_hadeethenc_ids": row["expected_ids"]}
             for row in reference["results"]]
    output.mkdir(parents=True, exist_ok=True)
    measured = {}
    for strategy, docs, vectors, dest in (("multi", old_documents, old_matrix, output / "before.json"),
                                         ("multi_context", documents, matrix, reports / "multilingual-e5-base_multi_context.json")):
        report = run_diagnostics(cases, reference["label_status"], snapshot, model, vectors, docs,
                                 strategy=strategy, model_name=reference["model"], embedding_ms=embedding_ms if strategy == "multi_context" and encoded else None)
        if strategy == "multi_context":
            report["embedding_source"] = "partial_reencode" if encoded else "cached_experiment"
            report["embedding_changed_count"] = len(changed)
            report["embedding_reused_count"] = len(documents) - len(changed)
        report["document_tokenizer"] = reference["document_tokenizer"]
        report["document_tokenizer_revision"] = reference["document_tokenizer_revision"]
        report["query_preprocessing"] = "recognized French request prefix removal"
        report["original_queries"] = [row["query"] for row in reference["results"]]
        dest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        dest.with_suffix(".md").write_text(markdown_report(report))
        measured[strategy] = {k: report[k] for k in ("top_1_accuracy", "top_3_accuracy", "mean_latency_ms", "documents_sha256", "truncation")}
        print(strategy, json.dumps(measured[strategy]), flush=True)

    by_id = {str(r["payload"]["id"]): r["payload"] for r in snapshot["records"]}
    probes = []
    for query in ("la couronne", "les parents reçoivent une couronne de lumière parce que leur enfant a appris et pratiqué le Coran", "la colère", "la mère", "les intentions"):
        vector = model.encode(["query: " + query], normalize_embeddings=True, show_progress_bar=False)[0]
        result = {"query": query}
        for phase, docs, vectors in (("before", old_documents, old_matrix), ("after", documents, matrix)):
            doc_ids, doc_groups = group_documents(docs)
            scores, document_scores, ranking = rank_documents(vectors, doc_groups, vector)
            top = []
            for row in ranking[:5]:
                indices = np.flatnonzero(doc_groups == row)
                winning = int(indices[np.argmax(document_scores[indices])])
                top.append({"id": doc_ids[row], "score": float(scores[row]), "title": by_id[doc_ids[row]]["title"],
                            "winning_document": docs[winning].kind, "search_text": docs[winning].text})
            result[phase] = top
            result[phase + "_tracked_ranks"] = {hid: int(np.flatnonzero(ranking == doc_ids.index(hid))[0]) + 1 for hid in ("4181", "58226")}
        probes.append(result)
    french_hits, arabic_hits = [], []
    for hid, payload in by_id.items():
        for field in ("title", "hadeeth", "explanation", "hints", "hadeeth_ar", "explanation_ar", "hints_ar"):
            value = payload.get(field) or ""
            text = "\n".join(value) if isinstance(value, list) else value
            text = re.sub("[\u064b-\u065f\u0670\u0640]", "", text)
            if field.endswith("_ar"):
                if re.search(r"\b(?:[وفبلك]?تاج(?:ا|ان|ين)?|التاج)\b", text):
                    arabic_hits.append({"id": hid, "field": field})
            elif re.search(r"\bcouronnes?\b", text, re.I):
                french_hits.append({"id": hid, "field": field})
    summary = {"label_status": reference["label_status"], "snapshot_sha256": reference["snapshot_sha256"],
               "changed_passages": len(changed), "affected_ids": affected_ids,
               "text_preservation_verified": True, "comparison": measured,
               "coverage": {"records": len(by_id), "french_crown_matches": french_hits, "arabic_crown_matches": arabic_hits,
                            "conclusion": "Requested parents/Quran/crown narration not located in this snapshot; no conclusion about all external sources or authenticity."},
               "probes_unlabelled": probes}
    (output / "comparison.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"coverage": summary["coverage"], "crown_ranks_after": probes[0]["after_tracked_ranks"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
