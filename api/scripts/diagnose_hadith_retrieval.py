#!/usr/bin/env python3
"""Reproducible retrieval diagnostics on a frozen official-source snapshot."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter

API_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(API_DIR))

from app.core.hadith_config import HadithConfig
from app.services.hadeethenc_client import HadeethEncClient
from app.services.hadith_documents import QUERY_PREFIX, build_documents, inspect_document
from app.services.hadith_index import load_embedding_model, load_index
from scripts.evaluate_hadith_search import validate_corpus


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def benchmark_cases(corpus, provisional=False):
    if not provisional:
        return validate_corpus(corpus), "human_reviewed"
    cases = validate_corpus(corpus, preview=True)
    normalized = []
    for case in cases:
        ids = case.get("expected_hadeethenc_ids") or [candidate["hadeethenc_id"] for candidate in case.get("candidates_for_review", [])]
        if not ids or any(not isinstance(i, str) or not i.isascii() or not i.isdigit() for i in ids):
            raise ValueError("Each provisional case needs explicit source candidates")
        normalized.append({**case, "expected_hadeethenc_ids": ids})
    return normalized, "provisional_candidates_not_human_validated"


def freeze_snapshot(path, config):
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    _, ids, meta = load_index(config)
    client = HadeethEncClient(config.base_url)
    categories = {str(c["id"]): c["title"] for c in client.list_categories(config.language)}
    records = [json.loads((API_DIR / ".cache" / "hadeethenc" / config.language / f"{hid}.json").read_text(encoding="utf-8")) for hid in ids]
    snapshot = {"language": config.language, "base_url": config.base_url, "categories": categories, "records": records, "original_index_sha256": meta["index_sha256"]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, ensure_ascii=False), encoding="utf-8")
    return snapshot


def group_documents(documents):
    import numpy as np
    ids = list(dict.fromkeys(doc.hadeethenc_id for doc in documents))
    positions = {hid: i for i, hid in enumerate(ids)}
    return ids, np.asarray([positions[doc.hadeethenc_id] for doc in documents])


def rank_documents(matrix, groups, vector):
    import numpy as np
    document_scores = matrix @ vector
    scores = np.full(int(groups.max()) + 1, -np.inf, dtype=np.float32)
    np.maximum.at(scores, groups, document_scores)
    return scores, document_scores, np.argsort(-scores, kind="stable")


def assert_same_experiment(reference, snapshot, documents, cases, strategy):
    expected = {
        "strategy": strategy,
        "snapshot_sha256": fingerprint(snapshot),
        "documents_sha256": fingerprint([{"id": doc.hadeethenc_id, "kind": doc.kind, "text": doc.text} for doc in documents]),
        "benchmark_sha256": fingerprint([{k: case[k] for k in ("query", "expected_hadeethenc_ids")} for case in cases]),
    }
    for key, value in expected.items():
        if reference.get(key) != value:
            raise ValueError(f"A/B comparison changed {key}; only the model may change")


def run_diagnostics(cases, label_status, snapshot, model, matrix, documents, *, strategy, model_name, embedding_ms):
    import numpy as np
    import torch
    by_id = {str(record["payload"]["id"]): record["payload"] for record in snapshot["records"]}
    ids, groups = group_documents(documents)
    doc_indices = {hid: np.flatnonzero(groups == i) for i, hid in enumerate(ids)}
    missing = {hid for case in cases for hid in case["expected_hadeethenc_ids"] if hid not in by_id}
    if missing:
        raise ValueError(f"Expected IDs outside snapshot: {sorted(missing)}")
    documents_audit = [inspect_document(doc, model.tokenizer, model.max_seq_length) for doc in documents]
    if strategy in ("multi", "multi_context") and any(audit["truncated"] for audit in documents_audit):
        raise ValueError("A multi-passage document would be truncated")
    lost = Counter(section["section"] for audit in documents_audit for section in audit["sections"] if section["status"] == "lost")
    truncated = Counter(section["section"] for audit in documents_audit for section in audit["sections"] if section["status"] == "truncated")
    model.encode([QUERY_PREFIX + cases[0]["query"]], normalize_embeddings=True, show_progress_bar=False)
    results = []
    for case in cases:
        query = case["query"]
        started = perf_counter()
        vector = model.encode([QUERY_PREFIX + query], normalize_embeddings=True, show_progress_bar=False)[0]
        scores, document_scores, ranking = rank_documents(matrix, groups, vector)
        elapsed_ms = (perf_counter() - started) * 1000
        top = []
        for row in ranking[:5]:
            hid = ids[row]
            indices = doc_indices[hid]
            winning_row = int(indices[np.argmax(document_scores[indices])])
            top.append({"id": hid, "score": float(scores[row]), "title": by_id[hid]["title"], "winning_document": documents[winning_row].kind})
        expected = []
        for hid in case["expected_hadeethenc_ids"]:
            row = ids.index(hid)
            expected.append({"id": hid, "rank": int(np.flatnonzero(ranking == row)[0]) + 1, "score": float(scores[row]), "source_url": f"https://hadeethenc.com/fr/browse/hadith/{hid}", "documents": [documents_audit[i] for i in doc_indices[hid]]})
        expected_ids = set(case["expected_hadeethenc_ids"])
        results.append({"query": query, "query_text": QUERY_PREFIX + query, "query_tokens": len(model.tokenizer(QUERY_PREFIX + query)["input_ids"]), "expected_ids": case["expected_hadeethenc_ids"], "top_1_hit": top[0]["id"] in expected_ids, "top_3_hit": any(row["id"] in expected_ids for row in top[:3]), "latency_ms": elapsed_ms, "top_5": top, "expected_documents": expected})
    return {
        "label_status": label_status,
        "runtime": {
            "python": platform.python_version(), "device": str(model.device),
            "torch_threads": torch.get_num_threads(),
            "packages": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "sentence-transformers", "transformers")},
        },
        "model": model_name, "model_revision": getattr(model[0].auto_model.config, "_commit_hash", None),
        "strategy": strategy, "snapshot_sha256": fingerprint(snapshot),
        "documents_sha256": fingerprint([{"id": doc.hadeethenc_id, "kind": doc.kind, "text": doc.text} for doc in documents]),
        "benchmark_sha256": fingerprint([{k: case[k] for k in ("query", "expected_hadeethenc_ids")} for case in cases]),
        "prefix_audit": {"all_passages_prefixed": all(doc.text.startswith("passage: ") for doc in documents), "all_queries_prefixed": all(row["query_text"].startswith("query: ") for row in results)},
        "hadith_count": len(ids), "embedding_count": len(documents), "embedding_build_ms": embedding_ms,
        "truncation": {"max_tokens": model.max_seq_length, "truncated_documents": sum(audit["truncated"] for audit in documents_audit), "lost_sections": dict(lost), "partially_truncated_sections": dict(truncated)},
        "query_count": len(results), "top_1_accuracy": sum(row["top_1_hit"] for row in results) / len(results), "top_3_accuracy": sum(row["top_3_hit"] for row in results) / len(results),
        "mean_latency_ms": sum(row["latency_ms"] for row in results) / len(results),
        "latency_scope": "warm query embedding + cosine + max score per hadith; excludes HTTP and model load",
        "results": results,
    }


def markdown_report(report):
    lines = [f"# Retrieval — {report['model']} / {report['strategy']}", "", f"Labels : **{report['label_status']}**. Les taux sont provisoires si les labels ne sont pas relus humainement.", "", f"Top-1 : {report['top_1_accuracy']:.1%} · Top-3 : {report['top_3_accuracy']:.1%} · Latence moyenne : {report['mean_latency_ms']:.2f} ms.", "", "Latence à chaud : encodage + cosinus + regroupement par ID, hors HTTP.", "", f"{report['hadith_count']} hadiths, {report['embedding_count']} embeddings ; {report['truncation']['truncated_documents']} documents tronqués à {report['truncation']['max_tokens']} tokens.", "", f"Sections entièrement perdues : `{json.dumps(report['truncation']['lost_sections'], ensure_ascii=False)}`.", "", "## Échecs Top-3", ""]
    for result in report["results"]:
        if result["top_3_hit"]:
            continue
        lines += [f"### {result['query']}", "", f"IDs attendus provisoires : {', '.join(result['expected_ids'])}", "", "| Rang | ID | Cosinus | Titre officiel |", "| --- | --- | --- | --- |"]
        for rank, row in enumerate(result["top_5"], 1):
            title = row["title"].replace("|", "\\|").replace("\n", " ").replace("\r", " ")
            lines.append(f"| {rank} | {row['id']} | {row['score']:.6f} | {title} |")
        for expected in result["expected_documents"]:
            lines += ["", f"#### Source attendue : [{expected['id']}]({expected['source_url']}) — rang {expected['rank']}, score {expected['score']:.6f}", ""]
            for document in expected["documents"]:
                lines += [f"Document `{document['kind']}` : **{document['token_count']} tokens**, {document['retained_token_count']} conservés.", "", "| Section | Tokens | Conservés | État |", "| --- | --- | --- | --- |"]
                for section in document["sections"]:
                    lines.append(f"| {section['section']} | {section['tokens']} | {section['retained_tokens']} | {section['status']} |")
                lines += ["", "`search_text` exact (texte de recherche uniquement, source : HadeethEnc) :", "", "```text", document["search_text"], "```", ""]
    if all(result["top_3_hit"] for result in report["results"]):
        lines.append("Aucun échec Top-3 sur ces labels provisoires.")
    return "\n".join(lines).rstrip("\n") + "\n"


def main():
    import numpy as np
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", choices=("original", "semantic_first", "multi"), required=True)
    parser.add_argument("--model", default="intfloat/multilingual-e5-small")
    parser.add_argument("--corpus", type=Path, default=API_DIR / "evaluation" / "hadith_search_corpus.json")
    parser.add_argument("--snapshot", type=Path, default=API_DIR / ".cache" / "hadith_retrieval" / "snapshot.json")
    parser.add_argument("--output-dir", type=Path, default=API_DIR / "evaluation" / "hadith_retrieval")
    parser.add_argument("--provisional-labels", action="store_true")
    parser.add_argument("--reuse-original-index", action="store_true")
    parser.add_argument("--compare-to", type=Path, help="Require exactly the same documents, corpus and queries as this reference report")
    args = parser.parse_args()
    cases, label_status = benchmark_cases(json.loads(args.corpus.read_text(encoding="utf-8")), args.provisional_labels)
    config = HadithConfig.from_env()
    snapshot = freeze_snapshot(args.snapshot, config)
    model = load_embedding_model(args.model)
    document_tokenizer = model.tokenizer
    document_tokenizer_name = args.model
    document_tokenizer_revision = getattr(model[0].auto_model.config, "_commit_hash", None)
    document_max_tokens = model.max_seq_length
    reference = None
    if args.compare_to:
        from transformers import AutoTokenizer

        reference = json.loads(args.compare_to.read_text(encoding="utf-8"))
        # Freeze document segmentation too: a tokenizer change must not alter the A/B corpus.
        document_tokenizer_name = reference.get("document_tokenizer", reference["model"])
        document_tokenizer_revision = reference.get("document_tokenizer_revision", reference["model_revision"])
        document_tokenizer = AutoTokenizer.from_pretrained(document_tokenizer_name, revision=document_tokenizer_revision, trust_remote_code=False)
        document_max_tokens = reference["truncation"]["max_tokens"]
    documents = [doc for record in snapshot["records"] for doc in build_documents(record["payload"], snapshot["categories"], args.strategy, document_tokenizer, document_max_tokens)]
    if reference is not None:
        assert_same_experiment(reference, snapshot, documents, cases, args.strategy)
        if any(len(model.tokenizer(doc.text)["input_ids"]) > model.max_seq_length for doc in documents):
            raise ValueError("The compared model would truncate frozen documents")
        print("A/B verified: identical source snapshot, documents, queries and expected IDs", flush=True)
    name = args.model.rsplit("/", 1)[-1] + "_" + args.strategy
    cache_path = args.snapshot.parent / (name + ".npz")
    key = fingerprint({"model": args.model, "revision": getattr(model[0].auto_model.config, "_commit_hash", None), "documents": [{"id": doc.hadeethenc_id, "text": doc.text} for doc in documents]})
    started = perf_counter()
    if args.reuse_original_index:
        embedding_source = "verified_original_index"
        if args.strategy != "original" or args.model != config.model_name:
            parser.error("Original index can only be reused with its original layout and model")
        matrix, ids, _ = load_index(config)
        if ids != [doc.hadeethenc_id for doc in documents]:
            raise ValueError("Original ID order differs from frozen snapshot")
        expected_ids = {hid for case in cases for hid in case["expected_hadeethenc_ids"]}
        rows = [i for i, doc in enumerate(documents) if doc.hadeethenc_id in expected_ids]
        rebuilt = model.encode([documents[i].text for i in rows], normalize_embeddings=True, show_progress_bar=False)
        if not np.allclose(matrix[rows], rebuilt, atol=0.00001):
            raise ValueError("Expected documents do not reproduce the original index embeddings")
        print("Original index reproduced for all expected candidate IDs", flush=True)
    elif cache_path.is_file():
        embedding_source = "cached_experiment"
        with np.load(cache_path, allow_pickle=False) as archive:
            if archive["key"].item() != key:
                raise ValueError("Cached experiment differs from current snapshot/model")
            matrix = archive["embeddings"]
    else:
        embedding_source = "encoded_snapshot"
        print(f"Embedding {len(documents)} documents with {args.model} ({args.strategy})", flush=True)
        matrix = model.encode([doc.text for doc in documents], batch_size=16, normalize_embeddings=True, show_progress_bar=True)
        np.savez_compressed(cache_path, embeddings=matrix, key=np.asarray(key))
    embedding_ms = (perf_counter() - started) * 1000
    report = run_diagnostics(cases, label_status, snapshot, model, matrix, documents, strategy=args.strategy, model_name=args.model, embedding_ms=embedding_ms)
    report["embedding_source"] = embedding_source
    report["embedding_preparation_ms"] = embedding_ms
    report["embedding_build_ms"] = embedding_ms if embedding_source == "encoded_snapshot" else None
    report["document_tokenizer"] = document_tokenizer_name
    report["document_tokenizer_revision"] = document_tokenizer_revision
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / (name + ".json")).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / (name + ".md")).write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps({k: value for k, value in report.items() if k != "results"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
