#!/usr/bin/env python3
"""Compare request-prefix cleanup with unchanged E5-base embeddings and ranking."""

import json
from pathlib import Path
import sys
from time import perf_counter

API_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(API_DIR))

from app.services.hadith_query import normalize_hadith_query
from scripts.hadith_experiment import HadithExperiment


def main():
    report_path = API_DIR / "evaluation" / "hadith_retrieval" / "multilingual-e5-base_multi.json"
    reference = json.loads(report_path.read_text(encoding="utf-8"))
    snapshot = json.loads((API_DIR / ".cache" / "hadith_retrieval" / "snapshot.json").read_text(encoding="utf-8"))
    titles = {str(record["payload"]["id"]): record["payload"]["title"] for record in snapshot["records"]}
    index = HadithExperiment()
    index.load()
    index.rank("réchauffement du modèle")

    def retrieve(query, expected):
        started = perf_counter()
        ranked = index.rank(query, 5)
        latency = (perf_counter() - started) * 1000
        ids = [hid for hid, _ in ranked]
        return {"query_used": query, "latency_ms": latency,
                "top_1_hit": ids[0] in expected if expected else None,
                "top_3_hit": bool(set(ids[:3]) & set(expected)) if expected else None,
                "top_5": [{"id": hid, "score": score, "title": titles[hid]} for hid, score in ranked]}

    def compare(query, expected, *, swap=False):
        cleaned = normalize_hadith_query(query)
        # Alternate execution order to avoid always timing the cleaned query last.
        if swap:
            after, before = retrieve(cleaned, expected), retrieve(query, expected)
        else:
            before, after = retrieve(query, expected), retrieve(cleaned, expected)
        return {"query": query, "normalized_query": cleaned, "expected_ids": expected,
                "changed": query != cleaned, "before": before, "after": after}

    original = [compare(row["query"], row["expected_ids"], swap=bool(i % 2))
                for i, row in enumerate(reference["results"])]
    for result, old in zip(original, reference["results"]):
        actual, expected = result["before"]["top_5"], old["top_5"]
        if ([row["id"] for row in actual] != [row["id"] for row in expected]
                or any(abs(a["score"] - b["score"]) > 1e-5 for a, b in zip(actual, expected))):
            raise ValueError("The untreated queries no longer reproduce the original benchmark")

    templates = ["Je cherche le hadith sur {subject}", "Donnez moi hadith qui parle de {subject}",
                 "Pouvez-vous me donner un hadith sur {subject}"]
    wrapped = []
    for case in reference["results"]:
        subject = normalize_hadith_query(case["query"])
        for template in templates:
            result = compare(template.format(subject=subject), case["expected_ids"], swap=bool(len(wrapped) % 2))
            if result["normalized_query"] != subject:
                raise ValueError("An equivalent request changed the subject")
            wrapped.append(result)
    # Broad topics have no single validated answer: retain rankings, no accuracy.
    exploratory = [compare(query, []) for query in (
        "Je cherche le hadith sur la colère", "Donnez moi hadith qui parle de la colère",
        "Je cherche le hadith sur les intentions", "Donnez moi un hadith qui parle de la mère",
    )]

    def metrics(rows, phase):
        return {"top_1_accuracy": sum(row[phase]["top_1_hit"] for row in rows) / len(rows),
                "top_3_accuracy": sum(row[phase]["top_3_hit"] for row in rows) / len(rows),
                "mean_latency_ms": sum(row[phase]["latency_ms"] for row in rows) / len(rows)}

    report = {"label_status": reference["label_status"], "model": reference["model"],
              "model_revision": reference["model_revision"], "snapshot_sha256": reference["snapshot_sha256"],
              "documents_sha256": reference["documents_sha256"], "benchmark_sha256": reference["benchmark_sha256"],
              "reference_raw_results_verified": True,
              "latency_scope": "warm query encoding + unchanged cosine/max ranking; excludes model load and HTTP",
              "normalization_scope": "recognized French request prefixes at the beginning only; subject preserved",
              "original_benchmark": {"count": len(original), "changed_count": sum(row["changed"] for row in original),
                                     "before": metrics(original, "before"), "after": metrics(original, "after"), "results": original},
              "synthetic_prefix_variations": {"count": len(wrapped), "templates": templates,
                                             "before": metrics(wrapped, "before"), "after": metrics(wrapped, "after"),
                                             "results": wrapped},
              "exploratory_topics_unlabelled": exploratory}
    output = API_DIR / "evaluation" / "hadith_query_cleanup"
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Retrait des amorces de recherche", "",
             "Labels provisoires, non validés humainement. Le corpus, E5-base, ses passages et le classement restent identiques.", "",
             "Les vingt classements sans nettoyage reproduisent les IDs et scores du benchmark initial.", "",
             "| Jeu | Requêtes | Top-1 avant | Top-1 après | Top-3 avant | Top-3 après |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for key, label in (("original_benchmark", "Benchmark initial"), ("synthetic_prefix_variations", "Amorces synthétiques")):
        part = report[key]
        before, after = part["before"], part["after"]
        lines.append(f"| {label} | {part['count']} | {before['top_1_accuracy']:.0%} | {after['top_1_accuracy']:.0%} | {before['top_3_accuracy']:.0%} | {after['top_3_accuracy']:.0%} |")
    lines += ["", "Les 60 variations sont construites à partir des mêmes 20 sujets : ce ne sont pas 60 nouveaux cas indépendants. Elles mesurent la résistance aux trois amorces, pas la qualité générale.", "",
              "## Requêtes modifiées du benchmark initial", ""]
    for result in original:
        if result["changed"]:
            lines += [f"- Original : {result['query']}", f"  - Recherche utilisée : {result['normalized_query']}",
                      f"  - Top 3 avant : {', '.join(row['id'] for row in result['before']['top_5'][:3])}.",
                      f"  - Top 3 après : {', '.join(row['id'] for row in result['after']['top_5'][:3])}."]
    lines += ["", "## Échecs Top-3 restant sur le benchmark initial", ""]
    for result in original:
        if not result["after"]["top_3_hit"]:
            lines.append(f"- {result['query']} — candidats attendus : {', '.join(result['expected_ids'])}.")
    lines += ["", "## Essais libres sans réponse unique validée", "",
              "Les sujets courts peuvent rester ambigus. Le nettoyage n'est pas une garantie d'amélioration pour chaque recherche.", ""]
    for result in exploratory:
        lines += [f"### {result['query']}", "", f"Recherche utilisée : **{result['normalized_query']}**.", ""]
        for phase, label in (("before", "Avant"), ("after", "Après")):
            lines.append(label + " :")
            lines.append("")
            for rank, row in enumerate(result[phase]["top_5"][:3], 1):
                title = " ".join(row["title"].splitlines())
                lines.append(f"{rank}. [{row['id']}](https://hadeethenc.com/fr/browse/hadith/{row['id']}) — {title}")
            lines.append("")
    lines += ["## Reproduction", "", "Dans l'environnement Python du lanceur Docker :", "",
              "```bash", ".cache/hadith-cli-venv/bin/python scripts/evaluate_hadith_query_cleanup.py", "```", "",
              "`results.json` contient les requêtes exactes avant/après, tous les Top 5, scores, latences et empreintes de la comparaison.", ""]
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({key: {k: v for k, v in report[key].items() if k != "results"}
                      for key in ("original_benchmark", "synthetic_prefix_variations")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
