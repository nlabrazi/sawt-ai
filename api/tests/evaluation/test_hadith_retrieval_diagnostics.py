import numpy as np
import pytest

from app.services.hadith_documents import SearchDocument
from scripts.diagnose_hadith_retrieval import assert_same_experiment, benchmark_cases, fingerprint, group_documents, rank_documents


def test_provisional_labels_require_explicit_opt_in_and_do_not_mutate_corpus():
    corpus = {"cases": [{"query": "requête témoin", "expected_hadeethenc_ids": [], "review_status": "pending", "candidates_for_review": [{"hadeethenc_id": "1"}]}]}
    with pytest.raises(ValueError, match="Validation humaine"):
        benchmark_cases(corpus)
    cases, status = benchmark_cases(corpus, provisional=True)
    assert cases[0]["expected_hadeethenc_ids"] == ["1"]
    assert status == "provisional_candidates_not_human_validated"
    assert corpus["cases"][0]["expected_hadeethenc_ids"] == []


def test_max_score_per_hadith_deduplicates_results_without_averaging():
    docs = [SearchDocument(hid, "test", "passage: test", ()) for hid in ["8", "8", "5", "9"]]
    ids, groups = group_documents(docs)
    # First hadith wins through its second document. Averaging would rank it last.
    matrix = np.array([[0, 1], [1, 0], [0.8, 0.6], [0.6, 0.8]], dtype=np.float32)
    scores, document_scores, ranking = rank_documents(matrix, groups, np.array([1, 0], dtype=np.float32))
    assert [ids[i] for i in ranking] == ["8", "5", "9"]
    assert scores.tolist() == pytest.approx([1, 0.8, 0.6])
    assert document_scores.tolist() == pytest.approx([0, 1, 0.8, 0.6])


def test_ab_comparison_rejects_changed_queries_documents_or_snapshot():
    snapshot = {"source": "fixed"}
    documents = [SearchDocument("1", "title", "passage: exact source", ())]
    cases = [{"query": "description naturelle", "expected_hadeethenc_ids": ["1"]}]
    reference = {"strategy": "multi", "snapshot_sha256": fingerprint(snapshot), "documents_sha256": fingerprint([{"id": "1", "kind": "title", "text": "passage: exact source"}]), "benchmark_sha256": fingerprint(cases)}
    assert_same_experiment(reference, snapshot, documents, cases, "multi")
    for key in reference:
        with pytest.raises(ValueError, match="only the model may change"):
            assert_same_experiment({**reference, key: "changed"}, snapshot, documents, cases, "multi")
