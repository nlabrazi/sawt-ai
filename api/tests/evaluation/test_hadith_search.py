import pytest

from scripts.evaluate_hadith_search import evaluate, validate_corpus


def test_unreviewed_labels_cannot_produce_accuracy():
    corpus = {"cases": [{"query": "requête témoin", "candidate_hadeethenc_ids": ["4709"], "expected_hadeethenc_ids": [], "review_status": "pending"}]}
    with pytest.raises(ValueError, match="Validation humaine"):
        validate_corpus(corpus)
    assert len(validate_corpus(corpus, preview=True)) == 1


class FakeIndex:
    def load(self):
        pass

    def rank(self, query, limit):
        return [("1", 0.9), ("2", 0.8), ("3", 0.7)]


def test_metrics_count_hits_at_one_and_three_separately():
    cases = [{"query": "première requête", "expected_hadeethenc_ids": ["1"]}, {"query": "deuxième requête", "expected_hadeethenc_ids": ["3"]}, {"query": "troisième requête", "expected_hadeethenc_ids": ["9"]}]
    report = evaluate(cases, FakeIndex())
    assert report["top_1_accuracy"] == pytest.approx(1 / 3)
    assert report["top_3_accuracy"] == pytest.approx(2 / 3)
    assert report["mean_latency_ms"] >= 0
    preview = evaluate(cases, FakeIndex(), preview=True)
    assert "top_1_accuracy" not in preview
    assert "top_3_accuracy" not in preview
    assert preview["status"] == "pending_human_review"
