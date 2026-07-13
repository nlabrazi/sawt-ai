from app.core.detection_policy import build_detection_policy


def test_build_detection_policy_exposes_shared_acceptance_thresholds():
    assert build_detection_policy() == {
        "min_accepted_similarity": 0.8,
        "min_probable_similarity": 0.6,
        "min_matched_word_count": 3,
        "min_score_margin": 0.08,
    }
