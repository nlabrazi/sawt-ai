from app.core.detection_policy import build_detection_policy


def test_build_detection_policy_exposes_shared_acceptance_thresholds():
    assert build_detection_policy() == {
        "min_accepted_similarity": 0.8,
        "min_probable_similarity": 0.6,
        "min_proposal_similarity": 0.72,
        "min_proposal_matched_word_count": 4,
        "min_proposal_score_margin": 0.05,
        "max_ambiguous_rescue_score": 0.92,
        "min_matched_word_count": 3,
        "min_score_margin": 0.08,
        "progressive_analysis_step_seconds": 5,
    }
