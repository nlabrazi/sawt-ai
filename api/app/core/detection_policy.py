MIN_ACCEPTED_SIMILARITY = 0.8
MIN_PROBABLE_SIMILARITY = 0.6
MIN_PROPOSAL_SIMILARITY = 0.72
MIN_PROPOSAL_MATCHED_WORD_COUNT = 4
MIN_PROPOSAL_SCORE_MARGIN = 0.05
MIN_MATCHED_WORD_COUNT = 3
MIN_SCORE_MARGIN = 0.08
PROGRESSIVE_ANALYSIS_STEP_SECONDS = 5


def build_detection_policy() -> dict[str, float | int]:
    return {
        "min_accepted_similarity": MIN_ACCEPTED_SIMILARITY,
        "min_probable_similarity": MIN_PROBABLE_SIMILARITY,
        "min_proposal_similarity": MIN_PROPOSAL_SIMILARITY,
        "min_proposal_matched_word_count": MIN_PROPOSAL_MATCHED_WORD_COUNT,
        "min_proposal_score_margin": MIN_PROPOSAL_SCORE_MARGIN,
        "min_matched_word_count": MIN_MATCHED_WORD_COUNT,
        "min_score_margin": MIN_SCORE_MARGIN,
        "progressive_analysis_step_seconds": PROGRESSIVE_ANALYSIS_STEP_SECONDS,
    }
