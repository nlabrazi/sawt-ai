MIN_ACCEPTED_SIMILARITY = 0.8
MIN_PROBABLE_SIMILARITY = 0.6
MIN_MATCHED_WORD_COUNT = 3
MIN_SCORE_MARGIN = 0.08


def build_detection_policy() -> dict[str, float | int]:
    return {
        "min_accepted_similarity": MIN_ACCEPTED_SIMILARITY,
        "min_probable_similarity": MIN_PROBABLE_SIMILARITY,
        "min_matched_word_count": MIN_MATCHED_WORD_COUNT,
        "min_score_margin": MIN_SCORE_MARGIN,
    }
