from app.services.inference_pipeline import compute_imam_status


def test_compute_imam_status_returns_disabled_when_feature_is_off():
    assert compute_imam_status([{"name": "X", "score": 0.99}], detect_imam=False) == "disabled"


def test_compute_imam_status_returns_unknown_without_predictions():
    assert compute_imam_status([], detect_imam=True) == "unknown"


def test_compute_imam_status_uses_score_thresholds():
    assert compute_imam_status([{"name": "A", "score": 0.9}], detect_imam=True) == "high"
    assert compute_imam_status([{"name": "A", "score": 0.7}], detect_imam=True) == "medium"
    assert compute_imam_status([{"name": "A", "score": 0.4}], detect_imam=True) == "low"
