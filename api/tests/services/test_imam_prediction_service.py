import app.services.imam_prediction_service as imam_prediction_service


def test_preflight_imam_resources_does_not_raise_when_feature_is_unavailable(monkeypatch):
    def fail_load():
        raise imam_prediction_service.ImamResourcesUnavailableError("missing model")

    monkeypatch.setattr(imam_prediction_service, "load_imam_resources", fail_load)

    imam_prediction_service.preflight_imam_resources()


def test_get_imam_service_health_returns_unavailable_when_cached_error_is_present(monkeypatch):
    monkeypatch.setattr(imam_prediction_service, "encoder", None)
    monkeypatch.setattr(imam_prediction_service, "model", None)
    monkeypatch.setattr(imam_prediction_service, "index_to_name", None)
    monkeypatch.setattr(
        imam_prediction_service,
        "imam_resources_error",
        imam_prediction_service.ImamResourcesUnavailableError("missing model"),
    )

    assert imam_prediction_service.get_imam_service_health() == {
        "available": False,
        "status": "unavailable",
        "message": "La reconnaissance de l’imam est temporairement indisponible.",
    }


class DummyWaveform:
    def __init__(self, length: int):
        self.shape = (1, length)

    def __getitem__(self, item):
        sl = item[-1]
        new_len = min(self.shape[-1], sl.stop)
        return DummyWaveform(new_len)


def test_calibrate_audio_for_imam_truncates_long_audio():
    long_audio = DummyWaveform(320_000)
    calibrated = imam_prediction_service.calibrate_audio_for_imam(long_audio, max_samples=160_000)
    assert calibrated.shape == (1, 160_000)


def test_calibrate_audio_for_imam_preserves_short_audio():
    short_audio = DummyWaveform(64_000)
    calibrated = imam_prediction_service.calibrate_audio_for_imam(short_audio, max_samples=160_000)
    assert calibrated.shape == (1, 64_000)

