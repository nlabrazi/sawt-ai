from types import SimpleNamespace

import app.services.imam_prediction_service as imam_prediction_service


def test_preflight_imam_resources_does_not_raise_when_feature_is_unavailable(monkeypatch):
    def fail_load():
        raise imam_prediction_service.ImamResourcesUnavailableError("missing model")

    monkeypatch.setattr(imam_prediction_service, "load_imam_resources", fail_load)

    imam_prediction_service.preflight_imam_resources()


def test_load_imam_resources_passes_speechbrain_a_string_device(monkeypatch, tmp_path):
    model_path = tmp_path / "model.pt"
    model_path.touch()
    received = {}

    class Classifier:
        @staticmethod
        def from_hparams(**kwargs):
            received.update(kwargs)
            return object()

    class Model:
        def __init__(self, *_args):
            pass

        def load_state_dict(self, _state):
            pass

        def to(self, _device):
            pass

        def eval(self):
            pass

    class Device:
        def __str__(self):
            return "cpu"

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        device=lambda _name: Device(),
        load=lambda *_args, **_kwargs: {
            "input_dim": 1, "hidden_dim": 1, "num_classes": 1,
            "model_state_dict": {}, "labels": {"0": "imam"},
        },
    )
    monkeypatch.setattr(imam_prediction_service, "MODEL_PATH", model_path)
    monkeypatch.setattr(imam_prediction_service, "encoder", None)
    monkeypatch.setattr(imam_prediction_service, "model", None)
    monkeypatch.setattr(imam_prediction_service, "index_to_name", None)
    monkeypatch.setattr(imam_prediction_service, "imam_resources_error", None)
    monkeypatch.setattr(imam_prediction_service, "_import_runtime_dependencies", lambda: (fake_torch, None, None, None, Classifier))
    monkeypatch.setattr(imam_prediction_service, "_build_imam_model", lambda _nn: Model)

    imam_prediction_service.load_imam_resources()

    assert received["run_opts"]["device"] == "cpu"


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
