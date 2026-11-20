from contextlib import nullcontext
from types import SimpleNamespace

import pytest

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


def _reset_imam_globals(monkeypatch):
    monkeypatch.setattr(imam_prediction_service, "encoder", None)
    monkeypatch.setattr(imam_prediction_service, "model", None)
    monkeypatch.setattr(imam_prediction_service, "index_to_name", None)
    monkeypatch.setattr(imam_prediction_service, "imam_resources_error", None)


def test_get_imam_service_health_available_states(monkeypatch):
    _reset_imam_globals(monkeypatch)
    assert imam_prediction_service.get_imam_service_health() == {
        "available": True,
        "status": "available",
        "message": None,
    }

    monkeypatch.setattr(imam_prediction_service, "encoder", object())
    monkeypatch.setattr(imam_prediction_service, "model", object())
    monkeypatch.setattr(imam_prediction_service, "index_to_name", {0: "Mishary"})
    assert imam_prediction_service.get_imam_service_health() == {
        "available": True,
        "status": "available",
        "message": None,
    }


def test_remember_imam_resources_error_messages_and_caching(monkeypatch):
    _reset_imam_globals(monkeypatch)

    generic_err = imam_prediction_service._remember_imam_resources_error(RuntimeError("boom"))
    assert "could not be initialized" in str(generic_err)
    assert imam_prediction_service.imam_resources_error is None

    mod_err = imam_prediction_service._remember_imam_resources_error(ModuleNotFoundError("torch"))
    assert "dependencies are not installed" in str(mod_err)
    assert imam_prediction_service.imam_resources_error is mod_err
    assert imam_prediction_service._remember_imam_resources_error(FileNotFoundError("x")) is mod_err


def test_build_index_to_name_map_supports_list_dict_formats_and_fallback():
    assert imam_prediction_service.build_index_to_name_map(["A", "B"]) == {0: "A", 1: "B"}
    assert imam_prediction_service.build_index_to_name_map({"0": "Mishary", "2": "Sudais"}) == {
        0: "Mishary",
        2: "Sudais",
    }
    assert imam_prediction_service.build_index_to_name_map({"Mishary": 0, "Sudais": 1}) == {
        0: "Mishary",
        1: "Sudais",
    }
    assert imam_prediction_service.build_index_to_name_map({"bad": "map"}) == {}
    assert imam_prediction_service.build_index_to_name_map(None) == {}


def test_build_imam_model_constructs_and_runs_forward():
    class FakeModule:
        pass

    fake_nn = SimpleNamespace(
        Module=FakeModule,
        Sequential=lambda *layers: (lambda x: (x, len(layers))),
        Linear=lambda a, b: ("linear", a, b),
        ReLU=lambda: "relu",
        Dropout=lambda p: ("dropout", p),
    )

    mlp_cls = imam_prediction_service._build_imam_model(fake_nn)
    instance = mlp_cls(192, 64, 3)
    assert instance.forward("vec") == ("vec", 7)


def test_load_and_preflight_imam_resources_cache_and_missing_model(monkeypatch, tmp_path):
    _reset_imam_globals(monkeypatch)
    monkeypatch.setattr(imam_prediction_service, "MODEL_PATH", tmp_path / "missing.pt")

    with pytest.raises(imam_prediction_service.ImamResourcesUnavailableError, match="Imam model not found"):
        imam_prediction_service.load_imam_resources()

    # Subsequent calls re-raise the cached error directly
    with pytest.raises(imam_prediction_service.ImamResourcesUnavailableError, match="Imam model not found"):
        imam_prediction_service.load_imam_resources()

    _reset_imam_globals(monkeypatch)
    cached_encoder, cached_model, cached_map = object(), object(), {0: "Alafasy"}
    monkeypatch.setattr(imam_prediction_service, "encoder", cached_encoder)
    monkeypatch.setattr(imam_prediction_service, "model", cached_model)
    monkeypatch.setattr(imam_prediction_service, "index_to_name", cached_map)

    assert imam_prediction_service.load_imam_resources() == (cached_encoder, cached_model, cached_map)
    imam_prediction_service.preflight_imam_resources()


def test_load_audio_and_extract_embedding_downmix_resample_and_squeeze(monkeypatch):
    class FakeTensor:
        def __init__(self, shape, dims=None):
            self.shape = shape
            self._dims = dims if dims is not None else len(shape)
            self.ops = []

        def mean(self, dim, keepdim):
            self.ops.append(("mean", dim, keepdim))
            self.shape = (1, self.shape[1])
            return self

        def squeeze(self, dim):
            self.ops.append(("squeeze", dim))
            return self

        def unsqueeze(self, dim):
            self.ops.append(("unsqueeze", dim))
            return self

        def to(self, device):
            self.ops.append(("to", device))
            return self

        def dim(self):
            return self._dims

        def cpu(self):
            self.ops.append(("cpu",))
            return self

    stereo = FakeTensor((2, 100))
    resampled = []

    class FakeResample:
        def __init__(self, src_sr, dst_sr):
            resampled.append((src_sr, dst_sr))

        def __call__(self, wf):
            wf.ops.append(("resampled",))
            return wf

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        device=lambda name: name,
        no_grad=lambda: nullcontext(),
    )
    fake_torchaudio = SimpleNamespace(
        load=lambda _path: (stereo, 44100),
        transforms=SimpleNamespace(Resample=FakeResample),
    )
    monkeypatch.setattr(
        imam_prediction_service,
        "_import_runtime_dependencies",
        lambda: (fake_torch, None, fake_torchaudio, None, None),
    )

    loaded = imam_prediction_service.load_audio("sample.wav")
    assert resampled == [(44100, imam_prediction_service.TARGET_SAMPLE_RATE)]
    assert ("mean", 0, True) in loaded.ops
    assert ("resampled",) in loaded.ops
    assert ("squeeze", 0) in loaded.ops

    emb_tensor = FakeTensor((1, 1, 192), dims=3)
    fake_encoder = SimpleNamespace(encode_batch=lambda _x: emb_tensor)
    embedding = imam_prediction_service.extract_embedding(FakeTensor((100,)), fake_encoder)
    assert ("squeeze", 1) in embedding.ops
    assert ("squeeze", 0) in embedding.ops
    assert ("cpu",) in embedding.ops


def test_predict_imam_returns_ranked_results_and_maps_errors(monkeypatch):
    _reset_imam_globals(monkeypatch)

    class FakeVector:
        def __init__(self, values):
            self._values = values
            self.shape = (len(values),)

        def unsqueeze(self, _dim):
            return self

        def to(self, _device):
            return self

        def cpu(self):
            return self

        def tolist(self):
            return list(self._values)

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        device=lambda name: name,
        no_grad=lambda: nullcontext(),
        topk=lambda probs, k: (FakeVector([0.75, 0.25][:k]), FakeVector([0, 2][:k])),
    )
    fake_F = SimpleNamespace(softmax=lambda _logits, dim: [FakeVector([0.75, 0.0, 0.25])])

    monkeypatch.setattr(
        imam_prediction_service,
        "_import_runtime_dependencies",
        lambda: (fake_torch, None, None, fake_F, None),
    )
    monkeypatch.setattr(
        imam_prediction_service,
        "load_imam_resources",
        lambda: (object(), lambda _x: "logits", {0: "Mishary"}),
    )
    monkeypatch.setattr(imam_prediction_service, "load_audio", lambda _p: "audio")
    monkeypatch.setattr(
        imam_prediction_service,
        "extract_embedding",
        lambda _a, _e: FakeVector([0.1, 0.2]),
    )

    predictions = imam_prediction_service.predict_imam("recitation.wav")
    assert predictions == [
        {"name": "Mishary", "score": 0.75},
        {"name": "class_2", "score": 0.25},
    ]

    # Propagates ImamResourcesUnavailableError unchanged
    monkeypatch.setattr(
        imam_prediction_service,
        "load_imam_resources",
        lambda: (_ for _ in ()).throw(imam_prediction_service.ImamResourcesUnavailableError("no model")),
    )
    with pytest.raises(imam_prediction_service.ImamResourcesUnavailableError, match="no model"):
        imam_prediction_service.predict_imam("recitation.wav")

    # ModuleNotFoundError becomes ImamResourcesUnavailableError
    _reset_imam_globals(monkeypatch)
    monkeypatch.setattr(
        imam_prediction_service,
        "_import_runtime_dependencies",
        lambda: (_ for _ in ()).throw(ModuleNotFoundError("speechbrain")),
    )
    with pytest.raises(imam_prediction_service.ImamResourcesUnavailableError, match="dependencies"):
        imam_prediction_service.predict_imam("recitation.wav")

    # Unexpected error wraps into ImamPredictionError
    _reset_imam_globals(monkeypatch)
    monkeypatch.setattr(
        imam_prediction_service,
        "_import_runtime_dependencies",
        lambda: (fake_torch, None, None, fake_F, None),
    )
    monkeypatch.setattr(
        imam_prediction_service,
        "load_imam_resources",
        lambda: (_ for _ in ()).throw(RuntimeError("unexpected")),
    )
    with pytest.raises(imam_prediction_service.ImamPredictionError, match="Imam prediction failed"):
        imam_prediction_service.predict_imam("recitation.wav")
