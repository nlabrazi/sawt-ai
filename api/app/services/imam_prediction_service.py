from __future__ import annotations

# ROLE
# ----
# Prédit les 3 imams les plus probables à partir de l'audio.

import logging
import os
from pathlib import Path
from typing import Any

TARGET_SAMPLE_RATE = 16000
DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[3]
    / "training"
    / "artifacts"
    / "models"
    / "imam_ecapa_v2"
    / "best_model.pt"
)
MODEL_PATH = Path(os.getenv("IMAM_MODEL_PATH", str(DEFAULT_MODEL_PATH)))
logger = logging.getLogger(__name__)

encoder: Any | None = None
model: Any | None = None
index_to_name: dict[int, str] | None = None
imam_resources_error: Exception | None = None


class ImamPredictionError(Exception):
    pass


class ImamResourcesUnavailableError(ImamPredictionError):
    pass


def _get_imam_resources_error_message(exc: Exception) -> str:
    if isinstance(exc, FileNotFoundError):
        return f"Imam model not found: {MODEL_PATH}"

    if isinstance(exc, ModuleNotFoundError):
        return "Imam prediction dependencies are not installed."

    return "Imam prediction resources could not be initialized."


def _remember_imam_resources_error(exc: Exception) -> ImamResourcesUnavailableError:
    global imam_resources_error

    if isinstance(imam_resources_error, ImamResourcesUnavailableError):
        return imam_resources_error

    error = ImamResourcesUnavailableError(
        _get_imam_resources_error_message(exc)
    )

    if isinstance(exc, (FileNotFoundError, ModuleNotFoundError)):
        imam_resources_error = error

    return error


def build_index_to_name_map(label_map: object) -> dict[int, str]:
    if isinstance(label_map, list):
        return {i: str(name) for i, name in enumerate(label_map)}

    if isinstance(label_map, dict):
        # cas 1: {"0": "Mishary"} ou {0: "Mishary"}
        if all(str(k).isdigit() for k in label_map.keys()):
            return {int(k): str(v) for k, v in label_map.items()}

        # cas 2: {"Mishary": 0}
        if all(str(v).isdigit() for v in label_map.values()):
            return {int(v): str(k) for k, v in label_map.items()}

    return {}


def _import_runtime_dependencies():
    import torch
    import torch.nn as nn
    import torchaudio
    import torch.nn.functional as F
    from speechbrain.inference.classifiers import EncoderClassifier

    return torch, nn, torchaudio, F, EncoderClassifier


def _build_imam_model(nn_module):
    class ImamEmbeddingMLP(nn_module.Module):
        def __init__(self, input_dim, hidden_dim, num_classes):
            super().__init__()

            self.net = nn_module.Sequential(
                nn_module.Linear(input_dim, hidden_dim),
                nn_module.ReLU(),
                nn_module.Dropout(0.2),
                nn_module.Linear(hidden_dim, hidden_dim),
                nn_module.ReLU(),
                nn_module.Dropout(0.2),
                nn_module.Linear(hidden_dim, num_classes),
            )

        def forward(self, x):
            return self.net(x)

    return ImamEmbeddingMLP


def load_imam_resources() -> tuple[Any, Any, dict[int, str]]:
    global encoder, model, index_to_name

    if encoder is not None and model is not None and index_to_name is not None:
        return encoder, model, index_to_name

    if imam_resources_error is not None:
        raise imam_resources_error

    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Imam model not found: {MODEL_PATH}")

        torch, nn, _torchaudio, _F, EncoderClassifier = _import_runtime_dependencies()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ImamEmbeddingMLP = _build_imam_model(nn)

        if encoder is None:
            encoder = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                run_opts={"device": device},
            )

        checkpoint = torch.load(MODEL_PATH, map_location=device)

        loaded_model = ImamEmbeddingMLP(
            checkpoint["input_dim"],
            checkpoint["hidden_dim"],
            checkpoint["num_classes"],
        )
        loaded_model.load_state_dict(checkpoint["model_state_dict"])
        loaded_model.to(device)
        loaded_model.eval()

        loaded_index_to_name = build_index_to_name_map(checkpoint.get("labels", {}))

        model = loaded_model
        index_to_name = loaded_index_to_name

        logger.info("Imam prediction model loaded from %s", MODEL_PATH)

        return encoder, model, index_to_name
    except Exception as exc:
        unavailable_error = _remember_imam_resources_error(exc)
        logger.exception("Imam prediction resources are unavailable")
        raise unavailable_error from exc


def load_audio(audio_path: str):
    torch, _nn, torchaudio, _F, _EncoderClassifier = _import_runtime_dependencies()
    waveform, sr = torchaudio.load(audio_path)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != TARGET_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, TARGET_SAMPLE_RATE)
        waveform = resampler(waveform)

    return waveform.squeeze(0)


def extract_embedding(
    audio,
    loaded_encoder,
):
    torch, _nn, _torchaudio, _F, _EncoderClassifier = _import_runtime_dependencies()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        emb = loaded_encoder.encode_batch(audio.unsqueeze(0).to(device))

        if emb.dim() == 3:
            emb = emb.squeeze(1)

        return emb.squeeze(0).cpu()


def predict_imam(audio_path: str) -> list[dict[str, str | float]]:
    try:
        torch, _nn, _torchaudio, F, _EncoderClassifier = _import_runtime_dependencies()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loaded_encoder, loaded_model, loaded_index_to_name = load_imam_resources()
        audio = load_audio(audio_path)
        embedding = extract_embedding(audio, loaded_encoder)

        with torch.no_grad():
            logits = loaded_model(embedding.unsqueeze(0).to(device))
            probs = F.softmax(logits, dim=1)[0].cpu()

        k = min(3, probs.shape[0])
        top_scores, top_indices = torch.topk(probs, k=k)

        results = []

        for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
            name = loaded_index_to_name.get(int(idx), f"class_{int(idx)}")

            results.append({
                "name": name,
                "score": float(score),
            })

        return results
    except ImamResourcesUnavailableError:
        raise
    except ModuleNotFoundError as exc:
        unavailable_error = _remember_imam_resources_error(exc)
        logger.exception("Imam prediction dependencies are unavailable")
        raise unavailable_error from exc
    except Exception as exc:
        logger.exception("Imam prediction failed")
        raise ImamPredictionError("Imam prediction failed.") from exc
