# ROLE
# ----
# Prédit les 3 imams les plus probables à partir de l'audio.

from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from speechbrain.inference.classifiers import EncoderClassifier

TARGET_SAMPLE_RATE = 16000
MODEL_PATH = Path("/training/artifacts/models/imam_ecapa_v2/best_model.pt")
SUMMARY_PATH = Path("/training/artifacts/models/imam_ecapa_v2/summary.json")

print("MODEL_PATH =", MODEL_PATH)
print("MODEL_EXISTS =", MODEL_PATH.exists())


def build_index_to_name_map(label_map):
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


class ImamEmbeddingMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": device},
)

checkpoint = torch.load(MODEL_PATH, map_location=device)

model = ImamEmbeddingMLP(
    checkpoint["input_dim"],
    checkpoint["hidden_dim"],
    checkpoint["num_classes"],
)

model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

raw_label_map = checkpoint.get("labels", {})
print("RAW_LABEL_MAP =", raw_label_map)

index_to_name = build_index_to_name_map(raw_label_map)
print("INDEX_TO_NAME =", index_to_name)


def load_audio(audio_path):
    waveform, sr = torchaudio.load(audio_path)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != TARGET_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, TARGET_SAMPLE_RATE)
        waveform = resampler(waveform)

    return waveform.squeeze(0)


def extract_embedding(audio):
    with torch.no_grad():
        emb = encoder.encode_batch(audio.unsqueeze(0).to(device))

        if emb.dim() == 3:
            emb = emb.squeeze(1)

        return emb.squeeze(0).cpu()


def predict_imam(audio_path):
    try:
        audio = load_audio(audio_path)
        embedding = extract_embedding(audio)

        with torch.no_grad():
            logits = model(embedding.unsqueeze(0).to(device))
            probs = F.softmax(logits, dim=1)[0].cpu()

        k = min(3, probs.shape[0])
        top_scores, top_indices = torch.topk(probs, k=k)

        results = []

        for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
            name = index_to_name.get(int(idx), f"class_{int(idx)}")

            results.append({
                "name": name,
                "score": float(score),
            })

        return results

    except Exception as e:
        print("[ERROR] imam prediction failed:", repr(e))
        return []
