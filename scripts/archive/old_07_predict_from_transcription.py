import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import json
import shutil
import numpy as np
import whisper
import pickle
import io
import contextlib

from termcolor import colored
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from difflib import SequenceMatcher
from utils.mfcc import extract_mfcc_from_audio

# 📍 Config
AUDIO_PATH = "audios/Turkmensitan_03.mp3"
MODEL_PATH = "models/model_cnn_imam_v2.keras"
JSONL_PATH = "mfcc_data/mfcc_data_v2.jsonl"
QURAN_VERSES_PATH = "quran_versets.json"
LABELS_PATH = "dataset/labels.json"

SCORE_HIGH = 0.60
SCORE_MEDIUM = 0.35

# 🧠 Load Quran versets
with open(QURAN_VERSES_PATH, "r", encoding="utf-8") as f:
    versets = json.load(f)

# 🧼 Nettoyage transcription
SUPPRESSIONS = [
    "أعوذ بالله من الشيطان الرجيم",
    "بسم الله الرحمن الرحيم",
    "بسم الله الرحمن الر حيم",
]
def clean_text(t):
    for s in SUPPRESSIONS:
        t = t.replace(s, "")
    return t.strip()

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

# 🧠 Pré-indexation des fenêtres de versets
indexed_segments = []
for sourate in versets:
    for window in range(1, 6):
        for i in range(len(sourate["verses"]) - window + 1):
            segment = sourate["verses"][i:i+window]
            indexed_segments.append({
                "sourate_id": sourate["id"],
                "sourate_name": sourate["name"],
                "start": segment[0]["id"],
                "end": segment[-1]["id"],
                "verses": segment,
                "text": " ".join(v["text"] for v in segment)
            })

# 🎧 Transcription segmentée (sans affichage)
print("🎧 Transcription (streaming)...")
model_whisper = whisper.load_model("medium")
with contextlib.redirect_stdout(io.StringIO()):
    result = model_whisper.transcribe(AUDIO_PATH, language="ar", word_timestamps=False, verbose=False)
segments = result["segments"]

transcription = ""
best_match, best_score = None, 0
progress_len = 60

for idx, segment in enumerate(segments):
    transcription += segment["text"]
    transcription_clean = clean_text(transcription)

    for entry in indexed_segments:
        score = similar(transcription_clean, entry["text"])
        if score > best_score:
            best_score = score
            best_match = entry
            best_match["similarity"] = score

        # EARLY STOP
        individual_scores = [similar(transcription_clean, v["text"]) for v in entry["verses"]]
        if any(s >= SCORE_HIGH for s in individual_scores):
            break
        for j in range(len(individual_scores) - 1):
            if individual_scores[j] >= SCORE_MEDIUM and individual_scores[j+1] >= SCORE_MEDIUM:
                break

    # 🔄 Affichage de la barre colorée
    pct = int(((idx+1) / len(segments)) * 100)
    bar = "".join(["\033[92m█" if i < pct else "\033[91m█" for i in range(progress_len)])
    print(f"\r{bar}\033[0m {pct}% analysé", end="", flush=True)

print("\n")

# 🧾 Affichage du meilleur match
if best_match:
    print("🔍 Meilleur match :")
    print(f"📖 Sourate {best_match['sourate_id']} | Versets {best_match['start']} à {best_match['end']}")
    print(f"🕌 Texte : {' '.join([v['text'] for v in best_match['verses']])}")
    print(f"🔹 Score global : {best_match['similarity']:.2%}\n")
else:
    print("❌ Aucun verset détecté.")
    exit()

# 🧠 Prédiction de l'imam
sourate_target = best_match["sourate_id"]
X, y = [], []
with open(JSONL_PATH, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        if data["sourate"] == sourate_target:
            X.append(data["mfcc"])
            y.append(data["imam"])
if not X:
    print("❌ Aucun échantillon MFCC trouvé pour cette sourate.")
    exit()
X = np.array(X)
y = np.array(y)

with open("dataset/label_encoder_imam.pkl", "rb") as f:
    label_encoder = pickle.load(f)
model = load_model(MODEL_PATH)

# Extraction MFCC en mémoire
import librosa
signal, sr = librosa.load(AUDIO_PATH, sr=None)
mfcc_test = extract_mfcc_from_audio(y=signal, sr=sr)

preds = model.predict(np.array([mfcc_test]))[0]
top_indices = preds.argsort()[-3:][::-1]

print("\n👤 TOP 3 des imams les plus probables :")
for idx in top_indices:
    imam = label_encoder.inverse_transform([idx])[0]
    print(f"  {imam:<30} : {preds[idx]*100:.1f}%")

# Nettoyage cache
shutil.rmtree(os.path.expanduser("~/.cache/whisper"), ignore_errors=True)
print("\n🧹 Cache Whisper supprimé.")
