import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import json
import shutil
import numpy as np
import whisper
import pickle

from termcolor import colored
from time import time, sleep
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from difflib import SequenceMatcher
from rich.progress import Progress, BarColumn, TimeElapsedColumn, TextColumn

# 🔁 Import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.mfcc import extract_mfcc_from_audio

# 📍 Config
AUDIO_PATH = "audios/Turkmensitan_03.mp3"
MODEL_PATH = "models/model_cnn_imam_v2.keras"
JSONL_PATH = "mfcc_data/mfcc_data_v2.jsonl"
QURAN_VERSES_PATH = "quran_versets.json"
LABELS_PATH = "dataset/labels.json"

# 🔧 Paramètres
SCORE_HIGH = 0.60
SCORE_MEDIUM = 0.35
WINDOW_SIZES = [1, 2, 3, 4, 5]

# 🧠 Chargement du Coran
with open(QURAN_VERSES_PATH, "r", encoding="utf-8") as f:
    versets = json.load(f)

# 📚 Indexation des combinaisons versets
combinations = []
for sourate in versets:
    verses = sourate["verses"]
    for w in WINDOW_SIZES:
        for i in range(len(verses) - w + 1):
            segment = verses[i:i+w]
            combined_text = " ".join(v["text"] for v in segment)
            combinations.append({
                "sourate_id": sourate["id"],
                "sourate_name": sourate["name"],
                "start_verse": segment[0]["id"],
                "end_verse": segment[-1]["id"],
                "verses": segment,
                "text": combined_text
            })

# 🔧 Nettoyage
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
print("🎧 Transcription (streaming)...")
model_whisper = whisper.load_model("medium")
result = model_whisper.transcribe(AUDIO_PATH, language="ar", verbose=False)
segments = result["segments"]

# 🔍 Analyse avec early stop + barre custom
best_match = None
best_score = 0.0
transcription_progressive = ""

with Progress(
    TextColumn("🔍 Analyse audio...", justify="left"),
    BarColumn(bar_width=None, complete_style="bold magenta"),
    TextColumn("{task.percentage:>3.0f}%"),
    TimeElapsedColumn(),
) as progress:
    task = progress.add_task("analyse", total=len(segments))

    for segment in segments:
        transcription_progressive += segment["text"]
        current_text = transcription_progressive.strip()

        for item in combinations:
            score = SequenceMatcher(None, current_text, item["text"]).ratio()

            if score > best_score:
                best_score = score
                best_match = {**item, "similarity": score}

            # 🎯 Early stop conditions
            individual_scores = [SequenceMatcher(None, current_text, v["text"]).ratio() for v in item["verses"]]
            if any(s >= SCORE_HIGH for s in individual_scores):
                progress.update(task, completed=len(segments))
                break
            for i in range(len(individual_scores) - 1):
                if individual_scores[i] >= SCORE_MEDIUM and individual_scores[i+1] >= SCORE_MEDIUM:
                    progress.update(task, completed=len(segments))
                    break

        progress.update(task, advance=1)

# 📊 Résultat final
if best_match:
    print("\n🔍 Meilleur match :")
    print(f"📖 Sourate {best_match['sourate_id']} | Versets {best_match['start_verse']} à {best_match['end_verse']}")
    print(f"🕌 Texte : {' '.join([v['text'] for v in best_match['verses']])}")
    print(f"🔹 Score global : {best_match['similarity']:.2%}\n")
else:
    print("❌ Aucun verset détecté.")
    exit()

# 🔊 Extraction MFCC + prédiction imam
y, sr = whisper.audio.load_audio(AUDIO_PATH), whisper.audio.SAMPLE_RATE
mfcc_values = extract_mfcc_from_audio(y, sr)
model = load_model(MODEL_PATH)
preds = model.predict(np.array([mfcc_values]))[0]

# 🔎 Chargement des labels
with open("dataset/label_encoder_imam.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# 🔝 TOP 3 imams
top_indices = preds.argsort()[-3:][::-1]
print("\n👤 TOP 3 des imams les plus probables :")
for idx in top_indices:
    imam = label_encoder.inverse_transform([idx])[0]
    print(f"  {imam:<30} : {preds[idx]*100:.1f}%")

# 🧹 Nettoyage Whisper
shutil.rmtree(os.path.expanduser("~/.cache/whisper"), ignore_errors=True)
print("\n🧹 Cache Whisper supprimé.")
