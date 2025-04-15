import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import io
import json
import numpy as np
import pickle
import shutil
import whisper

import contextlib
# Suppression des logs stdout de Whisper (ex. : chargement modèle)
with contextlib.redirect_stdout(io.StringIO()):
    model = whisper.load_model("large")

from termcolor import colored
from tensorflow.keras.models import load_model
from rich.progress import Progress, BarColumn, TimeElapsedColumn, TextColumn

# 🔁 Import utils
from utils.mfcc import extract_mfcc_from_audio
from transcribe_audio import transcribe_audio
from detect_versets import load_versets, detect_best_match
from predict_imam import predict_imam

# 📍 Config
AUDIO_PATH = "audios/082_test.wav"
MODEL_PATH = "models/model_cnn_imam_v3.keras"
JSONL_PATH = "mfcc_data/mfcc_data_v2.jsonl"
QURAN_VERSES_PATH = "quran_versets.json"
LABEL_ENCODER_PATH = "dataset/label_encoder_imam.pkl"

# 🎧 Transcription (streaming)
print("🎧 Transcription (streaming)...")
segments = transcribe_audio(AUDIO_PATH)
# 🔎 Affiche la transcription brute concaténée (debug)
transcribed_text = "".join([s["text"] for s in segments]).strip()
print("\n📝 Transcription complète (debug) :")
print(transcribed_text)
print("-" * 80)

# 📖 Chargement des versets
versets = load_versets(QURAN_VERSES_PATH)

# 🔍 Détection versets avec progress
with Progress(
    TextColumn("🔍 Analyse audio...", justify="left"),
    BarColumn(bar_width=None, complete_style="bold green"),
    TextColumn("{task.percentage:>3.0f}%"),
    TimeElapsedColumn(),
) as progress:
    task = progress.add_task("analyse", total=len(segments))
    SCORE_HIGH = 0.60
    SCORE_MEDIUM = 0.35
    best_match = detect_best_match(segments, versets, SCORE_HIGH, SCORE_MEDIUM, progress, task)

# 📊 Résultat final
if best_match:
    print("\n🔍 Meilleur match :")
    print(f"📖 Sourate {best_match['sourate_id']} | Versets {best_match['start_verse']} à {best_match['end_verse']}")
    print(f"🕌 Texte : {' '.join([v['text'] for v in best_match['verses']])}")
    print(f"🔹 Score : {best_match['similarity'] * 100:.2f}%\n")

    # 🧠 Prédiction imam
    top_imams = predict_imam(AUDIO_PATH, MODEL_PATH, JSONL_PATH, best_match['sourate_id'], LABEL_ENCODER_PATH)
    print("👤 TOP 3 des imams les plus probables :")
    for imam, score in top_imams:
        print(f"  {imam:<30} : {score * 100:.1f}%")
else:
    print("❌ Aucun verset significatif détecté.")

# 🧹 Nettoyage Whisper
shutil.rmtree(os.path.expanduser("~/.cache/whisper"), ignore_errors=True)
print("\n🧹 Cache Whisper supprimé.")
