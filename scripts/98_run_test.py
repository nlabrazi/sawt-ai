import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import shutil
from transcribe_audio import transcribe_audio
from detect_versets import load_versets, detect_top_versets
from predict_imam import predict_imam

AUDIO_PATH = "audios/082_Dosari_live2.wav"
QURAN_VERSES_PATH = "quran_versets.json"
MODEL_PATH = "models/model_cnn_imam_v4.keras"
LABEL_ENCODER_PATH = "dataset/label_encoder_imam.pkl"


# ✅ Étape 1 - Transcription
print("🎧 Transcription complète avec Whisper...")
segments = transcribe_audio(AUDIO_PATH)
text = " ".join([s["text"] for s in segments])
print("\n📝 Transcription brute :")
print(text)
print("-" * 80)


# ✅ Étape 2 - Matching versets
print("\n🔍 Recherche des versets les plus proches...")
versets = load_versets(QURAN_VERSES_PATH)
match = detect_top_versets(segments, versets)


print(f"\n📖 Sourate {match['sourate_id']} ({match['sourate_name']}) | Versets {match['start_verse']}-{match['end_verse']}")
from termcolor import colored
score_str = colored(f"{match['similarity']*100:.2f}%", "cyan", attrs=["bold"])
print(f"🔹 Score : {score_str}")
print("🕌 Texte :", " ".join([v["text"] for v in match["verses"]]))


# ✅ Étape 3 - Prédiction imam
print("\n👳 Prédiction de l'imam (si modèle dispo)...")
top_imams = predict_imam(AUDIO_PATH, MODEL_PATH, LABEL_ENCODER_PATH)
for imam, score in top_imams:
    score_str = colored(f"{score*100:.2f}%", "magenta" if score > 0.8 else "yellow")
    print(f"  ➤ {imam:<30} : {score_str}")


# ✅ Nettoyage
shutil.rmtree(os.path.expanduser("~/.cache/whisper"), ignore_errors=True)
print("\n🧹 Cache Whisper supprimé.")
