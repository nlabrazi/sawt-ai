import os
import json
import librosa
import scipy.signal
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from tqdm import tqdm
from utils.mfcc import extract_mfcc_from_audio

# 📁 Chemins
AUDIO_BASE_PATH = "/mnt/e/DIN/data"
OUTPUT_JSONL = "mfcc_data/mfcc_data_augmented.jsonl"
MAX_WORKERS = 4
os.makedirs("mfcc_data", exist_ok=True)

# 🔧 Augmentations
def apply_pitch_shift(y, sr): return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=2)
def apply_time_stretch(y, sr): return librosa.effects.time_stretch(y, rate=1.1)
def apply_white_noise(y, sr): return y + 0.005 * np.random.randn(len(y))
def apply_low_pass_filter(y, sr):
    b, a = scipy.signal.butter(6, 0.1, btype='low', analog=False)
    return scipy.signal.lfilter(b, a, y)
def apply_random_gain(y, sr):
    gain = np.random.uniform(0.7, 1.3)
    return y * gain
def apply_background_noise(y, sr):
    noise = np.random.normal(0, 0.003, len(y))
    return y + noise
from scipy.signal import fftconvolve
def apply_reverb(y, sr):
    impulse = np.zeros_like(y)
    impulse[0] = 1.0
    impulse[5000:] += 0.5 * np.random.randn(len(y) - 5000)
    return fftconvolve(y, impulse, mode='full')[:len(y)]

# 🎛️ Registre des augmentations
AUGMENTATIONS = {
    "original": lambda y, sr: y,
    "pitch_shift": apply_pitch_shift,
    "time_stretch": apply_time_stretch,
    "white_noise": apply_white_noise,
    "reverb": apply_reverb,
    "low_pass_filter": apply_low_pass_filter,
    "random_gain": apply_random_gain,
    "background_noise": apply_background_noise,
}

# 🧠 Charger les clés déjà présentes
existing_keys = set()
if os.path.exists(OUTPUT_JSONL):
    with open(OUTPUT_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                key = (data["imam"], data["sourate"], data["file_path"], data["augmentation"])
                existing_keys.add(key)
            except json.JSONDecodeError:
                continue

# 🔐 Verrou pour accès au fichier
write_lock = Lock()
mfcc_counter = 0  # Compteur global

# 🔧 Fonction de traitement
def process_file(file_info):
    global mfcc_counter
    file_path, imam, sourate = file_info

    try:
        y, sr = librosa.load(file_path, sr=None)
    except Exception as e:
        tqdm.write(f"❌ Erreur chargement : {file_path} → {e}")
        return

    for aug_name, aug_func in AUGMENTATIONS.items():
        key = (imam, sourate, file_path, aug_name)
        if key in existing_keys:
            continue

        # tqdm.write(f"🛠️  {imam} | Sourate {sourate} | Augmentation : {aug_name}")

        try:
            y_aug = aug_func(y, sr)
            mfcc_mean = extract_mfcc_from_audio(y=y_aug, sr=sr)

            result = {
                "imam": imam,
                "sourate": sourate,
                "file_path": file_path,
                "augmentation": aug_name,
                "mfcc": mfcc_mean
            }

            with write_lock:
                with open(OUTPUT_JSONL, "a", encoding="utf-8") as f_out:
                    f_out.write(json.dumps(result) + "\n")
                existing_keys.add(key)
                mfcc_counter += 1

        except Exception as e:
            tqdm.write(f"❌ Erreur augmentation ({aug_name}) : {file_path} → {e}")

# 📂 Collecte des fichiers audio
files_to_process = []
for imam in os.listdir(AUDIO_BASE_PATH):
    imam_path = os.path.join(AUDIO_BASE_PATH, imam)
    if not os.path.isdir(imam_path):
        continue
    for filename in os.listdir(imam_path):
        if filename.endswith(".mp3") or filename.endswith(".wav"):
            file_path = os.path.join(imam_path, filename)
            try:
                sourate = int(filename.split("-")[0])
                files_to_process.append((file_path, imam, sourate))
            except ValueError:
                print(f"⚠️ Fichier ignoré : {filename}")

print(f"🚀 Lancement du traitement sur {len(files_to_process)} fichiers audio...")

# 🔄 Exécution multi-thread + barre de progression
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    with tqdm(total=len(files_to_process), desc="Traitement des fichiers", dynamic_ncols=True) as pbar:

        def wrapper(file_info):
            process_file(file_info)
            pbar.update(1)

        list(executor.map(wrapper, files_to_process))

print(f"\n✅ Tous les fichiers traités.")
print(f"📊 Nombre total de MFCCs extraits : {mfcc_counter}")

# 🧹 Tri du fichier JSONL final
print("🔄 Tri du fichier JSONL final...")

with open(OUTPUT_JSONL, "r", encoding="utf-8") as f:
    lines = [json.loads(line.strip()) for line in f if line.strip()]

sorted_lines = sorted(
    lines,
    key=lambda x: (x["imam"], int(x["sourate"]), x["augmentation"])
)

with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
    for line in sorted_lines:
        f.write(json.dumps(line) + "\n")

print("✅ Fichier trié par imam, sourate, augmentation.")
