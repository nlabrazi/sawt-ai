############### FIX OLD MFCC from 13 to 26 entires ###############

import json
import os
from tqdm import tqdm
from utils.mfcc import extract_mfcc_from_audio

INPUT_JSONL = "mfcc_data/mfcc_data_augmented.jsonl"
OUTPUT_JSONL = "mfcc_data/mfcc_data_augmented_fixed.jsonl"

with open(INPUT_JSONL, "r", encoding="utf-8") as f:
    data = [json.loads(line.strip()) for line in f]

fixed_data = []

print(f"🔍 Correction des anciennes entrées MFCC (13 valeurs)...")

for entry in tqdm(data):
    mfcc = entry["mfcc"]
    if len(mfcc) == 13:
        try:
            mfcc_fixed = extract_mfcc_from_audio(
                AUDIO_PATH=entry["file_path"]
            )
            entry["mfcc"] = mfcc_fixed
        except Exception as e:
            print(f"❌ Erreur retraitement : {entry['file_path']} → {e}")
            continue
    fixed_data.append(entry)

# 🔃 Tri par imam puis sourate
fixed_data.sort(key=lambda x: (x["imam"], x["sourate"]))

# 💾 Sauvegarde du fichier corrigé
with open(OUTPUT_JSONL, "w", encoding="utf-8") as f_out:
    for entry in fixed_data:
        f_out.write(json.dumps(entry) + "\n")

print(f"\n✅ Sauvegarde terminée dans {OUTPUT_JSONL}")
