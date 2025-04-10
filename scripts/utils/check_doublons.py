############### CHECK DOUBLONS ###############

import json
from collections import Counter

mfcc_path = "mfcc_data/mfcc_data.jsonl"
counter = Counter()

with open(mfcc_path, "r", encoding="utf-8") as f:
    for line in f:
        try:
            data = json.loads(line.strip())
            counter[data["imam"]] += 1
        except json.JSONDecodeError:
            print("⚠️ Erreur de lecture JSON – ligne ignorée")

print("📊 Nombre de fichiers par imam :")
for imam, count in sorted(counter.items(), key=lambda x: x[0]):
    print(f"  ➤ {imam:<25} : {count}")

print(f"\n🎯 Total : {sum(counter.values())} fichiers")


import json
import hashlib
from collections import defaultdict

input_path = "mfcc_data/mfcc_data.jsonl"
output_path = "mfcc_data/mfcc_data_cleaned.jsonl"

seen = set()
unique_entries = []
hash_counter = defaultdict(int)

with open(input_path, "r", encoding="utf-8") as f:
    for line in f:
        try:
            data = json.loads(line.strip())
            key = (
                data["imam"],
                data["sourate"],
                hashlib.sha1(json.dumps(data["mfcc"]).encode()).hexdigest()  # hash MFCC
            )
            if key not in seen:
                seen.add(key)
                unique_entries.append(data)
            else:
                hash_counter[(data["imam"], data["sourate"])] += 1
        except json.JSONDecodeError:
            print("⚠️ Ligne JSON invalide ignorée.")

# Sauvegarde du nouveau fichier propre
with open(output_path, "w", encoding="utf-8") as f:
    for entry in unique_entries:
        f.write(json.dumps(entry) + "\n")

print(f"✅ Nettoyage terminé : {len(unique_entries)} lignes uniques sauvegardées.")
print(f"🧼 Fichier nettoyé : {output_path}")
print(f"🗑️ Doublons supprimés : {len(seen)} → {len(seen) + sum(hash_counter.values()) - len(unique_entries)} doublons")

# Optionnel : afficher un résumé par imam/sourate
if hash_counter:
    print("\n📌 Doublons détectés :")
    for (imam, sourate), count in sorted(hash_counter.items()):
        print(f"  ➤ {imam} - Sourate {sourate} : {count} doublon(s)")
