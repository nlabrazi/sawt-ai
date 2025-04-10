import json
from collections import defaultdict

JSONL_PATH = "mfcc_data/mfcc_data_ultracleaned.jsonl"
TARGET_SOURATE = 112

counts = defaultdict(int)

with open(JSONL_PATH, "r", encoding="utf-8") as f:
    for line in f:
        entry = json.loads(line)
        if entry["sourate"] == TARGET_SOURATE:
            counts[entry["imam"]] += 1

print(f"📊 Nombre d'échantillons pour la sourate {TARGET_SOURATE} :")
for imam, count in sorted(counts.items(), key=lambda x: -x[1]):
    print(f"  {imam:<30} : {count}")
