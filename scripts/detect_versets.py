import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import json
from difflib import SequenceMatcher
from utils.normalize_arabic import normalize_arabic

def load_versets(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def detect_top_versets(segments, versets, window_sizes=[1, 2, 3, 4, 5], top_k=5):
    """
    Retourne les top_k versets ou combinaisons les plus proches
    """
    transcription = normalize_arabic(" ".join([s["text"] for s in segments]).strip())
    matches = []

    for sourate in versets:
        verses = sourate["verses"]
        for w in window_sizes:
            for i in range(len(verses) - w + 1):
                segment = verses[i:i + w]
                combined_text = normalize_arabic(" ".join(v["text"] for v in segment))
                score = SequenceMatcher(None, transcription, combined_text).ratio()

                matches.append({
                    "sourate_id": sourate["id"],
                    "sourate_name": sourate["name"],
                    "start_verse": segment[0]["id"],
                    "end_verse": segment[-1]["id"],
                    "verses": segment,
                    "text": combined_text,
                    "similarity": score
                })

    matches.sort(key=lambda x: -x["similarity"])
    return matches[0]
