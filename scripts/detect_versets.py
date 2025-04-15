import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import json
from rapidfuzz.fuzz import ratio
from utils.normalize_arabic import normalize_arabic
from rich.progress import Progress, BarColumn, TimeElapsedColumn, TextColumn

SCORE_HIGH = 0.60
SCORE_MEDIUM = 0.35

def load_versets(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def detect_best_match(segments, versets, high_threshold, medium_threshold, progress, task):
    WINDOW_SIZES = [1, 2, 3, 4, 5]
    combinations = []
    for sourate in versets:
        verses = sourate["verses"]
        for w in WINDOW_SIZES:
            for i in range(len(verses) - w + 1):
                segment = verses[i:i + w]
                combined_text = " ".join(v["text"] for v in segment)
                combinations.append({
                    "sourate_id": sourate["id"],
                    "sourate_name": sourate["name"],
                    "start_verse": segment[0]["id"],
                    "end_verse": segment[-1]["id"],
                    "verses": segment,
                    "text": normalize_arabic(combined_text)
                })

    best_match = None
    best_score = 0.0

    for segment in segments:
        current_text = normalize_arabic(segment["text"])
        if not current_text.strip():
            progress.update(task, advance=1)
            continue

        for item in combinations:
            score = ratio(current_text, item["text"]) / 100
            if score > best_score:
                best_score = score
                best_match = {**item, "similarity": score}

            individual_scores = [
                ratio(current_text, normalize_arabic(v["text"])) / 100
                for v in item["verses"]
            ]

            if any(s >= high_threshold for s in individual_scores):
                progress.update(task, completed=progress.tasks[task].total)
                return best_match

            for i in range(len(individual_scores) - 1):
                if individual_scores[i] >= medium_threshold and individual_scores[i + 1] >= medium_threshold:
                    progress.update(task, completed=progress.tasks[task].total)
                    return best_match

        progress.update(task, advance=1)

    return best_match
