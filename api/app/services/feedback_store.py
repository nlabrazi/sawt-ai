# ROLE
# ----
# Stockage V1 du feedback utilisateur dans un fichier JSONL local.
# Chaque ligne = 1 feedback.

import json
from datetime import datetime, timezone
from pathlib import Path

FEEDBACK_FILE = Path("data/feedback.jsonl")


def save_feedback(payload: dict) -> None:
    FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        **payload,
    }

    with FEEDBACK_FILE.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")
