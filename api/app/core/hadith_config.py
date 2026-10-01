"""Read on demand: Hadith must never block Quran startup."""

import os
from dataclasses import dataclass
from pathlib import Path

ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets"

@dataclass(frozen=True)
class HadithConfig:
    base_url: str
    language: str
    model_name: str
    strategy: str
    index_path: Path
    meta_path: Path

    @classmethod
    def from_env(cls):
        return cls(
            base_url=os.getenv("HADEETHENC_BASE_URL", "https://hadeethenc.com/api/v1").rstrip("/"),
            language=os.getenv("HADITH_LANGUAGE", "fr"),
            model_name=os.getenv("HADITH_EMBEDDING_MODEL", "intfloat/multilingual-e5-base"),
            strategy=os.getenv("HADITH_INDEX_STRATEGY", "multi_context"),
            index_path=Path(os.getenv("HADITH_INDEX_PATH", str(ASSETS_DIR / "hadith_index.npz"))),
            meta_path=Path(os.getenv("HADITH_INDEX_META_PATH", str(ASSETS_DIR / "hadith_index_meta.json"))),
        )
