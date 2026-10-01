"""In-memory cosine retrieval. Heavy dependencies are imported on demand."""

import hashlib
import json
from threading import Lock

from app.core.hadith_config import HadithConfig


class HadithIndexError(Exception):
    pass


def load_embedding_model(model_name: str, revision: str | None = None):
    from sentence_transformers import SentenceTransformer

    # Keep inference on CPU, independently of the Quran models.
    return SentenceTransformer(model_name, revision=revision, device="cpu", trust_remote_code=False)


def load_index(config: HadithConfig):
    import numpy as np

    try:
        meta = json.loads(config.meta_path.read_text(encoding="utf-8"))
        if meta["schema_version"] not in (1, 2) or meta["model"] != config.model_name or meta["language"] != config.language:
            raise ValueError("Index configuration mismatch")
        if hashlib.sha256(config.index_path.read_bytes()).hexdigest() != meta["index_sha256"]:
            raise ValueError("Index/metadata checksum mismatch")
        with np.load(config.index_path, allow_pickle=False) as archive:
            embeddings = archive["embeddings"].astype(np.float32)
        rows = meta["items"]
        ids = [row["hadeethenc_id"] for row in rows]
        # Multi-embedding strategies store several passages per hadith, so
        # duplicate IDs are expected and intentional — do not require uniqueness.
        if not ids or any(not isinstance(i, str) or not i.isascii() or not i.isdigit() for i in ids):
            raise ValueError("Invalid IDs")
        if any(row["language"] != config.language for row in rows):
            raise ValueError("Invalid language")
        if embeddings.ndim != 2 or embeddings.shape != (len(ids), meta["dimension"]) or not np.isfinite(embeddings).all():
            raise ValueError("Invalid embedding matrix")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        if (norms <= 0).any():
            raise ValueError("Zero embedding")
        return embeddings / norms, ids, meta
    except Exception as exc:
        raise HadithIndexError("Index Hadith absent, incompatible ou invalide.") from exc


def load_search_documents(config: HadithConfig) -> dict[str, str]:
    """Source snapshot is search evidence only; results still come from the API."""
    try:
        meta = json.loads(config.meta_path.read_text(encoding="utf-8"))
        if (
            meta["schema_version"] != 2 or meta["model"] != config.model_name
            or meta["language"] != config.language or meta["base_url"] != config.base_url
        ):
            raise ValueError("Rebuild or upgrade the Hadith index to schema 2")
        if hashlib.sha256(config.index_path.read_bytes()).hexdigest() != meta["index_sha256"]:
            raise ValueError("Index/metadata checksum mismatch")
        documents = meta["search_documents"]
        if (
            not isinstance(documents, dict) or not documents
            or set(documents) != {row["hadeethenc_id"] for row in meta["items"]}
            or any(not hid.isascii() or not hid.isdigit() for hid in documents)
            or any(not isinstance(text, str) or not text.strip() for text in documents.values())
        ):
            raise ValueError("Incomplete search documents")
        digest = hashlib.sha256(json.dumps(documents, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if digest != meta["search_documents_sha256"]:
            raise ValueError("Search documents checksum mismatch")
        return documents
    except Exception as exc:
        raise HadithIndexError("Textes de recherche Hadith absents, incompatibles ou invalides.") from exc


class HadithIndex:
    def __init__(self, config: HadithConfig | None = None):
        self.config = config or HadithConfig.from_env()
        self._resources = None
        self._documents = None
        self._documents_lock = Lock()
        self._load_lock = Lock()
        self._encode_lock = Lock()

    def source_documents(self) -> dict[str, str]:
        if self._documents is None:
            with self._documents_lock:
                if self._documents is None:
                    self._documents = load_search_documents(self.config)
        return self._documents

    def load(self):
        if self._resources is None:
            with self._load_lock:
                if self._resources is None:
                    try:
                        matrix, ids, meta = load_index(self.config)
                        model = load_embedding_model(self.config.model_name, meta.get("model_revision"))
                        if model.get_sentence_embedding_dimension() != matrix.shape[1]:
                            raise ValueError("Model/index dimension mismatch")
                        self._resources = (matrix, ids, model)
                    except Exception as exc:
                        raise HadithIndexError("Impossible de charger le moteur Hadith.") from exc
        return self._resources

    def rank(self, query: str, limit: int = 3, *, candidate_ids: set[str] | None = None) -> list[tuple[str, float]]:
        if not 1 <= limit <= 5:
            raise ValueError("limit must be between 1 and 5")
        if candidate_ids == set():
            return []
        import numpy as np

        matrix, ids, model = self.load()
        try:
            with self._encode_lock:
                vector = np.asarray(model.encode([f"query: {query}"], normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)[0]
            norm = np.linalg.norm(vector)
            if vector.shape != (matrix.shape[1],) or not np.isfinite(vector).all() or norm <= 0:
                raise ValueError("Invalid query embedding")
            scores = matrix @ (vector / norm)
            return _aggregate_scores(ids, scores, limit, candidate_ids=candidate_ids)
        except Exception as exc:
            raise HadithIndexError("Impossible d'encoder la recherche Hadith.") from exc


def _aggregate_scores(ids: list[str], scores, limit: int, *, candidate_ids: set[str] | None = None) -> list[tuple[str, float]]:
    """Return top-k (hadith_id, score) pairs using max-per-ID aggregation.

    When an index stores several passages per hadith (multi / multi_context
    strategies), a single hadith may appear multiple times in *ids*. Taking
    the maximum score across all its passages before ranking ensures that
    each hadith is returned at most once and is ranked by its best passage.

    For single-embedding indexes (ids are unique) the result is identical to
    a plain argsort, so this function is strategy-agnostic.
    """
    best: dict[str, float] = {}
    for hadith_id, score in zip(ids, scores.tolist()):
        if candidate_ids is not None and hadith_id not in candidate_ids:
            continue
        if hadith_id not in best or score > best[hadith_id]:
            best[hadith_id] = score

    ranked = sorted(best.items(), key=lambda x: x[1], reverse=True)
    return ranked[:limit]
