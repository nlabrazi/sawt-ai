"""Read-only access to the exact E5-base experiment, for manual terminal trials."""

import json
from pathlib import Path

API_DIR = Path(__file__).resolve().parents[1]


class HadithExperiment:
    def __init__(self, *, strategy="multi"):
        if strategy not in ("multi", "multi_context"):
            raise ValueError("Unknown experiment strategy")
        self.strategy = strategy
        self.resources = None

    def load(self):
        if self.resources is not None:
            return self.resources
        import numpy as np
        from transformers import AutoTokenizer
        from app.services.hadith_documents import build_documents
        from app.services.hadith_index import load_embedding_model
        from scripts.diagnose_hadith_retrieval import fingerprint, group_documents

        cache = API_DIR / ".cache" / "hadith_retrieval"
        name = "multilingual-e5-base_" + self.strategy
        report_path = API_DIR / "evaluation" / "hadith_retrieval" / (name + ".json")
        snapshot_path = cache / "snapshot.json"
        matrix_path = cache / (name + ".npz")
        if not all(path.is_file() for path in (report_path, snapshot_path, matrix_path)):
            raise ValueError("Le cache du benchmark E5-base manque. Voir evaluation/HADITH_SEARCH.md pour le reconstruire.")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        if report["strategy"] != self.strategy or fingerprint(snapshot) != report["snapshot_sha256"]:
            raise ValueError("Le corpus local ne correspond pas au benchmark E5-base.")
        tokenizer = AutoTokenizer.from_pretrained(
            report["document_tokenizer"], revision=report["document_tokenizer_revision"], trust_remote_code=False,
        )
        tokenizer.model_max_length = 10**9  # Long documents are explicitly split below.
        documents = [doc for record in snapshot["records"] for doc in build_documents(
            record["payload"], snapshot["categories"], self.strategy, tokenizer, report["truncation"]["max_tokens"],
        )]
        if fingerprint([{"id": doc.hadeethenc_id, "kind": doc.kind, "text": doc.text} for doc in documents]) != report["documents_sha256"]:
            raise ValueError("Les passages locaux diffèrent du benchmark E5-base.")
        key = fingerprint({"model": report["model"], "revision": report["model_revision"],
                           "documents": [{"id": doc.hadeethenc_id, "text": doc.text} for doc in documents]})
        with np.load(matrix_path, allow_pickle=False) as archive:
            if archive["key"].item() != key:
                raise ValueError("La matrice ne correspond pas aux passages du benchmark.")
            matrix = archive["embeddings"]
        model = load_embedding_model(report["model"], report["model_revision"])
        if (matrix.shape != (len(documents), model.get_sentence_embedding_dimension())
                or not np.isfinite(matrix).all()
                or not np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=0.001)):
            raise ValueError("La matrice E5-base est invalide.")
        ids, groups = group_documents(documents)
        self.resources = matrix, ids, groups, model
        return self.resources

    def rank(self, query, limit=3):
        import numpy as np
        from app.services.hadith_documents import QUERY_PREFIX
        from scripts.diagnose_hadith_retrieval import rank_documents

        matrix, ids, groups, model = self.load()
        vector = np.asarray(model.encode([QUERY_PREFIX + query], normalize_embeddings=True,
                                         show_progress_bar=False), dtype=np.float32)[0]
        if not np.isfinite(vector).all() or np.linalg.norm(vector) <= 0:
            raise ValueError("Le modèle n'a pas produit une recherche valide.")
        scores, _, ranking = rank_documents(matrix, groups, vector)
        return [(ids[row], float(scores[row])) for row in ranking[:limit]]
