"""Official HadeethEnc API only. Source strings are returned unchanged."""

import json
import re
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from app.core.hadith_config import HadithConfig
from app.schemas.hadith import HadithResult


class HadeethEncError(Exception):
    pass


class HadeethEncNotFound(HadeethEncError):
    pass


class HadeethEncClient:
    def __init__(self, base_url: str | None = None, timeout: float = 10):
        self.base_url = (base_url or HadithConfig.from_env().base_url).rstrip("/")
        self.timeout = timeout

    def _get(self, endpoint: str, **params) -> Any:
        url = f"{self.base_url}/{endpoint}/?{urlencode(params)}"
        request = Request(url, headers={"Accept": "application/json", "User-Agent": "Sawt-AI-Hadith/0.1"})
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            if exc.code == 404:
                raise HadeethEncNotFound("Fiche HadeethEnc introuvable.") from exc
            raise HadeethEncError(f"HadeethEnc HTTP {exc.code}.") from exc
        except (URLError, OSError, TimeoutError) as exc:
            raise HadeethEncError("HadeethEnc indisponible ou délai dépassé.") from exc
        except (ValueError, UnicodeError) as exc:
            raise HadeethEncError("Réponse HadeethEnc invalide.") from exc

    def get_hadith(self, hadith_id: str, language: str = "fr") -> dict:
        hadith_id = str(hadith_id)
        if not re.fullmatch(r"[0-9]+", hadith_id):
            raise HadeethEncError("Identifiant HadeethEnc invalide.")
        payload = self._get("hadeeths/one", id=hadith_id, language=language)
        if not isinstance(payload, dict) or str(payload.get("id")) != hadith_id:
            raise HadeethEncError("Réponse HadeethEnc invalide.")
        for field in ("title", "hadeeth", "hadeeth_ar"):
            if not isinstance(payload.get(field), str) or not payload[field].strip():
                raise HadeethEncError(f"Champ HadeethEnc invalide : {field}.")
        for field in ("explanation", "grade", "attribution"):
            if payload.get(field) is not None and not isinstance(payload[field], str):
                raise HadeethEncError(f"Champ HadeethEnc invalide : {field}.")
        translations = payload.get("translations")
        if not isinstance(translations, list) or language not in translations:
            raise HadeethEncNotFound("Traduction HadeethEnc introuvable.")
        return payload

    def list_categories(self, language: str = "fr", *, roots: bool = False) -> list[dict]:
        payload = self._get("categories/roots" if roots else "categories/list", language=language)
        if not isinstance(payload, list) or not payload:
            raise HadeethEncError("Catégories HadeethEnc invalides.")
        for category in payload:
            if not isinstance(category, dict) or not str(category.get("id", "")).isdigit() or not isinstance(category.get("title"), str):
                raise HadeethEncError("Catégorie HadeethEnc invalide.")
        return payload

    def iter_hadith_ids(self, language: str = "fr"):
        seen: set[str] = set()
        for category in self.list_categories(language, roots=True):
            page = 1
            while True:
                payload = self._get("hadeeths/list", language=language, category_id=category["id"], page=page, per_page=100)
                try:
                    entries = payload["data"]
                    current_page = int(payload["meta"]["current_page"])
                    last_page = int(payload["meta"]["last_page"])
                    if not isinstance(entries, list) or current_page != page or last_page < page:
                        raise ValueError
                    for entry in entries:
                        hadith_id = str(entry["id"])
                        if not hadith_id.isascii() or not hadith_id.isdigit():
                            raise ValueError
                        if hadith_id not in seen:
                            seen.add(hadith_id)
                            yield hadith_id
                except (KeyError, TypeError, ValueError) as exc:
                    raise HadeethEncError("Pagination HadeethEnc invalide.") from exc
                if page == last_page:
                    break
                page += 1


def to_hadith_result(payload: dict, language: str = "fr") -> HadithResult:
    return HadithResult(
        id=str(payload["id"]), title=payload["title"],
        arabic=payload["hadeeth_ar"], translation=payload["hadeeth"],
        explanation=payload.get("explanation"), grade=payload.get("grade"),
        attribution=payload.get("attribution"),
        source_url=f"https://hadeethenc.com/{language}/browse/hadith/{payload['id']}",
    )
