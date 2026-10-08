"""Backend-only Content Sync access for the manual tafsir pilot import."""

import base64
import json
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from app.services.tafsir_sources import TafsirSourceError

ENVIRONMENTS = {
    "prelive": ("https://apis-prelive.quran.foundation/content", "https://prelive-oauth2.quran.foundation"),
    "production": ("https://apis.quran.foundation/content", "https://oauth2.quran.foundation"),
}
TIMEOUT_SECONDS = 90
MAX_RESPONSE_BYTES = 128 * 1024 * 1024
USER_AGENT = "Sawt-AI/1.0 (+https://sawt-ai.nabster.dev; Quran Foundation Content Sync importer)"


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Neither the client secret nor access token may follow another URL.
        return None


class QuranFoundationClient:
    def __init__(self):
        self.environment = os.getenv("QF_ENV", "").strip() or "prelive"
        if self.environment not in ENVIRONMENTS:
            raise TafsirSourceError("QF_ENV doit être prelive ou production.")
        self.api_origin, self.auth_origin = ENVIRONMENTS[self.environment]
        self.client_id = os.getenv("QF_CLIENT_ID", "").strip()
        self.client_secret = os.getenv("QF_CLIENT_SECRET", "").strip()
        if not self.client_id or not self.client_secret or any(
            character.isspace() for character in self.client_id + self.client_secret
        ) or ":" in self.client_id:
            raise TafsirSourceError("Configurer QF_CLIENT_ID et QF_CLIENT_SECRET dans le backend (Developer Console Quran Foundation).")
        self.opener = build_opener(_NoRedirect())
        self.access_token = None

    def _json(self, request):
        # Cloudflare rejects urllib's generic Python client signature (HTTP 403/1010).
        request.add_header("User-Agent", USER_AGENT)
        try:
            with self.opener.open(request, timeout=TIMEOUT_SECONDS) as response:
                data = response.read(MAX_RESPONSE_BYTES + 1)
            if len(data) > MAX_RESPONSE_BYTES:
                raise TafsirSourceError("Réponse Quran Foundation trop volumineuse ; import interrompu.")
            payload = json.loads(data)
            if not isinstance(payload, dict):
                raise ValueError()
            return payload
        except HTTPError:
            raise  # The caller handles the status without logging the body or credentials.
        except (URLError, OSError) as exc:
            raise TafsirSourceError("Impossible de joindre Quran Foundation ; relancer l'import plus tard.") from exc
        except (ValueError, UnicodeError) as exc:
            raise TafsirSourceError("Réponse JSON Quran Foundation invalide.") from exc

    def _authenticate(self):
        basic = base64.b64encode(f"{self.client_id}:{self.client_secret}".encode()).decode("ascii")
        request = Request(
            self.auth_origin + "/oauth2/token", method="POST",
            data=urlencode({"grant_type": "client_credentials", "scope": "content"}).encode(),
            headers={"Authorization": f"Basic {basic}", "Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
        )
        try:
            payload = self._json(request)
        except HTTPError as exc:
            raise TafsirSourceError(f"Authentification Quran Foundation refusée (HTTP {exc.code}) ; vérifier les accès et QF_ENV.") from exc
        token = payload.get("access_token")
        if not isinstance(token, str) or not token or any(character.isspace() for character in token):
            raise TafsirSourceError("Jeton Quran Foundation absent ou invalide.")
        self.access_token = token

    def get(self, path: str):
        parsed = urlsplit(path)
        if parsed.scheme or parsed.netloc or parsed.fragment or not (
            parsed.path in ("/api/v4/resources/tafsirs", "/api/v4/resources/sync")
            or parsed.path in ("/api/v4/resources/snapshots/tafsirs/14", "/api/v4/resources/snapshots/tafsirs/91")
        ):
            raise TafsirSourceError("Chemin Content Sync inattendu ; aucun accès envoyé à cette adresse.")
        for attempt in range(2):
            if self.access_token is None:
                self._authenticate()
            request = Request(self.api_origin + path, headers={
                "Accept": "application/json", "x-auth-token": self.access_token,
                "x-client-id": self.client_id,
            })
            try:
                return self._json(request)
            except HTTPError as exc:
                if exc.code == 401 and attempt == 0:
                    self.access_token = None
                    continue
                raise TafsirSourceError(
                    f"Quran Foundation indisponible ou accès refusé (HTTP {exc.code}, "
                    f"{self.environment}, {parsed.path}) ; vérifier les permissions et QF_ENV."
                ) from exc

    def bootstrap_tafsir(self, resource_id: int):
        """Finish bootstrap pagination before fetching the current resource copy."""
        path = "/api/v4/resources/sync?" + urlencode({
            "bootstrap": "true", "resources": f"tafsirs:{resource_id}", "per_page": 100,
        })
        visited, snapshot_path, upper = set(), None, None
        for _ in range(100):
            if path in visited:
                raise TafsirSourceError("Pagination Content Sync en boucle.")
            visited.add(path)
            sync = self.get(path).get("sync")
            if not isinstance(sync, dict) or (
                type(sync.get("sync_until_sequence")) is not int
                or sync["sync_until_sequence"] < 0
                or type(sync.get("has_more")) is not bool
                or not isinstance(sync.get("mutations"), list)
            ):
                raise TafsirSourceError("Page Content Sync invalide.")
            if upper is not None and upper != sync["sync_until_sequence"]:
                raise TafsirSourceError("La borne de pagination Content Sync a changé.")
            upper = sync["sync_until_sequence"]
            for mutation in sync["mutations"]:
                if not isinstance(mutation, dict) or (
                    mutation.get("resource_group") != "tafsirs"
                    or type(mutation.get("resource_id")) is not int
                    or mutation["resource_id"] != resource_id
                    or mutation.get("type") != "RESOURCE_CREATE"
                    or mutation.get("snapshot_url") != f"/api/v4/resources/snapshots/tafsirs/{resource_id}"
                    or snapshot_path is not None
                ):
                    raise TafsirSourceError("Ressource Content Sync inattendue ou en doublon.")
                snapshot_path = mutation["snapshot_url"]
            if sync["has_more"]:
                path = sync.get("next_page_url")
                if not isinstance(path, str) or urlsplit(path).path != "/api/v4/resources/sync":
                    raise TafsirSourceError("Page suivante Content Sync invalide.")
                continue
            token = sync.get("next_sync_token")
            if snapshot_path is None:
                raise TafsirSourceError("Tafsir absent de Content Sync ; vérifier sa disponibilité et les permissions de l'application.")
            if not isinstance(token, str) or not token.strip() or sync.get("next_page_url") is not None:
                raise TafsirSourceError("Checkpoint final Content Sync invalide.")
            return self.get(snapshot_path), {"sync_until_sequence": upper, "next_sync_token": token}
        raise TafsirSourceError("Trop de pages Content Sync ; import interrompu.")
