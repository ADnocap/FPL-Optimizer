"""FPL authentication via the PingOne OIDC refresh-token flow (2025+).

FPL moved login to account.premierleague.com (PingOne) in Aug 2025; the old
password POST is dead.  One-time setup (browser):

1. Log in at https://fantasy.premierleague.com
2. DevTools Console:
   copy(JSON.parse(localStorage.getItem(
       Object.keys(localStorage).find(k=>k.startsWith('oidc.user:')))).refresh_token)
3. Put it in .env as FPL_REFRESH_TOKEN=<token>
   (optional: FPL_CLIENT_ID from the localStorage key name — the part after
   the last ':'; defaults to the fantasy web app client id)

Access tokens (~1-2h) are minted on demand; rotated refresh tokens are
persisted back to .env automatically.  All API calls send
``X-Api-Authorization: Bearer <access_token>`` — the header the web app
itself uses.
"""

from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

TOKEN_URL = "https://account.premierleague.com/as/token"
DEFAULT_CLIENT_ID = "bfcbaf69-aade-4c1b-8f00-c1cb8a193030"  # fantasy web app

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    ),
    "Referer": "https://fantasy.premierleague.com/",
    "Origin": "https://fantasy.premierleague.com",
}


class FPLAuthError(RuntimeError):
    pass


class FPLAuth:
    """Mints and refreshes FPL access tokens from a stored refresh token."""

    def __init__(self, env_path: Path | None = None) -> None:
        self.env_path = env_path or Path(__file__).resolve().parents[3] / ".env"
        self._load_env()
        self.refresh_token = os.environ.get("FPL_REFRESH_TOKEN", "")
        self.client_id = os.environ.get("FPL_CLIENT_ID", DEFAULT_CLIENT_ID)
        self._access_token: str | None = None
        self._expires_at: float = 0.0
        if not self.refresh_token:
            raise FPLAuthError(
                "FPL_REFRESH_TOKEN not set. One-time setup: log in at "
                "fantasy.premierleague.com, run the DevTools snippet in "
                "src/fpl_optimizer/live/auth.py's docstring, and add the token to .env"
            )

    def _load_env(self) -> None:
        if self.env_path.exists():
            for line in self.env_path.read_text(encoding="utf-8").splitlines():
                if "=" in line and not line.strip().startswith("#"):
                    k, _, v = line.partition("=")
                    os.environ.setdefault(k.strip(), v.strip())

    def _persist_refresh_token(self, new_token: str) -> None:
        """Write the rotated refresh token back to .env."""
        self.refresh_token = new_token
        os.environ["FPL_REFRESH_TOKEN"] = new_token
        try:
            text = self.env_path.read_text(encoding="utf-8") if self.env_path.exists() else ""
            if re.search(r"^FPL_REFRESH_TOKEN=", text, flags=re.M):
                text = re.sub(
                    r"^FPL_REFRESH_TOKEN=.*$",
                    f"FPL_REFRESH_TOKEN={new_token}",
                    text,
                    flags=re.M,
                )
            else:
                text = text.rstrip("\n") + f"\nFPL_REFRESH_TOKEN={new_token}\n"
            self.env_path.write_text(text, encoding="utf-8")
        except OSError:
            logger.warning("Could not persist rotated refresh token to %s", self.env_path)

    def access_token(self) -> str:
        """Return a valid access token, refreshing if within 60s of expiry."""
        if self._access_token and time.time() < self._expires_at - 60:
            return self._access_token
        resp = requests.post(
            TOKEN_URL,
            data={
                "grant_type": "refresh_token",
                "client_id": self.client_id,
                "refresh_token": self.refresh_token,
            },
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                **_BROWSER_HEADERS,
            },
            timeout=30,
        )
        if resp.status_code != 200:
            raise FPLAuthError(
                f"Token refresh failed ({resp.status_code}): {resp.text[:300]}. "
                "Re-extract FPL_REFRESH_TOKEN from a fresh browser login."
            )
        data = resp.json()
        self._access_token = data["access_token"]
        self._expires_at = time.time() + int(data.get("expires_in", 3600))
        rotated = data.get("refresh_token")
        if rotated and rotated != self.refresh_token:
            self._persist_refresh_token(rotated)
            logger.info("Refresh token rotated and persisted")
        return self._access_token

    def headers(self) -> dict[str, str]:
        return {
            "X-Api-Authorization": f"Bearer {self.access_token()}",
            "Content-Type": "application/json",
            "X-Requested-With": "XMLHttpRequest",
            **_BROWSER_HEADERS,
        }
