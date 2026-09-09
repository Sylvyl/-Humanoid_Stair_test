from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import secrets
import time


@dataclass(slots=True)
class Session:
    csrf: str
    expires_at: float


class SessionManager:
    def __init__(self, operator_token: str, ttl_s: int = 900):
        if len(operator_token) < 20:
            raise ValueError("operator token must contain at least 20 characters")
        self._token_digest = hashlib.sha256(operator_token.encode()).digest()
        self.ttl_s = ttl_s
        self.sessions: dict[str, Session] = {}

    def login(self, candidate: str) -> tuple[str, str] | None:
        digest = hashlib.sha256(candidate.encode()).digest()
        if not hmac.compare_digest(digest, self._token_digest):
            return None
        session_id = secrets.token_urlsafe(32)
        csrf = secrets.token_urlsafe(24)
        self.sessions[session_id] = Session(csrf=csrf, expires_at=time.time() + self.ttl_s)
        return session_id, csrf

    def validate(self, session_id: str | None, csrf: str | None = None) -> bool:
        if not session_id:
            return False
        session = self.sessions.get(session_id)
        if session is None or session.expires_at <= time.time():
            self.sessions.pop(session_id, None)
            return False
        if csrf is not None and not hmac.compare_digest(session.csrf, csrf):
            return False
        session.expires_at = time.time() + self.ttl_s
        return True


def new_operator_token() -> str:
    return secrets.token_urlsafe(32)

