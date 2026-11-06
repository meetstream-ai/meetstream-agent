"""
Per-session tokens for the MeetStream bridge sockets.

The bridge is reachable from the internet (a tunnel or a public host), and
whoever opens ``/bridge`` can speak through the agent into the meeting and
spend the OpenAI Realtime credit. So each bot gets its own token in the
socket URLs it is created with::

    wss://<host>/bridge/<token>          control
    wss://<host>/bridge/audio/<token>    audio

Token format (easy to make in any language)::

    <expires_unix>.<nonce>.<hex HMAC-SHA256(BRIDGE_SECRET, "<expires_unix>.<nonce>")>

The bridge checks the signature and the expiry before it accepts the socket,
and binds the token to the ``bot_id`` of the first ``ready`` message: the
other channel, and any reconnect, must name the same bot.

Make one (prints the token and both socket paths)::

    uv run python -m app.meetstream.session_token [--ttl-hours 12]
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import logging
import os
import secrets
import threading
import time

logger = logging.getLogger("bridge.meetstream.auth")

#: A shorter secret is refused: tokens signed with it are easy to forge.
MIN_SECRET_LENGTH = 32
DEFAULT_TTL_SECONDS = 12 * 60 * 60


def bridge_secret() -> str | None:
    """``BRIDGE_SECRET`` from the environment, or None when unset or too short."""
    value = (os.getenv("BRIDGE_SECRET") or "").strip()
    return value if len(value) >= MIN_SECRET_LENGTH else None


def _signature(secret: str, payload: str) -> str:
    return hmac.new(secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()


def mint(secret: str, ttl_seconds: int = DEFAULT_TTL_SECONDS, *, now: float | None = None) -> str:
    """A new token that is valid for ``ttl_seconds``."""
    expires = int((time.time() if now is None else now) + ttl_seconds)
    payload = f"{expires}.{secrets.token_hex(16)}"
    return f"{payload}.{_signature(secret, payload)}"


def verify(token: str, secret: str, *, now: float | None = None) -> bool:
    """True when the signature matches and the token has not expired."""
    try:
        expires, nonce, signature = token.split(".")
        if not hmac.compare_digest(signature.encode("ascii"), _signature(secret, f"{expires}.{nonce}").encode("ascii")):
            return False
        return int(expires) > (time.time() if now is None else now)
    except (ValueError, UnicodeEncodeError):
        return False


class TokenBindings:
    """Which bot each token belongs to: the first ``ready`` decides."""

    def __init__(self) -> None:
        self._bots: dict[str, tuple[str, int]] = {}
        self._lock = threading.Lock()

    def bind(self, token: str, bot_id: str) -> bool:
        now = time.time()
        expires = int(token.split(".", 1)[0])
        with self._lock:
            for stale in [t for t, (_, exp) in self._bots.items() if exp <= now]:
                del self._bots[stale]
            bound = self._bots.setdefault(token, (bot_id, expires))
        return hmac.compare_digest(bound[0].encode("utf-8"), bot_id.encode("utf-8"))


bindings = TokenBindings()


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(description="Make a session token for one MeetStream bot.")
    parser.add_argument("--ttl-hours", type=float, default=DEFAULT_TTL_SECONDS / 3600,
                        help="how long the token can be used to connect (default 12)")
    args = parser.parse_args()
    secret = bridge_secret()
    if secret is None:
        raise SystemExit(f"Set BRIDGE_SECRET (at least {MIN_SECRET_LENGTH} characters) in .env first.")
    token = mint(secret, int(args.ttl_hours * 3600))
    print(token)
    print(f"control: wss://<your host>/bridge/{token}")
    print(f"audio:   wss://<your host>/bridge/audio/{token}")


if __name__ == "__main__":
    main()
