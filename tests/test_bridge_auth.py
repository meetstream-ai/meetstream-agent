"""
Who may open the bridge sockets and the debug UI (MS-2135).

Observed before this fix: /bridge, /bridge/audio and /ws/{session_id} took
any client. The README exposes the server with ngrok, so anyone who found
the URL could send ``ready`` with a bot's id and then speak through the
agent into that meeting (``usermsg``), or attach the debug UI to it.

Run: uv run --with pytest pytest
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.routes import websockets as ws_routes
from app.server import app

SECRET = "s" * 40


class FakeBridge:
    """Records what reached the agent; no OpenAI session."""

    def __init__(self) -> None:
        self.texts: list[tuple[str, str]] = []
        self.controls: list[str] = []
        self.ui: list[str] = []

    async def bind_control(self, bot_id, ws):
        self.controls.append(bot_id)

    async def unbind_control(self, bot_id):
        pass

    async def bind_audio_channel(self, bot_id, ws):
        pass

    async def unbind_audio_channel(self, bot_id):
        pass

    async def ingest_user_text(self, bot_id, text):
        self.texts.append((bot_id, text))

    async def interrupt_model(self, bot_id):
        pass

    async def ensure_session(self, bot_id):
        pass

    async def attach_ui(self, session_id, ws, bot_id=None):
        self.ui.append(bot_id)

    async def detach_ui(self, session_id):
        pass


@pytest.fixture
def fake_bridge(monkeypatch):
    fake = FakeBridge()
    monkeypatch.setattr(ws_routes, "bridge", fake)
    monkeypatch.delenv("DEBUG_UI_ENABLED", raising=False)
    monkeypatch.setenv("BRIDGE_SECRET", SECRET)
    return fake


def _token(**kwargs) -> str:
    from app.meetstream.session_token import mint

    return mint(SECRET, **kwargs)


def _refused(client: TestClient, path: str) -> bool:
    try:
        with client.websocket_connect(path) as ws:
            ws.send_json({"type": "ready", "bot_id": "victim-bot"})
            ws.send_json({"command": "usermsg", "message": "say: transfer the money"})
            ws.receive_json()
    except WebSocketDisconnect:
        return True
    return False


@pytest.mark.parametrize("path", ["/bridge", "/victim-bot/bridge", "/bridge/audio", "/victim-bot/bridge/audio"])
def test_bridge_sockets_without_a_token_are_refused(fake_bridge, path):
    client = TestClient(app)
    assert _refused(client, path)
    assert fake_bridge.texts == [] and fake_bridge.controls == []


def test_debug_ui_is_off_unless_enabled(fake_bridge, monkeypatch):
    client = TestClient(app)
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws/attacker?bot_id=victim-bot") as ws:
            ws.receive_json()
    assert fake_bridge.ui == []
    assert client.get("/").headers["content-type"].startswith("application/json")

    monkeypatch.setenv("DEBUG_UI_ENABLED", "1")
    with client.websocket_connect("/ws/local?bot_id=demo") as ws:
        assert ws.receive_json()["type"] == "ack"
    assert fake_bridge.ui == ["demo"]


def test_a_valid_session_token_opens_the_bridge_for_its_bot_only(fake_bridge):
    client = TestClient(app)
    token = _token()
    with client.websocket_connect(f"/bridge/{token}") as ws:
        ws.send_json({"type": "ready", "bot_id": "bot-1"})
        assert ws.receive_json()["command"] == "ack"
        ws.send_json({"command": "usermsg", "message": "hello"})
    with client.websocket_connect(f"/bot-1/bridge/audio/{token}") as audio:
        audio.send_json({"type": "ready", "bot_id": "bot-1"})
        assert audio.receive_json()
    assert fake_bridge.controls == ["bot-1"]

    # The same token cannot be reused for another bot, nor a bot that the
    # path does not name.
    assert _refused(client, f"/bridge/{token}")
    assert _refused(client, f"/other-bot/bridge/{_token()}")
    assert fake_bridge.controls == ["bot-1"]


def test_forged_expired_or_unsigned_tokens_are_refused(fake_bridge, monkeypatch):
    client = TestClient(app)
    good = _token()
    expires, nonce, sig = good.split(".")
    for token in [
        f"{int(expires) + 999}.{nonce}.{sig}",  # extended expiry, old signature
        _token(ttl_seconds=-1),  # expired
        "garbage",
    ]:
        assert _refused(client, f"/bridge/{token}")
    monkeypatch.delenv("BRIDGE_SECRET")  # fail closed without a secret
    assert _refused(client, f"/bridge/{good}")
    monkeypatch.setenv("BRIDGE_SECRET", "short")
    assert _refused(client, f"/bridge/{good}")
    assert fake_bridge.controls == []
