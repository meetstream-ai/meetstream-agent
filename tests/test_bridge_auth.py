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

from app.realtime.pipeline import RealtimeMeetingBridge
from app.routes import websockets as ws_routes
from app.server import app

SECRET = "s" * 40


class FakeBridge(RealtimeMeetingBridge):
    """The real socket registry, with no OpenAI session behind it."""

    def __init__(self) -> None:
        super().__init__()
        self.texts: list[tuple[str, str]] = []
        self.controls: list[str] = []
        self.ui: list[str] = []
        self.closed: list[str] = []

    async def bind_control(self, bot_id, ws):
        bound = await super().bind_control(bot_id, ws)
        if bound:
            self.controls.append(bot_id)
        return bound

    async def ensure_session(self, bot_id):
        pass

    async def close_session(self, bot_id):
        self.closed.append(bot_id)

    async def ingest_user_text(self, bot_id, text):
        self.texts.append((bot_id, text))

    async def interrupt_model(self, bot_id):
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


def _close_code(client: TestClient, path: str, bot_id: str) -> int | None:
    """The close code the server sent, or None if the socket stayed open."""
    try:
        with client.websocket_connect(path) as ws:
            ws.send_json({"type": "ready", "bot_id": bot_id})
            ws.receive_json()
    except WebSocketDisconnect as closed:
        return closed.code
    return None


def test_refusals_use_policy_violation(fake_bridge):
    client = TestClient(app)
    assert _close_code(client, "/bridge/garbage", "bot-1") == 1008
    assert _close_code(client, "/bridge", "bot-1") == 1008
    assert _close_code(client, "/ws/x", "bot-1") == 1008


def test_a_path_bot_mismatch_does_not_bind_the_token(fake_bridge):
    # Refused because the path names another bot: the token must stay free
    # for the bot it was made for.
    client = TestClient(app)
    token = _token()
    assert _close_code(client, f"/bot-1/bridge/{token}", "bot-2") == 1008
    with client.websocket_connect(f"/bot-2/bridge/{token}") as ws:
        ws.send_json({"type": "ready", "bot_id": "bot-2"})
        assert ws.receive_json()["command"] == "ack"
    assert fake_bridge.controls == ["bot-2"]


def test_the_audio_channel_is_held_to_the_bot_bound_on_control(fake_bridge):
    client = TestClient(app)
    token = _token()
    with client.websocket_connect(f"/bridge/{token}") as ws:
        ws.send_json({"type": "ready", "bot_id": "bot-1"})
        ws.receive_json()
    assert _close_code(client, f"/bridge/audio/{token}", "bot-2") == 1008


def test_verify_rejects_tampered_tokens():
    from app.meetstream.session_token import verify

    good = _token()
    expires, nonce, sig = good.split(".")
    assert verify(good, SECRET)
    for bad in [
        f"{expires}.{nonce[:-1]}{'1' if nonce[-1] == '0' else '0'}.{sig}",  # nonce changed
        f"{expires}.{nonce}.{sig}.x",  # extra part
        f"{expires}.{nonce}.{sig[:-1]}é",  # non-ASCII signature
        "",
    ]:
        assert not verify(bad, SECRET)
    assert not verify(good, "t" * 40)  # another secret


def test_tokens_are_redacted_from_uvicorn_logs(caplog):
    """uvicorn logs every WebSocket path; the token in it must not reach the log."""
    import logging

    from app.server import create_app

    create_app()  # installs the filter
    token = _token()
    with caplog.at_level(logging.INFO, logger="uvicorn.error"):
        logging.getLogger("uvicorn.error").info(
            '%s - "WebSocket %s" [accepted]', "1.2.3.4:5", f"/bot-1/bridge/audio/{token}"
        )
    message = caplog.records[-1].getMessage()
    assert token not in message
    assert "/bot-1/bridge/audio/<token>" in message


def test_a_second_socket_cannot_take_over_a_live_session(fake_bridge):
    """Same token, same bot, while the real bot is connected: refused on both channels."""
    client = TestClient(app)
    token = _token()
    with client.websocket_connect(f"/bridge/{token}") as control,             client.websocket_connect(f"/bridge/audio/{token}") as audio:
        control.send_json({"type": "ready", "bot_id": "bot-1"})
        control.receive_json()
        audio.send_json({"type": "ready", "bot_id": "bot-1"})
        audio.receive_json()
        registered = fake_bridge.control_ws["bot-1"]

        assert _close_code(client, f"/bridge/{token}", "bot-1") == 1008
        assert _close_code(client, f"/bot-1/bridge/audio/{token}", "bot-1") == 1008
        # The refused sockets left the real ones in place and the session up.
        assert fake_bridge.control_ws["bot-1"] is registered
        assert "bot-1" in fake_bridge.audio_ws
        assert fake_bridge.closed == []

        control.send_json({"command": "usermsg", "message": "still mine"})
    assert fake_bridge.texts == [("bot-1", "still mine")]
    assert fake_bridge.controls == ["bot-1"]


def test_the_bot_can_reconnect_after_its_socket_closes(fake_bridge):
    client = TestClient(app)
    token = _token()
    for _ in range(2):
        with client.websocket_connect(f"/bridge/{token}") as ws:
            ws.send_json({"type": "ready", "bot_id": "bot-1"})
            assert ws.receive_json()["command"] == "ack"
    assert fake_bridge.controls == ["bot-1", "bot-1"]
    assert fake_bridge.control_ws == {}
