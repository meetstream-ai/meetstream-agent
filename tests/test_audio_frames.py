"""
Binary audio frames from MeetStream.

Seen live (November 2026): MeetStream sends type ``0x04`` frames (24 kHz mono,
with the rate and channel count after the type byte), which the decoder did not
know, so every frame was dropped and the agent heard nothing.

Run: uv run --with pytest python -m pytest tests
"""

from __future__ import annotations

import struct

import pytest
from fastapi.testclient import TestClient

from app.meetstream.audio import decode_audio_frame, decode_audio_frame_with_rate
from app.realtime.pipeline import RealtimeMeetingBridge
from app.routes import websockets as ws_routes
from app.server import app

SPEAKER_ID, SPEAKER_NAME = "spk-1234", "Test Speaker"


def _ids() -> bytes:
    sid, name = SPEAKER_ID.encode(), SPEAKER_NAME.encode()
    return struct.pack("<H", len(sid)) + sid + struct.pack("<H", len(name)) + name


def frame_v1(pcm: bytes) -> bytes:
    return b"\x01" + _ids() + pcm


def frame_v4(pcm: bytes, rate: int = 24000, channels: int = 1) -> bytes:
    return b"\x04" + struct.pack("<IB", rate, channels) + _ids() + pcm


PCM_40MS_24K = b"\x01\x00" * 960


def test_type_1_frames_decode_as_before():
    assert decode_audio_frame(frame_v1(b"\x02\x00" * 4)) == (SPEAKER_ID, SPEAKER_NAME, b"\x02\x00" * 4)
    assert decode_audio_frame_with_rate(frame_v1(b"\x02\x00"))[3] is None


def test_type_4_frames_decode_with_their_sample_rate():
    sid, name, pcm, rate = decode_audio_frame_with_rate(frame_v4(PCM_40MS_24K))
    assert (sid, name, pcm, rate) == (SPEAKER_ID, SPEAKER_NAME, PCM_40MS_24K, 24000)
    # The old 3-tuple API only returns frames at MEETSTREAM_IN_RATE.
    assert decode_audio_frame(frame_v4(PCM_40MS_24K)) is None


@pytest.mark.parametrize(
    "data",
    [
        frame_v4(PCM_40MS_24K, channels=2),  # multi-channel: not handled
        frame_v4(PCM_40MS_24K, rate=0),
        b"\x07" + _ids() + PCM_40MS_24K,  # unknown type: skipped
        frame_v1(b"")[:6],  # truncated inside the speaker id
        b"\x04\xc0\x5d\x00",  # too short
    ],
)
def test_frames_that_cannot_be_used_are_skipped(data):
    assert decode_audio_frame_with_rate(data) is None


def test_odd_byte_counts_are_trimmed_to_whole_samples():
    assert decode_audio_frame_with_rate(frame_v1(b"\x01\x00\x02"))[2] == b"\x01\x00"


class FakeSession:
    def __init__(self) -> None:
        self.audio: list[bytes] = []

    async def send_audio(self, pcm: bytes) -> None:
        self.audio.append(pcm)


class FakeBridge(RealtimeMeetingBridge):
    """The real audio path, with a recording session instead of OpenAI."""

    def __init__(self) -> None:
        super().__init__()
        self.session = FakeSession()

    async def ensure_session(self, bot_id):
        self.sessions[bot_id] = self.session

    async def close_session(self, bot_id):
        self.sessions.pop(bot_id, None)


def _model_bytes_for(path: str, frames: list[bytes], monkeypatch) -> list[int]:
    fake = FakeBridge()
    monkeypatch.setattr(ws_routes, "bridge", fake)
    client = TestClient(app)
    with client.websocket_connect(path) as ws:
        ws.send_json({"type": "ready", "bot_id": "bot-1"})
        ws.receive_json()
        for f in frames:
            ws.send_bytes(f)
        ws.send_json({"type": "ready", "bot_id": "bot-1"})  # round trip: frames handled
        ws.receive_json()
    return [len(a) for a in fake.session.audio]


def test_type_4_audio_reaches_the_model_at_the_right_rate(monkeypatch):
    """40 ms at 24 kHz is 1920 bytes at the model's 24 kHz: no rate change."""
    model_rate = RealtimeMeetingBridge()._audio.model_hz
    expected = int(960 * model_rate / 24000) * 2
    assert _model_bytes_for("/bridge/audio", [frame_v4(PCM_40MS_24K)] * 3, monkeypatch) == [expected] * 3


def test_type_1_audio_still_uses_the_configured_rate(monkeypatch):
    bridge = RealtimeMeetingBridge()
    in_hz, model_hz = bridge._audio.incoming_hz, bridge._audio.model_hz
    pcm = b"\x01\x00" * (in_hz // 25)  # 40 ms at MEETSTREAM_IN_RATE
    assert _model_bytes_for("/bridge/audio", [frame_v1(pcm)], monkeypatch) == [model_hz // 25 * 2]
