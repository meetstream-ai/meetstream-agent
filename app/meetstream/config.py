"""Environment-driven audio and speaker-filter settings for MeetStream."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import FrozenSet


def _parse_ignore_names() -> FrozenSet[str]:
    names = {"Meetstream Agent", "MeetStream Agent", "Meeting Assistant"}
    bot = os.getenv("MEETSTREAM_BOT_NAME", "").strip()
    if bot:
        names.add(bot)
    for part in os.getenv("MEETSTREAM_IGNORE_SPEAKERS", "").split(","):
        p = part.strip()
        if p:
            names.add(p)
    return frozenset(names)


def _parse_agent_keywords() -> tuple[str, ...]:
    raw = os.getenv("AGENT_SPEAKER_KEYWORDS", "bot,agent,assistant,ai")
    return tuple(k.strip().lower() for k in raw.split(",") if k.strip())


def _meeting_out_chunk_ms() -> int:
    """How much model PCM to batch before one ``sendaudio`` (smoother MeetStream playback)."""
    try:
        ms = int(os.getenv("MEETSTREAM_OUT_AUDIO_CHUNK_MS", "240"))
    except ValueError:
        ms = 240
    return max(40, min(2000, ms))


def _sendaudio_pace_seconds() -> float:
    """Optional pause between consecutive ``sendaudio`` JSON frames (reduces client choke)."""
    try:
        ms = float(os.getenv("MEETSTREAM_SENDAUDIO_PACE_MS", "0"))
    except ValueError:
        ms = 0.0
    return max(0.0, min(500.0, ms)) / 1000.0


@dataclass(frozen=True)
class AudioConfig:
    """Sample rates and speaker filtering for the bridge."""

    incoming_hz: int = int(os.getenv("MEETSTREAM_IN_RATE", "48000"))
    outgoing_hz: int = int(os.getenv("MEETSTREAM_OUT_RATE", "48000"))
    model_hz: int = int(os.getenv("MODEL_AUDIO_RATE", "24000"))
    meeting_out_chunk_ms: int = field(default_factory=_meeting_out_chunk_ms)
    sendaudio_pace_seconds: float = field(default_factory=_sendaudio_pace_seconds)
    ignored_speaker_names: FrozenSet[str] = field(default_factory=_parse_ignore_names)
    agent_speaker_keywords: tuple[str, ...] = field(default_factory=_parse_agent_keywords)


def load_audio_config() -> AudioConfig:
    return AudioConfig()
