"""Binary frame decoding, PCM resampling, and speaker filtering."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .config import AudioConfig, load_audio_config

try:
    from scipy.signal import resample_poly

    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


#: Byte 0 of a binary frame on the audio socket.
FRAME_PCM = 0x01
#: Sent by MeetStream live (seen November 2026) but not yet in its docs: like
#: ``0x01``, with the PCM format after the type byte (sample rate uint32 LE,
#: channels uint8). Observed: 24000 Hz, mono, 40 ms per frame.
FRAME_PCM_WITH_FORMAT = 0x04


def decode_audio_frame_with_rate(data: bytes) -> Optional[Tuple[str, str, bytes, Optional[int]]]:
    """
    Decode a MeetStream binary audio frame → (speaker_id, speaker_name, pcm_bytes, sample_rate).

    ``sample_rate`` is None for ``0x01`` frames (their rate is ``MEETSTREAM_IN_RATE``).
    Unknown types, multi-channel audio and truncated frames give None, so the
    caller skips them instead of failing the socket.
    """
    if len(data) < 5:
        return None
    sample_rate: Optional[int] = None
    if data[0] == FRAME_PCM:
        off = 1
    elif data[0] == FRAME_PCM_WITH_FORMAT:
        if len(data) < 10:
            return None
        sample_rate = int.from_bytes(data[1:5], "little")
        if data[5] != 1 or sample_rate <= 0:
            return None
        off = 6
    else:
        return None
    sid_len = int.from_bytes(data[off : off + 2], "little")
    off += 2
    if off + sid_len + 2 > len(data):
        return None
    speaker_id = data[off : off + sid_len].decode("utf-8", errors="replace")
    off += sid_len
    sname_len = int.from_bytes(data[off : off + 2], "little")
    off += 2
    if off + sname_len > len(data):
        return None
    speaker_name = data[off : off + sname_len].decode("utf-8", errors="replace")
    off += sname_len
    pcm = data[off:]
    return speaker_id, speaker_name, pcm[: len(pcm) - len(pcm) % 2], sample_rate


def decode_audio_frame(data: bytes) -> Optional[Tuple[str, str, bytes]]:
    """Decode a ``0x01`` frame → (speaker_id, speaker_name, pcm_bytes). See ``decode_audio_frame_with_rate``."""
    result = decode_audio_frame_with_rate(data)
    if result is None or result[3] is not None:
        return None
    return result[:3]


def resample_pcm16(pcm_bytes: bytes, src_hz: int, dst_hz: int) -> bytes:
    if src_hz == dst_hz:
        return pcm_bytes
    x = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)

    if _HAS_SCIPY:
        from math import gcd

        g = gcd(src_hz, dst_hz)
        up, down = dst_hz // g, src_hz // g
        y = resample_poly(x, up=up, down=down)
    else:
        n_out = int(len(x) * (dst_hz / src_hz))
        t_src = np.linspace(0.0, 1.0, num=len(x), endpoint=False)
        t_dst = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
        y = np.interp(t_dst, t_src, x)

    y = np.clip(y, -32768, 32767).astype(np.int16)
    return y.tobytes()


_cfg: Optional[AudioConfig] = None


def _get_cfg() -> AudioConfig:
    global _cfg
    if _cfg is None:
        _cfg = load_audio_config()
    return _cfg


def should_ignore_speaker(speaker_name: str, cfg: Optional[AudioConfig] = None) -> bool:
    """Drop audio from the bot / assistant to avoid echo (uses env-driven lists)."""
    c = cfg or _get_cfg()
    if speaker_name in c.ignored_speaker_names:
        return True
    lower = speaker_name.lower()
    return any(kw in lower for kw in c.agent_speaker_keywords)
