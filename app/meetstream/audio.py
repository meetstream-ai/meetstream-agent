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


def decode_audio_frame(data: bytes) -> Optional[Tuple[str, str, bytes]]:
    """Decode MeetStream binary audio frame → (speaker_id, speaker_name, pcm_bytes)."""
    if len(data) < 5 or data[0] != 0x01:
        return None
    sid_len = int.from_bytes(data[1:3], "little")
    speaker_id = data[3 : 3 + sid_len].decode("utf-8")
    off = 3 + sid_len
    sname_len = int.from_bytes(data[off : off + 2], "little")
    off += 2
    speaker_name = data[off : off + sname_len].decode("utf-8")
    off += sname_len
    return speaker_id, speaker_name, data[off:]


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
