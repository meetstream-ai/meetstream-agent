"""
MeetStream wire protocol: binary audio frames, speaker filtering, outbound commands.

See: https://docs.meetstream.ai/guides/get-started/bridge-server-architecture
"""

from .audio import decode_audio_frame, resample_pcm16, should_ignore_speaker
from .config import AudioConfig, load_audio_config

__all__ = [
    "AudioConfig",
    "decode_audio_frame",
    "load_audio_config",
    "resample_pcm16",
    "should_ignore_speaker",
]
