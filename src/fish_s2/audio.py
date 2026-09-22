"""Dependency-free PCM16/WAV value objects."""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from .errors import ProtocolError


@dataclass(frozen=True, slots=True)
class Audio:
    """Interleaved signed 16-bit little-endian PCM audio."""

    pcm: bytes
    sample_rate: int
    channels: int
    metrics: Mapping[str, float | str] = field(default_factory=dict)
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.sample_rate <= 0 or self.channels <= 0:
            raise ValueError("sample_rate and channels must be positive")
        if len(self.pcm) % (2 * self.channels):
            raise ValueError("PCM16 byte count is not aligned to complete frames")

    @property
    def frames(self) -> int:
        return len(self.pcm) // (2 * self.channels)

    @property
    def duration(self) -> float:
        return self.frames / self.sample_rate

    def to_wav_bytes(self) -> bytes:
        output = io.BytesIO()
        with wave.open(output, "wb") as wav:
            wav.setnchannels(self.channels)
            wav.setsampwidth(2)
            wav.setframerate(self.sample_rate)
            wav.writeframes(self.pcm)
        return output.getvalue()

    def save(self, path: str | Path) -> Path:
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.to_wav_bytes())
        return target


@dataclass(frozen=True, slots=True)
class AudioChunk:
    """One immediately playable live PCM chunk."""

    pcm: bytes
    sample_rate: int
    channels: int
    sequence: int
    segment: int
    request_id: str

    @property
    def frames(self) -> int:
        return len(self.pcm) // (2 * self.channels)

    @property
    def duration(self) -> float:
        return self.frames / self.sample_rate

    def as_audio(self) -> Audio:
        return Audio(self.pcm, self.sample_rate, self.channels)


def read_pcm16_wav(path: str | Path) -> Audio:
    source = Path(path)
    try:
        with wave.open(str(source), "rb") as wav:
            channels = wav.getnchannels()
            sample_rate = wav.getframerate()
            sample_width = wav.getsampwidth()
            compression = wav.getcomptype()
            pcm = wav.readframes(wav.getnframes())
    except (OSError, EOFError, wave.Error) as exc:
        raise ProtocolError(f"cannot read native WAV output {source}: {exc}") from exc
    if sample_width != 2 or compression != "NONE":
        raise ProtocolError(
            f"native WAV output must be uncompressed PCM16, got width={sample_width}, compression={compression}"
        )
    return Audio(pcm, sample_rate, channels)


def concatenate_audio(parts: list[Audio], *, pause_ms: int = 0) -> Audio:
    if not parts:
        raise ProtocolError("native request completed without final audio")
    rate = parts[0].sample_rate
    channels = parts[0].channels
    if any(part.sample_rate != rate or part.channels != channels for part in parts):
        raise ProtocolError("native outputs use incompatible audio formats")
    pause_frames = round(rate * pause_ms / 1000)
    silence = b"\0" * (pause_frames * channels * 2)
    pcm = silence.join(part.pcm for part in parts)
    return Audio(pcm, rate, channels)
