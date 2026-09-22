"""Typed options for Fish S2 inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .errors import InvalidRequestError


@dataclass(frozen=True, slots=True)
class Sampling:
    """Fish sampling and chunking settings.

    ``numbers_safe()`` is the lower-entropy preset tested during the local
    numeric-speech investigation. It can reduce malformed numeric speech, but
    is not guaranteed to improve every voice or language.
    """

    seed: int | None = None
    temperature: float = 0.8
    top_p: float = 0.8
    top_k: int = 30
    max_tokens: int = 1024
    text_chunk_size: int = 200
    text_chunk_mode: str = "tag_aware"
    stream_chunk_frames: int = 8
    request_options: Mapping[str, str | int | float | bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.seed is not None and not 0 <= self.seed <= 0xFFFFFFFF:
            raise InvalidRequestError("seed must be in [0, 2^32-1]")
        if not 0.0 < self.temperature < 2.0:
            raise InvalidRequestError("temperature must be in (0, 2)")
        if not 0.0 < self.top_p <= 1.0:
            raise InvalidRequestError("top_p must be in (0, 1]")
        if self.top_k <= 0:
            raise InvalidRequestError("top_k must be positive")
        if self.max_tokens <= 0:
            raise InvalidRequestError("max_tokens must be positive")
        if self.text_chunk_size <= 0:
            raise InvalidRequestError("text_chunk_size must be positive")
        if self.stream_chunk_frames < 1 or self.stream_chunk_frames > 256:
            raise InvalidRequestError("stream_chunk_frames must be in [1, 256]")
        if self.text_chunk_mode not in {"default", "tag_aware", "japanese", "endline"}:
            raise InvalidRequestError(
                "text_chunk_mode must be default, tag_aware, japanese, or endline"
            )

    @classmethod
    def numbers_safe(cls, *, seed: int | None = None, **overrides: object) -> "Sampling":
        values: dict[str, object] = {
            "seed": seed,
            "temperature": 0.65,
            "top_p": 0.78,
            "top_k": 20,
        }
        values.update(overrides)
        return cls(**values)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Native worker settings fixed for the lifetime of one loaded model."""

    backend: str = "cuda"
    device: int = 0
    threads: int | None = None
    reference_cache_slots: int = 1
    mem_saver: bool = False
    bf16_storage: str = "raw"
    session_options: Mapping[str, str | int | float | bool] = field(default_factory=dict)
    load_timeout: float = 300.0
    command_timeout: float | None = None

    def __post_init__(self) -> None:
        if self.device < 0:
            raise InvalidRequestError("device must be non-negative")
        if self.threads is not None and self.threads <= 0:
            raise InvalidRequestError("threads must be positive")
        if self.reference_cache_slots < 0:
            raise InvalidRequestError("reference_cache_slots must be non-negative")
        if self.bf16_storage not in {"raw", "p2", "p3"}:
            raise InvalidRequestError("bf16_storage must be raw, p2, or p3")
        if self.load_timeout <= 0:
            raise InvalidRequestError("load_timeout must be positive")
        if self.command_timeout is not None and self.command_timeout <= 0:
            raise InvalidRequestError("command_timeout must be positive")
