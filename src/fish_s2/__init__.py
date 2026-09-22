"""Public API for the optimized Fish Audio S2 Pro runtime."""

from .audio import Audio, AudioChunk
from .errors import (
    BusyError,
    FishS2Error,
    GenerationError,
    InvalidRequestError,
    ModelLoadError,
    ProtocolError,
    RuntimeNotFoundError,
    WorkerCrashedError,
)
from .model import AudioStream, FishS2, Voice
from .options import RuntimeConfig, Sampling
from ._runtime import RuntimeInfo

__all__ = [
    "Audio",
    "AudioChunk",
    "AudioStream",
    "BusyError",
    "FishS2",
    "FishS2Error",
    "GenerationError",
    "InvalidRequestError",
    "ModelLoadError",
    "ProtocolError",
    "RuntimeConfig",
    "RuntimeInfo",
    "RuntimeNotFoundError",
    "Sampling",
    "Voice",
    "WorkerCrashedError",
]

__version__ = "0.1.0"
