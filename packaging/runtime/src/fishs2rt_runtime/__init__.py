"""Location helper for the pinned FishS2RT native runtime."""

from __future__ import annotations

from pathlib import Path

__version__ = "0.1.0"


def runtime_path() -> Path:
    path = Path(__file__).resolve().parent / "_native" / "audiocpp_cli.exe"
    if not path.is_file():
        raise FileNotFoundError(f"FishS2RT native worker is missing: {path}")
    return path


__all__ = ["runtime_path"]
