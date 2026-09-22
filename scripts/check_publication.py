"""Fail when the prospective Git tree contains private or release-unsafe files."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_ROOTS = {"artifacts", "models", "outputs", "research", "reports", "webui"}
FORBIDDEN_SUFFIXES = {
    ".ckpt",
    ".dll",
    ".etl",
    ".exe",
    ".flac",
    ".gguf",
    ".lib",
    ".mp3",
    ".ncu-rep",
    ".nsys-rep",
    ".onnx",
    ".ogg",
    ".pdb",
    ".pt",
    ".pth",
    ".safetensors",
    ".trace",
    ".wav",
}
TEXT_SUFFIXES = {
    "",
    ".c",
    ".cc",
    ".cmake",
    ".cmd",
    ".cpp",
    ".cu",
    ".cuh",
    ".h",
    ".hpp",
    ".in",
    ".inc",
    ".json",
    ".md",
    ".ps1",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
PRIVATE_PATTERNS = {
    "absolute Windows user path": re.compile(r"[A-Za-z]:\\Users\\", re.IGNORECASE),
    "absolute local project path": re.compile(r"[A-Za-z]:\\Projects\\", re.IGNORECASE),
    "private machine account": re.compile(r"\b" + "sur" + "vk" + r"\b", re.IGNORECASE),
}


def prospective_files() -> list[str]:
    completed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return [line for line in completed.stdout.splitlines() if line]


def main() -> int:
    failures: list[str] = []
    files = prospective_files()
    if not files:
        failures.append("prospective Git tree is empty")
    for raw in files:
        relative = PurePosixPath(raw)
        folded = {part.casefold() for part in relative.parts}
        suffix = relative.suffix.casefold()
        if relative.parts and relative.parts[0].casefold() in FORBIDDEN_ROOTS:
            failures.append(f"forbidden path: {raw}")
        if "__pycache__" in folded:
            failures.append(f"Python cache path: {raw}")
        if suffix in FORBIDDEN_SUFFIXES:
            failures.append(f"forbidden binary/media/model file: {raw}")
        path = ROOT / Path(*relative.parts)
        if suffix not in TEXT_SUFFIXES or not path.is_file():
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for label, pattern in PRIVATE_PATTERNS.items():
            if pattern.search(content):
                failures.append(f"{label}: {raw}")

    native_cmake = ROOT / "native" / "audio.cpp" / "CMakeLists.txt"
    if native_cmake.is_file() and "research/" in native_cmake.read_text(encoding="utf-8"):
        failures.append("native CMake depends on a private research path")

    if failures:
        print("Publication check FAILED:")
        for failure in sorted(set(failures)):
            print(f"  - {failure}")
        return 1
    print(f"Publication check PASS: {len(files)} prospective files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
