"""Build the optimized Windows CUDA worker from an editable source checkout."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=0)
    parser.add_argument(
        "--cuda-arch",
        default="auto",
        help="CMake CUDA architecture (default: detect the current GPU/toolchain)",
    )
    parser.add_argument("--version", default="0.1.0")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    native_root = root / "native" / "audio.cpp"
    script = native_root / "scripts" / "build_windows.ps1"
    if sys.platform != "win32" or not script.is_file():
        parser.error("fish-s2-build currently requires a Windows source checkout")
    command = [
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(script),
        "-Preset",
        "windows-cuda-portable-release",
        "-Target",
        "audiocpp_cli",
        "-CudaArchitectures",
        args.cuda_arch,
        "-CpuArch",
        "avx2",
        "-DeploymentBuild",
        "-ModelSet",
        "custom",
        "-Models",
        "fish_audio",
        "-Version",
        args.version,
    ]
    if args.jobs > 0:
        command.extend(["-Jobs", str(args.jobs)])
    return subprocess.call(command, cwd=native_root)


if __name__ == "__main__":
    raise SystemExit(main())
