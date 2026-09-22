"""Native runtime discovery and compatibility checks."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .errors import ProtocolError, RuntimeNotFoundError

RuntimeCommand = str | os.PathLike[str] | Sequence[str | os.PathLike[str]]
PROTOCOL_VERSION = 1
SDK_VERSION = "0.1.0"
_BASE_CAPABILITIES = frozenset({"streaming", "reference_cache", "q8", "bf16"})


@dataclass(frozen=True, slots=True)
class RuntimeInfo:
    """Identity and capabilities reported by a native worker."""

    product: str
    runtime_version: str
    protocol_version: int
    source_commit: str
    upstream_commit: str
    capabilities: frozenset[str]


def resolve_runtime(runtime: RuntimeCommand | None) -> tuple[str, ...]:
    """Resolve an explicit, environment, runtime-wheel or source-build worker."""

    if runtime is not None:
        if isinstance(runtime, (str, os.PathLike)):
            command = (str(Path(runtime).expanduser().resolve()),)
        else:
            command = tuple(str(item) for item in runtime)
        if not command:
            raise RuntimeNotFoundError("runtime command is empty")
        if not Path(command[0]).is_file():
            raise RuntimeNotFoundError(f"native runtime does not exist: {command[0]}")
        return command

    env_runtime = os.environ.get("FISH_S2_RUNTIME")
    if env_runtime:
        path = Path(env_runtime).expanduser().resolve()
        if path.is_file():
            return (str(path),)
        raise RuntimeNotFoundError(f"FISH_S2_RUNTIME does not exist: {path}")

    executable = "audiocpp_cli.exe" if sys.platform == "win32" else "audiocpp_cli"
    package_dir = Path(__file__).resolve().parent
    candidates: list[Path] = []

    try:
        from fishs2rt_runtime import runtime_path

        candidates.append(runtime_path())
    except (ImportError, FileNotFoundError, RuntimeError):
        pass

    candidates.append(package_dir / "_native" / executable)
    source_root = package_dir.parent.parent
    candidates.extend(
        [
            source_root
            / "native"
            / "audio.cpp"
            / "build"
            / "windows-cuda-portable-release"
            / "bin"
            / executable,
            source_root
            / "native"
            / "audio.cpp"
            / "build"
            / "windows-cuda-release"
            / "bin"
            / executable,
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return (str(candidate),)
    rendered = "\n".join(f"  - {path}" for path in candidates)
    raise RuntimeNotFoundError(
        "a compatible native runtime was not found. Pass runtime=..., set "
        "FISH_S2_RUNTIME, install the matching fishs2rt-runtime wheel, or run "
        "`fish-s2-build` from a source checkout. Checked:\n" + rendered
    )


def inspect_runtime(
    command: tuple[str, ...],
    *,
    bf16_storage: str = "raw",
    timeout: float = 15.0,
) -> RuntimeInfo:
    """Query and validate a worker before it allocates model or GPU memory."""

    try:
        completed = subprocess.run(
            [*command, "--build-info-json"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeNotFoundError(f"cannot inspect native runtime: {exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise ProtocolError(
            f"native runtime build-info query failed with exit code {completed.returncode}"
            + (f": {detail}" if detail else "")
        )
    try:
        payload = json.loads(completed.stdout)
        info = RuntimeInfo(
            product=str(payload["product"]),
            runtime_version=str(payload["runtime_version"]),
            protocol_version=int(payload["protocol_version"]),
            source_commit=str(payload["source_commit"]),
            upstream_commit=str(payload["upstream_commit"]),
            capabilities=frozenset(str(item) for item in payload["capabilities"]),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ProtocolError("native runtime returned invalid --build-info-json output") from exc

    if info.product != "FishS2RT":
        raise ProtocolError(f"expected a FishS2RT runtime, got {info.product!r}")
    if info.protocol_version != PROTOCOL_VERSION:
        raise ProtocolError(
            f"runtime protocol {info.protocol_version} is incompatible with SDK protocol "
            f"{PROTOCOL_VERSION}"
        )
    required = set(_BASE_CAPABILITIES)
    if bf16_storage in {"p2", "p3"}:
        required.add(f"bf16_{bf16_storage}")
    missing = sorted(required - info.capabilities)
    if missing:
        raise ProtocolError("native runtime is missing capabilities: " + ", ".join(missing))
    return info
