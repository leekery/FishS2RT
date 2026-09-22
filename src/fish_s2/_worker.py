"""Persistent subprocess adapter for the audio.cpp request loop."""

from __future__ import annotations

import collections
import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Iterable

from .errors import GenerationError, ModelLoadError, ProtocolError, WorkerCrashedError
from .options import RuntimeConfig

_EOF = object()
_CONTROLLED_ENV = {
    "FISH_AUDIO_EXPERIMENTAL_BF16_P2",
    "FISH_AUDIO_EXPERIMENTAL_BF16_P3",
    "FISH_AUDIO_EXPERIMENTAL_BF16_P2_SLOW",
    "GGML_CUDA_DISABLE_BF16_P2",
    "GGML_CUDA_DISABLE_BF16_P3",
    "FISH_DISABLE_FAST_INITIAL_PRUNE",
    "FISH_DISABLE_FUSED_BF16_ROUND",
    "FISH_DISABLE_F16_ATTENTION_CACHE",
    "FISH_DISABLE_F16_KV_STORAGE",
    "FISH_DISABLE_ASYNC_UPLOAD",
}


class NativeWorker:
    def __init__(
        self,
        runtime_command: tuple[str, ...],
        model_path: Path,
        config: RuntimeConfig,
        work_dir: Path,
    ) -> None:
        self._runtime_command = runtime_command
        self._model_path = model_path
        self._config = config
        self._work_dir = work_dir
        self._process: subprocess.Popen[str] | None = None
        self._stdout: queue.Queue[str | object] = queue.Queue(maxsize=128)
        self._stderr_tail: collections.deque[str] = collections.deque(maxlen=80)
        self._stderr_lock = threading.Lock()
        self._threads: list[threading.Thread] = []

    @property
    def pid(self) -> int | None:
        return self._process.pid if self._process is not None else None

    @property
    def alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self) -> None:
        if self.alive:
            return
        command = list(self._runtime_command)
        command.extend(
            [
                "--task",
                "tts",
                "--family",
                "fish_audio",
                "--model",
                str(self._model_path),
                "--mode",
                "streaming",
                "--backend",
                self._config.backend,
                "--device",
                str(self._config.device),
                "--request-loop",
                "--metrics",
                "--session-option",
                "fish_audio.eager_reference_cache=true",
                "--session-option",
                f"fish_audio.reference_cache_slots={self._config.reference_cache_slots}",
                "--session-option",
                f"fish_audio.mem_saver={'true' if self._config.mem_saver else 'false'}",
            ]
        )
        if self._config.threads is not None:
            command.extend(["--threads", str(self._config.threads)])
        for key, value in self._config.session_options.items():
            command.extend(["--session-option", f"{key}={_option_string(value)}"])

        env = os.environ.copy()
        for name in _CONTROLLED_ENV:
            env.pop(name, None)
        if self._config.bf16_storage in {"p2", "p3"}:
            env["FISH_AUDIO_EXPERIMENTAL_BF16_P2"] = "1"
        if self._config.bf16_storage == "p2":
            env["GGML_CUDA_DISABLE_BF16_P3"] = "1"
        elif self._config.bf16_storage == "p3":
            env["FISH_AUDIO_EXPERIMENTAL_BF16_P3"] = "1"

        startupinfo = None
        creationflags = 0
        if os.name == "nt":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            creationflags = subprocess.CREATE_NO_WINDOW
        try:
            self._process = subprocess.Popen(
                command,
                cwd=self._work_dir,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
                startupinfo=startupinfo,
                creationflags=creationflags,
            )
        except OSError as exc:
            raise ModelLoadError(f"cannot start native runtime: {exc}") from exc
        assert self._process.stdout is not None and self._process.stderr is not None
        self._threads = [
            threading.Thread(target=self._read_stdout, name="fish-s2-stdout", daemon=True),
            threading.Thread(target=self._read_stderr, name="fish-s2-stderr", daemon=True),
        ]
        for thread in self._threads:
            thread.start()
        deadline = time.monotonic() + self._config.load_timeout
        while True:
            line = self.next_line(deadline)
            if line == "request_loop_ready=1":
                return
            if line.startswith("request_loop_error="):
                self.close(force=True)
                raise ModelLoadError(line)

    def send(self, command: dict[str, object]) -> None:
        process = self._require_alive()
        assert process.stdin is not None
        try:
            process.stdin.write(json.dumps(command, ensure_ascii=False, separators=(",", ":")) + "\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise WorkerCrashedError(self._crash_message("native worker pipe closed")) from exc

    def next_line(self, deadline: float | None = None) -> str:
        timeout = None if deadline is None else max(0.0, deadline - time.monotonic())
        try:
            item = self._stdout.get(timeout=timeout)
        except queue.Empty as exc:
            raise TimeoutError("native worker command timed out") from exc
        if item is _EOF:
            raise WorkerCrashedError(self._crash_message("native worker exited"))
        assert isinstance(item, str)
        return item

    def wait_for_done(self, command_id: str) -> Iterable[str]:
        deadline = (
            None
            if self._config.command_timeout is None
            else time.monotonic() + self._config.command_timeout
        )
        done = f"request_loop_done={command_id}"
        error = f"request_loop_error={command_id}\t"
        while True:
            line = self.next_line(deadline)
            if line == done:
                return
            if line.startswith(error):
                raise GenerationError(line[len(error) :])
            yield line

    def close(self, *, force: bool = False) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        if process.poll() is None and not force:
            try:
                assert process.stdin is not None
                process.stdin.write('{"command":"quit","id":"close"}\n')
                process.stdin.flush()
                process.wait(timeout=5)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                force = True
        if process.poll() is None and force:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                stream.close()

    def _read_stdout(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        try:
            for line in self._process.stdout:
                self._stdout.put(line.rstrip("\r\n"))
        finally:
            self._stdout.put(_EOF)

    def _read_stderr(self) -> None:
        assert self._process is not None and self._process.stderr is not None
        for line in self._process.stderr:
            with self._stderr_lock:
                self._stderr_tail.append(line.rstrip("\r\n"))

    def _require_alive(self) -> subprocess.Popen[str]:
        if not self.alive or self._process is None:
            raise WorkerCrashedError(self._crash_message("native worker is not running"))
        return self._process

    def _crash_message(self, prefix: str) -> str:
        process = self._process
        code = process.poll() if process is not None else None
        with self._stderr_lock:
            tail = "\n".join(self._stderr_tail)
        detail = f"{prefix} (exit code {code})"
        return detail if not tail else detail + "\nNative stderr tail:\n" + tail


def _option_string(value: str | int | float | bool) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)
