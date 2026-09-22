"""High-level persistent Fish S2 model and streaming API."""

from __future__ import annotations

import json
import re
import secrets
import shutil
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal, Mapping

from ._runtime import RuntimeCommand, RuntimeInfo, inspect_runtime, resolve_runtime
from ._text import split_sentences_exact
from ._worker import NativeWorker
from .audio import Audio, AudioChunk, concatenate_audio, read_pcm16_wav
from .errors import BusyError, InvalidRequestError, ProtocolError, WorkerCrashedError
from .options import RuntimeConfig, Sampling

_STREAM_LINE = re.compile(r"^audio_out\[(?P<id>.+_stream_(?P<sequence>\d+))\]=(?P<path>.+)$")
_FINAL_LINE = re.compile(r"^audio_out=(?P<path>.+)$")
_METRIC_LINE = re.compile(r"^metrics\[(?P<id>[^]]+)]\.(?P<name>[^=]+)=(?P<value>.*)$")


@dataclass(frozen=True, slots=True)
class Voice:
    """A prepared reference voice owned by one loaded :class:`FishS2` instance."""

    reference_wav: Path
    transcript: str
    _owner: str


class AudioStream(Iterator[AudioChunk]):
    """Iterator over live chunks; ``result()`` returns the final native render."""

    def __init__(
        self,
        model: "FishS2",
        text: str,
        voice: Voice | None,
        sampling: Sampling,
        segmentation: Literal["native", "sentences"],
        pause_ms: int,
    ) -> None:
        self._model = model
        self._iterator = model._run_request(
            text=text,
            voice=voice,
            sampling=sampling,
            segmentation=segmentation,
            pause_ms=pause_ms,
        )
        self._result: Audio | None = None
        self._started = False
        self._done = False

    def __iter__(self) -> "AudioStream":
        return self

    def __next__(self) -> AudioChunk:
        if self._done:
            raise StopIteration
        self._started = True
        try:
            return next(self._iterator)
        except StopIteration as stop:
            self._result = stop.value
            self._done = True
            raise

    def result(self) -> Audio:
        if not self._done:
            for _ in self:
                pass
        if self._result is None:
            raise ProtocolError("stream ended without a final result")
        return self._result

    def close(self) -> None:
        if self._done:
            return
        self._iterator.close()
        self._done = True
        if self._started:
            self._model._cancel_active_request()

    def __enter__(self) -> "AudioStream":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class FishS2:
    """One persistent native Fish S2 model session.

    The instance owns exactly one native process. It keeps model weights and the
    reference cache alive until :meth:`unload` is called.
    """

    def __init__(
        self,
        model: str | Path,
        *,
        runtime: RuntimeCommand | None = None,
        config: RuntimeConfig | None = None,
    ) -> None:
        self.model_path = Path(model).expanduser().resolve()
        self.runtime_command = resolve_runtime(runtime)
        self.config = config or RuntimeConfig()
        self._owner = secrets.token_hex(16)
        self._temp: tempfile.TemporaryDirectory[str] | None = None
        self._worker: NativeWorker | None = None
        self._busy_lock = threading.Lock()
        self._default_voice: Voice | None = None
        self._runtime_info: RuntimeInfo | None = None

    @property
    def loaded(self) -> bool:
        return self._worker is not None and self._worker.alive

    @property
    def pid(self) -> int | None:
        return self._worker.pid if self._worker is not None else None

    @property
    def runtime_info(self) -> RuntimeInfo | None:
        """Validated worker identity, available after :meth:`load`."""

        return self._runtime_info

    def load(self) -> "FishS2":
        if self.loaded:
            return self
        if not self.model_path.is_file():
            raise InvalidRequestError(f"model does not exist: {self.model_path}")
        self._runtime_info = inspect_runtime(
            self.runtime_command,
            bf16_storage=self.config.bf16_storage,
            timeout=min(self.config.load_timeout, 30.0),
        )
        self._temp = tempfile.TemporaryDirectory(prefix="fish-s2-sdk-")
        work_dir = Path(self._temp.name)
        worker = NativeWorker(self.runtime_command, self.model_path, self.config, work_dir)
        try:
            worker.start()
        except Exception:
            worker.close(force=True)
            self._temp.cleanup()
            self._temp = None
            self._runtime_info = None
            raise
        self._worker = worker
        return self

    def unload(self) -> None:
        worker, self._worker = self._worker, None
        if worker is not None:
            worker.close(force=False)
        if self._temp is not None:
            self._temp.cleanup()
            self._temp = None
        self._default_voice = None
        self._runtime_info = None
        self._owner = secrets.token_hex(16)

    close = unload

    def prepare_voice(
        self,
        reference_wav: str | Path,
        transcript: str,
        *,
        make_default: bool = True,
    ) -> Voice:
        self.load()
        path = Path(reference_wav).expanduser().resolve()
        if not path.is_file():
            raise InvalidRequestError(f"reference WAV does not exist: {path}")
        if path.suffix.lower() != ".wav":
            raise InvalidRequestError("v0.1 accepts WAV references only; convert MP3/FLAC explicitly")
        transcript = transcript.strip()
        if not transcript:
            raise InvalidRequestError("reference transcript is empty")
        voice = Voice(path, transcript, self._owner)
        request_dir = self._new_request_dir("prepare")
        request_file = request_dir / "request.json"
        request_file.write_text(
            json.dumps(
                {
                    "requests": [
                        {
                            "id": "prepare_voice",
                            "text": ".",
                            "voice_ref": str(path),
                            "reference_text": transcript,
                            "max_tokens": 16,
                        }
                    ]
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        command_id = "prepare-" + secrets.token_hex(8)
        self._acquire_busy()
        try:
            worker = self._require_worker()
            worker.send(
                {"command": "prepare", "id": command_id, "request_sequence": str(request_file)}
            )
            for _ in worker.wait_for_done(command_id):
                pass
        finally:
            self._busy_lock.release()
            shutil.rmtree(request_dir, ignore_errors=True)
        if make_default:
            self._default_voice = voice
        return voice

    def stream(
        self,
        text: str,
        *,
        voice: Voice | None = None,
        sampling: Sampling | None = None,
        segmentation: Literal["native", "sentences"] = "native",
        pause_ms: int = 0,
    ) -> AudioStream:
        if not text:
            raise InvalidRequestError("text is empty")
        if segmentation not in {"native", "sentences"}:
            raise InvalidRequestError("segmentation must be native or sentences")
        if pause_ms < 0:
            raise InvalidRequestError("pause_ms must be non-negative")
        selected_voice = voice if voice is not None else self._default_voice
        if selected_voice is not None and selected_voice._owner != self._owner:
            raise InvalidRequestError("voice belongs to a different or unloaded FishS2 instance")
        self.load()
        return AudioStream(
            self,
            text,
            selected_voice,
            sampling or Sampling(),
            segmentation,
            pause_ms,
        )

    def generate(self, text: str, **kwargs: object) -> Audio:
        return self.stream(text, **kwargs).result()  # type: ignore[arg-type]

    def generate_to_file(self, text: str, path: str | Path, **kwargs: object) -> Path:
        return self.generate(text, **kwargs).save(path)

    def _run_request(
        self,
        *,
        text: str,
        voice: Voice | None,
        sampling: Sampling,
        segmentation: Literal["native", "sentences"],
        pause_ms: int,
    ) -> Iterator[AudioChunk]:
        self._acquire_busy()
        request_dir = self._new_request_dir("generate")
        try:
            segments = [text] if segmentation == "native" else split_sentences_exact(text)
            requests = [
                self._request_json(segment, index, voice, sampling, segmentation)
                for index, segment in enumerate(segments)
            ]
            request_file = request_dir / "requests.json"
            request_file.write_text(
                json.dumps({"requests": requests}, ensure_ascii=False), encoding="utf-8"
            )
            command_id = "generate-" + secrets.token_hex(8)
            worker = self._require_worker()
            worker.send(
                {
                    "command": "run",
                    "id": command_id,
                    "request_sequence": str(request_file),
                    "out_dir": str(request_dir),
                }
            )
            final_parts: list[Audio] = []
            metrics: dict[str, float | str] = {}
            global_sequence = 0
            for line in worker.wait_for_done(command_id):
                stream_match = _STREAM_LINE.match(line)
                if stream_match:
                    path = Path(stream_match.group("path"))
                    audio = read_pcm16_wav(path)
                    request_id = stream_match.group("id").rsplit("_stream_", 1)[0]
                    segment = _segment_from_id(request_id)
                    path.unlink(missing_ok=True)
                    yield AudioChunk(
                        pcm=audio.pcm,
                        sample_rate=audio.sample_rate,
                        channels=audio.channels,
                        sequence=global_sequence,
                        segment=segment,
                        request_id=request_id,
                    )
                    global_sequence += 1
                    continue
                final_match = _FINAL_LINE.match(line)
                if final_match:
                    path = Path(final_match.group("path"))
                    final_parts.append(read_pcm16_wav(path))
                    path.unlink(missing_ok=True)
                    continue
                metric_match = _METRIC_LINE.match(line)
                if metric_match:
                    key = f"{metric_match.group('id')}.{metric_match.group('name')}"
                    metrics[key] = _number_or_text(metric_match.group("value"))
            result = concatenate_audio(final_parts, pause_ms=pause_ms)
            return Audio(result.pcm, result.sample_rate, result.channels, metrics, sampling.seed)
        except (TimeoutError, WorkerCrashedError):
            self._cancel_active_request()
            raise
        finally:
            if self._busy_lock.locked():
                self._busy_lock.release()
            shutil.rmtree(request_dir, ignore_errors=True)

    def _request_json(
        self,
        text: str,
        index: int,
        voice: Voice | None,
        sampling: Sampling,
        segmentation: Literal["native", "sentences"],
    ) -> dict[str, object]:
        request: dict[str, object] = {
            "id": f"part_{index:04d}",
            "text": text,
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "top_k": sampling.top_k,
            "max_tokens": sampling.max_tokens,
            "text_chunk_size": (
                max(32768, len(text) + 1)
                if segmentation == "sentences"
                else sampling.text_chunk_size
            ),
            "text_chunk_mode": sampling.text_chunk_mode,
            "options": {
                **dict(sampling.request_options),
                "fish_audio.stream_chunk_frames": sampling.stream_chunk_frames,
            },
        }
        if sampling.seed is not None:
            request["seed"] = (sampling.seed + index) & 0xFFFFFFFF
        if voice is not None:
            request["voice_ref"] = str(voice.reference_wav)
            request["reference_text"] = voice.transcript
        return request

    def _new_request_dir(self, prefix: str) -> Path:
        if self._temp is None:
            raise ProtocolError("model temporary directory is unavailable")
        path = Path(self._temp.name) / f"{prefix}-{secrets.token_hex(8)}"
        path.mkdir(parents=True)
        return path

    def _require_worker(self) -> NativeWorker:
        if self._worker is None or not self._worker.alive:
            raise ProtocolError("model is not loaded")
        return self._worker

    def _acquire_busy(self) -> None:
        if not self._busy_lock.acquire(blocking=False):
            raise BusyError("this FishS2 instance already has a request in flight")

    def _cancel_active_request(self) -> None:
        worker, self._worker = self._worker, None
        if worker is not None:
            worker.close(force=True)
        if self._temp is not None:
            self._temp.cleanup()
            self._temp = None
        self._default_voice = None
        self._runtime_info = None
        self._owner = secrets.token_hex(16)
        if self._busy_lock.locked():
            self._busy_lock.release()

    def __enter__(self) -> "FishS2":
        return self.load()

    def __exit__(self, *_: object) -> None:
        self.unload()


def _segment_from_id(request_id: str) -> int:
    match = re.search(r"part_(\d+)$", request_id)
    return int(match.group(1)) if match else 0


def _number_or_text(value: str) -> float | str:
    try:
        return float(value)
    except ValueError:
        return value
