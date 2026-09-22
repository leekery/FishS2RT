# Complete Python API reference

Package import:

```python
import fish_s2

print(fish_s2.__version__)
```

The public package exports `FishS2`, `Voice`, `AudioStream`, `Audio`,
`AudioChunk`, `Sampling`, `RuntimeConfig`, `RuntimeInfo` and the exception
classes documented below. Names beginning with `_` are implementation details
and have no stability guarantee.

## `FishS2`

```text
FishS2(
    model: str | pathlib.Path,
    *,
    runtime: str | os.PathLike | Sequence[str | os.PathLike] | None = None,
    config: RuntimeConfig | None = None,
)
```

One instance owns at most one native worker and one loaded GGUF. Constructing an
instance resolves the runtime path but does not load model weights. `load()`, a
context-manager entry, `prepare_voice()`, `stream()` or `generate()` performs
the load.

Parameters:

- `model`: path to a Fish S2 Q8_0 or BF16 GGUF. The SDK never downloads,
  converts, quantizes or edits it.
- `runtime`: optional native command. A path is the normal form. A sequence is
  primarily useful for testing or a custom wrapper, for example
  `(sys.executable, "worker.py")`.
- `config`: process/session settings fixed for that load. Changing the object
  after construction is not supported; unload and create a new instance.

Runtime discovery order when `runtime=None`:

1. explicit `runtime=` argument;
2. `FISH_S2_RUNTIME` environment variable;
3. the matching `fishs2rt-runtime` distribution;
4. a legacy `fish_s2/_native/audiocpp_cli.exe`;
5. known `native/audio.cpp/build/...` source-build directories.

### Properties

#### `loaded: bool`

`True` only while the owned worker exists and has not exited. It does not probe
GPU memory independently.

#### `pid: int | None`

Native worker PID or `None`. It is useful for diagnostics and for confirming
that sequential calls reuse the same process.

```python
with FishS2("model.gguf") as tts:
    first_pid = tts.pid
    tts.generate("One.")
    tts.generate("Two.")
    assert tts.pid == first_pid
```

#### `runtime_info: RuntimeInfo | None`

`None` before load. After a successful pre-load handshake it identifies the
runtime version, protocol, source commits and capabilities. It is cleared by
`unload()` and by hard cancellation.

### `load() -> FishS2`

Loads the GGUF and creates the streaming Fish session. Repeated calls while the
worker is alive are idempotent and return the same instance.

Possible failures include `InvalidRequestError` for a missing model,
`RuntimeNotFoundError` during construction, and `ModelLoadError` or
`WorkerCrashedError` for native startup failures.

### `unload() -> None` / `close() -> None`

Requests a graceful worker shutdown, then force-stops it if it does not exit in
five seconds. Temporary request files, the default `Voice` and its ownership
generation are discarded. `close` is an alias of `unload`.

A `Voice` created before unload cannot be used after a later reload.

```python
tts = FishS2("model.gguf").load()
try:
    tts.generate("Loaded once.")
finally:
    tts.unload()
```

Do not call `load()`/`unload()` concurrently with generation from another
thread. The supported concurrency contract is one request at a time per
instance.

### Context manager

`__enter__` calls `load()` and `__exit__` calls `unload()` even when the block
raises.

```python
with FishS2("model.gguf") as tts:
    audio = tts.generate("Safe lifetime management.")
```

### `prepare_voice()`

```text
prepare_voice(
    reference_wav: str | pathlib.Path,
    transcript: str,
    *,
    make_default: bool = True,
) -> Voice
```

Validates an existing `.wav`, strips surrounding transcript whitespace, loads
the model if necessary and asks the native session to prepare/cache reference
codes. The file must remain available while the voice is used because requests
still carry its path; the native cache avoids repeated reference encoding when
the content matches.

- `make_default=True`: later calls without `voice=` use this voice.
- `make_default=False`: only calls explicitly given the returned handle use it.

```python
voice = tts.prepare_voice("speaker.wav", "Exact transcript.", make_default=False)
audio = tts.generate("Uses the explicit voice.", voice=voice)
```

Only WAV references are accepted in v0.1. The SDK intentionally does not hide a
lossy/implicit MP3 conversion. The transcript should match the spoken reference
as closely as possible.

### `stream()`

```text
stream(
    text: str,
    *,
    voice: Voice | None = None,
    sampling: Sampling | None = None,
    segmentation: Literal["native", "sentences"] = "native",
    pause_ms: int = 0,
) -> AudioStream
```

Loads the model if needed and returns a lazy `AudioStream`. The actual request
starts on first iteration or `result()`, not merely when the object is created.

- `text`: passed without digit/tag rewriting. An empty string is rejected.
- `voice`: explicit handle. When omitted, the default prepared voice is used;
  if no default exists the request is unconditioned.
- `sampling`: defaults to `Sampling()`.
- `segmentation="native"`: one request and Fish's internal continuation/chunking.
- `segmentation="sentences"`: the SDK sends independently conditioned sentence
  requests through the same worker. It preserves every input character.
- `pause_ms`: zero-valued PCM inserted only between final sentence outputs.
  It does not alter live chunks.

The stream must be exhausted or closed, preferably with `with`.

### `generate()`

```text
generate(text: str, **stream_keyword_arguments) -> Audio
```

Convenience wrapper that creates a stream, drains all live events and returns
its final `Audio`. It still executes the native streaming path; the discarded
live chunks are not used to construct the final result.

```python
audio = tts.generate(
    "One sentence. Another sentence.",
    sampling=Sampling(seed=42),
    segmentation="sentences",
    pause_ms=120,
)
```

### `generate_to_file()`

```text
generate_to_file(
    text: str,
    path: str | pathlib.Path,
    **stream_keyword_arguments,
) -> pathlib.Path
```

Equivalent to `generate(...).save(path)`. Parent directories are created and an
absolute resolved output path is returned.

## `Voice`

Frozen value object returned by `prepare_voice()`.

| Field | Type | Meaning |
|---|---|---|
| `reference_wav` | `Path` | Absolute source WAV path |
| `transcript` | `str` | Trimmed exact transcript |
| `_owner` | internal | Binds the handle to one loaded instance generation |

It is not a serialized speaker embedding and cannot be shared between
`FishS2` instances. Re-run `prepare_voice()` for another instance or after
unload/reload.

## `AudioStream`

An iterator and context manager yielding `AudioChunk`.

### Iteration

```python
with tts.stream("Live output.") as stream:
    for chunk in stream:
        consume(chunk)
    final = stream.result()
```

`sequence` is global and contiguous across all isolated sentence requests.
`segment` identifies the zero-based isolated sentence. For native segmentation
it is normally zero.

### `result() -> Audio`

Drains the stream if needed, then returns the final native render. Repeated calls
after completion return the same object. The final render, not a concatenation
of live stateful delta-decoder chunks, is authoritative for saving.

### `close() -> None`

- Before iteration begins: abandons the unused request object; the loaded model
  remains alive.
- After generation begins but before completion: hard-cancels by terminating the
  owned worker. The `FishS2` instance becomes unloaded and prepared `Voice`
  handles become invalid.
- After completion: no-op.

There is no cooperative keep-model-loaded cancellation in v0.1.

## `Audio`

Frozen final-audio value object.

| Member | Type | Meaning |
|---|---|---|
| `pcm` | `bytes` | Interleaved signed PCM16 little-endian |
| `sample_rate` | `int` | Native output sample rate |
| `channels` | `int` | Native channel count |
| `metrics` | mapping | Per-request native metrics |
| `seed` | `int | None` | Base sampling seed requested by the caller |
| `frames` | property | Complete interleaved frames |
| `duration` | property | `frames / sample_rate` seconds |

`to_wav_bytes() -> bytes` returns an uncompressed PCM16 WAV container.
`save(path) -> Path` writes those bytes, creates parents and returns the absolute
path.

```python
audio = tts.generate("Return bytes instead of a file.")
wav_body = audio.to_wav_bytes()
pcm_body = audio.pcm
```

Metrics keys currently follow `<request-id>.<native-name>`, for example
`part_0000.wall_ms`, `part_0000.audio_duration_ms`, `part_0000.rtf`,
`part_0000.x_realtime`, `part_0000.sample_rate` and
`part_0000.channels`. Treat the mapping as diagnostic data: native versions may
add keys.

## `AudioChunk`

Frozen live-audio value object.

| Member | Type | Meaning |
|---|---|---|
| `pcm` | `bytes` | Immediately playable PCM16 little-endian |
| `sample_rate` | `int` | Chunk sample rate |
| `channels` | `int` | Chunk channel count |
| `sequence` | `int` | Zero-based global event sequence |
| `segment` | `int` | Zero-based isolated sentence index |
| `request_id` | `str` | Native-safe request identifier |
| `frames` | property | Frame count |
| `duration` | property | Seconds |

`as_audio() -> Audio` wraps just that chunk as an `Audio` object, which is useful
for diagnostics or saving one chunk. It does not turn concatenated live audio
into the authoritative final render.

## `Sampling`

Frozen request configuration.

| Field | Default | Accepted values | Native effect |
|---|---:|---|---|
| `seed` | `None` | `0..2^32-1` | `None` lets native choose a random seed; isolated sentences use `(seed + index) mod 2^32` |
| `temperature` | `0.8` | `0 < value < 2` | Sampling entropy |
| `top_p` | `0.8` | `0 < value <= 1` | Nucleus cutoff |
| `top_k` | `30` | positive integer | Candidate limit |
| `max_tokens` | `1024` | positive integer | Maximum semantic tokens per request/chunk |
| `text_chunk_size` | `200` | positive integer | Native long-text character budget in native segmentation |
| `text_chunk_mode` | `tag_aware` | `default`, `tag_aware`, `japanese`, `endline` | Native text chunker |
| `stream_chunk_frames` | `8` | `1..256` | Codec frames emitted per live event |
| `request_options` | `{}` | string-key mapping | Advanced native request options |

`stream_chunk_frames` is written after `request_options`, so the typed value wins
if the advanced mapping contains the same key.

`Sampling.numbers_safe(seed=None, **overrides)` starts from temperature `0.65`,
top-p `0.78` and top-k `20`. It is a lower-entropy experimental preset, not a
promise that all numbers will be pronounced correctly.

```python
sampling = Sampling.numbers_safe(
    seed=42,
    max_tokens=768,
    stream_chunk_frames=6,
)
```

## `RuntimeConfig`

Frozen settings used when the worker starts.

| Field | Default | Meaning |
|---|---:|---|
| `backend` | `cuda` | Native backend string; this distribution is certified for CUDA |
| `device` | `0` | Non-negative backend device index |
| `threads` | `None` | Positive host thread count; `None` leaves native default |
| `reference_cache_slots` | `1` | Non-negative native reference cache capacity |
| `mem_saver` | `False` | Release AR runtime graphs after requests; trades memory for repeated-request speed |
| `bf16_storage` | `raw` | `raw`, `p2` or `p3`; meaningful for BF16 weights |
| `session_options` | `{}` | Advanced native session options |
| `load_timeout` | `300.0` | Positive seconds to wait for worker readiness |
| `command_timeout` | `None` | Positive per-command seconds, or unlimited |

Advanced `session_options` are appended after common typed options and may
override native map keys. Prefer typed fields when one exists.

The SDK sanitizes inherited optimization/debug environment flags before starting
its child. `bf16_storage="p2"` or `"p3"` then enables only the requested exact
storage profile in that child process. It never mutates global `os.environ`.

## `RuntimeInfo`

Immutable result of the native pre-load compatibility query.

| Field | Meaning |
|---|---|
| `product` | Must be `FishS2RT` |
| `runtime_version` | Release/build version reported by the worker |
| `protocol_version` | Integer control-protocol version checked by the SDK |
| `source_commit` | Git identity recorded by the native build |
| `upstream_commit` | audio.cpp source snapshot used by FishS2RT |
| `capabilities` | Frozen set such as `streaming`, `q8`, `bf16`, `bf16_p2` |

The handshake runs before the expensive model load. Selecting P2/P3 requires
the corresponding capability; missing or incompatible values raise
`ProtocolError`.

## Exceptions

All SDK-specific exceptions derive from `FishS2Error`, which derives from
`RuntimeError`.

| Exception | Typical cause | Worker state |
|---|---|---|
| `RuntimeNotFoundError` | No explicit/env/bundled/source runtime | No worker |
| `ModelLoadError` | Process cannot start or model load fails before ready | Closed |
| `InvalidRequestError` | Bad path, empty text/transcript, invalid option, stale/foreign `Voice` | Normally unchanged |
| `BusyError` | Second request on the same instance | Existing request continues |
| `GenerationError` | Native command rejects/fails a request | Worker normally remains reusable |
| `WorkerCrashedError` | Unexpected native exit/pipe closure | Instance is cancelled/unloaded during generation |
| `ProtocolError` | Missing/incompatible WAV or malformed worker response | Depends on failure point |

Python's built-in `TimeoutError` is raised for `load_timeout`/`command_timeout`.
A generation timeout hard-cancels the worker and leaves the instance unloaded.

```python
from fish_s2 import FishS2Error, GenerationError

try:
    audio = tts.generate("Request")
except GenerationError as exc:
    print("Native request failed:", exc)
except FishS2Error as exc:
    print("SDK/runtime failure:", exc)
```

## Thread and process safety

- A `FishS2` instance supports one in-flight request.
- Generation calls may originate from different threads only if the application
  serializes them. Otherwise the loser receives `BusyError`.
- Do not call `unload()` concurrently with a live request.
- Separate instances use separate processes, caches and VRAM.
- `Voice`, `AudioStream` and active requests are not transferable across
  processes or instances.
- `Audio` and completed `AudioChunk` objects are immutable byte containers and
  can be passed between application threads normally.
