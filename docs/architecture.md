# Architecture and extension guide

## Public boundary

The project has two layers:

```text
application code
    |
    | FishS2 / Voice / AudioStream / Audio
    v
Python SDK (src/fish_s2)
    |
    | persistent UTF-8 request-loop control + temporary WAV events
    v
optimized audio.cpp worker (audiocpp_cli.exe)
    |
    v
Fish S2 GGUF + CUDA backend
```

The SDK is not a second inference implementation. It owns lifecycle,
configuration, validation, streaming delivery and Python value objects while
the optimized C++ runtime performs model inference.

## Why a persistent subprocess in v0.1

- It preserves the already tested native CUDA/Fish implementation.
- One process keeps the GGUF and reference cache alive across requests.
- CUDA/DLL crashes are isolated from the embedding interpreter.
- Python ABI-specific extensions are unnecessary.
- Environment-based experimental profiles are scoped to one child.
- Terminating the owned process gives a definite GPU-release boundary.

The generic audio.cpp C API is useful for future embedding, but the current Fish
streaming implementation pushes events through an internal callback while its
pull-style `next_stream_event()` returns no events. A thin `ctypes` binding would
therefore not provide live Fish audio without native API work.

## Worker lifecycle

1. `FishS2` resolves a runtime command.
2. `load()` asks the worker for versioned JSON build information and rejects an
   incompatible protocol or missing capability before loading the model.
3. `load()` creates a private temporary directory and starts the worker without
   a shell or visible Windows console.
4. The CLI loads the GGUF, constructs a streaming session and prints readiness.
5. `prepare_voice()` sends a small request sequence that eagerly populates the
   reference cache.
6. `generate()`/`stream()` send request-sequence JSON while retaining the same
   worker.
7. Native callbacks write complete temporary PCM16 WAV blocks and announce each
   path on stdout.
8. Python reads each complete file, deletes it and yields immutable bytes.
9. Native final output is read separately and returned as `Audio`.
10. `unload()` requests quit, closes pipes and removes the private directory.

Stdout and stderr are drained by different daemon threads so one full pipe does
not block the other. The stdout event queue is bounded. Stderr retains only a
bounded tail for failure messages and does not intentionally log prompt/reference
content.

## Temporary data

Each instance uses `tempfile.TemporaryDirectory(prefix="fish-s2-sdk-")`.
Individual prepare/generate calls use random child directories. Successful live
and final WAVs are loaded into memory and deleted; the request directory is
removed in `finally`.

Applications that require a hardened privacy policy should additionally control
the OS temp location, crash dumps, swap/pagefile, filesystem ACLs and their own
logs. A process crash or power loss can leave OS-managed temporary artifacts;
the SDK does not claim secure deletion.

## Audio representation

The native worker writes uncompressed PCM16 WAV. Python validates sample width
and compression, strips the container and exposes interleaved signed little-
endian samples. WAV serialization uses the standard-library `wave` module.

No automatic normalization, limiter, dither, crossfade, resampling or silence is
applied. The only SDK transformation is explicit `pause_ms` silence between
final sentence-isolated parts.

## Live/final separation

The native streaming generator sends only newly confirmed codec frames through
a stateful DAC path. Its post-transformer keeps a device-side sliding KV cache
with the model's exact 128-frame causal window; the convolutional decoder
replays only the ten real post-transformer frames required at the left edge.
Only newly generated PCM is copied back. A separate exact-length canonical
decode is retained for the final result. The SDK labels these two channels
through types rather than pretending they are identical:

- `AudioChunk`: live playback/transport;
- `Audio`: authoritative final output.

This distinction must survive future protocols and bindings.

## Concurrency model

One instance owns one worker and a non-blocking request mutex. Starting a second
request raises `BusyError`; there is no hidden queue. A multi-user application
may implement:

- one application queue feeding one model instance;
- a bounded pool of instances when VRAM allows;
- separate OS workers/machines behind a service scheduler.

Never infer that Python threads create additional native model concurrency.

## Cancellation model

The native request loop reads its next command only after synchronous generation
returns. A `cancel` line written to stdin could not interrupt that work. v0.1
therefore uses truthful hard cancellation: close pipes/terminate the owned
process, invalidate voice handles, then require reload.

A future cooperative design requires an atomic native token checked between AR
steps and before DAC work. It must not call session reset concurrently with CUDA
kernels.

## Runtime/profile environment

Known Fish/BF16 experimental and diagnostic variables are removed from the
child environment. Requested P2/P3 settings are then applied to that child only.
This makes independent `FishS2` instances reproducible and prevents a stale
interactive shell variable from silently disabling an optimization.

## Package and release layout

```text
pyproject.toml
src/fish_s2/
native/audio.cpp/              source-only native snapshot
packaging/runtime/             Windows runtime-wheel project
tests/python/
tests/native/
tests/integration/
docs/
examples/
scripts/
UPSTREAM.md
runtime-lock.json
```

The SDK wheel is `py3-none-any`. The separately built runtime wheel is
`py3-none-win_amd64` and contains the executable, manifest, notices and selected
runtime DLLs. Both come from one Git tag and use one protocol lock. Weights are
never wheel contents.

## Extension points

### New application integration

Use the public classes only. Do not parse worker stdout or depend on temporary
filenames outside the SDK.

### New native worker build

Pass `runtime=...` and keep the same CLI/request-loop contract. Record binary
hash/version and rerun model-free plus Q8/BF16 smoke tests.

### Dedicated protocol v2

A future worker mode can replace temporary WAVs with versioned NDJSON/base64
PCM or binary framing. Required properties:

- stdout reserved for protocol and diagnostics on stderr;
- explicit protocol/build/capability hello (v1 already provides a pre-load
  `--build-info-json` query; a future transport should carry the same data);
- request IDs, contiguous sequence IDs and one terminal event;
- separate live and final channels;
- bounded messages and backpressure;
- no prompt/reference text in logs by default;
- cooperative cancellation only when truly supported.

The public `AudioStream` API can remain stable while `_worker.py` changes.

### In-process C API backend

Before adding `ctypes`/pybind:

- expose Fish push callbacks or an async event queue;
- define completion and callback lifetime;
- document handle thread-safety;
- add cancellation and timeout semantics;
- keep one session rather than duplicating offline/streaming weights;
- compare full final PCM/WAV with the subprocess backend.

## Verification layers

- Model-free tests use a fake request-loop worker to test installation,
  persistence, events, sentence joining, busy behavior and cancellation.
- Manual smoke uses the real optimized binary/model/reference and reports cold
  load, voice preparation, first chunk, generation wall and final duration.
- Matched performance validation fixes binary/model/settings/seed/reference and
  compares full output hashes/traces with counterbalanced process order.

Passing a fake-worker test proves the Python control plane, not model quality or
RTX performance. Passing one real smoke proves that request, not universal
quality.
