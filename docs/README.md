# FishS2RT documentation

This is the documentation index for the `fish_s2` Python package and its
optimized native Fish Audio S2 Pro runtime.

## Start here

| Goal | Document |
|---|---|
| Install from Git or a Windows wheel | [Installation](installation.md) |
| Generate the first voice clone | [Python SDK guide](python-sdk.md) |
| Look up every class, method, option and exception | [API reference](api-reference.md) |
| Understand Q8, BF16, P2/P3, sampling and segmentation | [Configuration](configuration.md) |
| Play or forward audio while inference is running | [Streaming](streaming.md) |
| Copy practical integration examples | [Recipes](recipes.md) |
| Run complete command-line examples | [Examples](../examples/README.md) |
| Diagnose model, runtime, CUDA, audio or lifecycle failures | [Troubleshooting](troubleshooting.md) |
| Understand the worker, temporary files and extension points | [Architecture](architecture.md) |
| Check model hashes and licenses | [Models and licenses](models-and-licenses.md) |
| Prepare a public repository or release | [Publishing](publishing.md) and [release checklist](release-checklist.md) |

## Capability matrix

| Capability | v0.1 status | Notes |
|---|---|---|
| Q8_0 GGUF | Supported and locally smoke-tested | Lower-memory profile on the measured system |
| BF16 GGUF | Supported by the same API | Uses substantially more VRAM |
| Persistent model process | Supported | Reused until `unload()`/`close()` |
| Eager reference preparation/cache | Supported | `prepare_voice()` |
| Live PCM streaming | Supported | `AudioChunk`, signed PCM16 little-endian |
| Final WAV/bytes/in-memory PCM | Supported | `Audio.save()`, `to_wav_bytes()`, `pcm` |
| Exact sentence isolation | Supported | Explicit `segmentation="sentences"` |
| Numeric lower-entropy preset | Supported | `Sampling.numbers_safe()`; not a correctness guarantee |
| Multiple sequential generations | Supported | Same worker PID and loaded weights |
| Concurrent requests on one instance | Rejected | Raises `BusyError` |
| Multiple model instances | Technically possible | Each consumes its own VRAM |
| Graceful cooperative cancellation | Not yet | Early close hard-stops the owned worker |
| Reference MP3/FLAC decoding | Not in core SDK | Convert explicitly to WAV |
| Multi-reference public API | Not yet | Native capability exists but is not exposed safely in v0.1 |
| Built-in playback | Intentionally excluded | Applications choose their own audio backend |
| GUI or HTTP server | Intentionally excluded | SDK is an embedding layer |
| Automatic model download | Intentionally excluded | User supplies and licenses weights |
| Direct in-process Python/C binding | Not yet | Current stable backend is an isolated worker |

## Supported environment

The v0.1 release family targets Windows 10/11 x64, an AVX2-capable CPU, a
compatible NVIDIA CUDA stack and Python 3.10+. Source builds select a CUDA
architecture explicitly or detect it locally. The first complete validation
matrix and the published performance table used an RTX 5070 12 GB.

The package has no required Python runtime dependencies. Optional immediate
playback examples use `sounddevice` through the `audio` extra.
