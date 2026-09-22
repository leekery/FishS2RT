# Installation

## Supported v0.1 environment

- Windows 10/11 x64;
- Python 3.10 or newer;
- an AVX2-capable x64 CPU;
- a compatible NVIDIA GPU, driver and CUDA build/runtime stack;
- a user-supplied Fish Audio S2 Pro Q8_0 or BF16 GGUF.

RTX 5070 12 GB with CUDA 12.8 and architecture `120a` is the first fully
measured configuration, not a hardware restriction. Source builds can select
another CUDA architecture. A binary release must state which architectures it
contains.

## Source checkout

```powershell
git clone https://github.com/leekery/FishS2RT.git
cd FishS2RT
python -m pip install -e .
fish-s2-build
```

The editable install exposes `import fish_s2` and the `fish-s2-build` command.
The final command builds the Fish-only native worker from
`native/audio.cpp`; it does not download model weights.

Native prerequisites:

- Visual Studio 2022 C++ Build Tools with MSVC x64;
- CMake and Ninja;
- a CUDA Toolkit supporting the selected architecture;
- enough free disk space for a C++/CUDA build tree.

On Windows, keep the checkout at a short path such as `C:\src\FishS2RT`.
NVIDIA's compiler can fail to create dependency files when a deeply nested
source and build path exceeds the traditional Windows path limit.

PowerShell 7 is not required. The build command uses Windows
`powershell.exe`. Automatic local architecture detection is the default:

```powershell
fish-s2-build
```

Select an architecture explicitly when producing a machine-specific binary:

```powershell
fish-s2-build --cuda-arch 120a-real --jobs 8
```

The worker is written below
`native/audio.cpp/build/windows-cuda-portable-release/bin/` and discovered
automatically by an editable SDK install.

## Binary release

FishS2RT intentionally separates two Python distributions while keeping one
Git repository and one release tag:

- `fish_s2`: the platform-independent Python SDK;
- `fishs2rt-runtime`: the matching Windows x64 worker, CUDA/MSVC runtime DLLs,
  build information, licenses and manifest.

When both are published, install them together with:

```powershell
python -m pip install "fish_s2[runtime]"
```

The import is still:

```python
from fish_s2 import FishS2
```

The SDK validates product identity, protocol version and capabilities by
running `audiocpp_cli.exe --build-info-json` before any model or GPU allocation.
An incompatible or arbitrary old worker fails early with `ProtocolError`.

## Verify without loading a model

```powershell
python -c "import fish_s2; print(fish_s2.__version__, fish_s2.__file__)"
python -c "from fish_s2._runtime import resolve_runtime, inspect_runtime; c=resolve_runtime(None); print(c); print(inspect_runtime(c))"
```

`fish_s2._runtime` is internal and intended here only for diagnosis. Normal
applications simply construct `FishS2`.

## Runtime discovery order

1. explicit `runtime=` constructor argument;
2. `FISH_S2_RUNTIME` environment variable;
3. installed `fishs2rt-runtime` wheel;
4. legacy in-package worker, if present;
5. known source-checkout build directories.

An explicit custom worker is allowed but still must pass the FishS2RT protocol
and capability handshake.

```python
tts = FishS2(
    "model.gguf",
    runtime=r"D:\fish-runtime\audiocpp_cli.exe",
)
```

```powershell
$env:FISH_S2_RUNTIME = "D:\fish-runtime\audiocpp_cli.exe"
python app.py
```

## Model weights

Installation never downloads, converts or bundles weights. Keep weights outside
the repository and pass their path:

```python
from fish_s2 import FishS2

tts = FishS2(r"D:\models\fish-audio-s2-pro-q8_0.gguf")
```

The same constructor accepts a compatible BF16 GGUF. Precision comes from the
file, not a hidden SDK switch. Record size and SHA-256 for reproducible tests:

```powershell
Get-Item D:\models\fish-audio-s2-pro-q8_0.gguf | Select-Object Length
Get-FileHash -Algorithm SHA256 D:\models\fish-audio-s2-pro-q8_0.gguf
```

Review [models and licenses](models-and-licenses.md) before redistributing
anything. Model weights have separate terms from the Apache-licensed runtime.

## Reference audio

v0.1 accepts WAV references. Convert other formats explicitly so the SDK never
silently changes the reference PCM:

```powershell
ffmpeg -i reference.mp3 -ac 1 -ar 44100 -c:a pcm_s16le reference.wav
```

Use an exact transcript of the converted reference.

## Optional playback dependency

The SDK does not require a playback package. Install the optional example
dependency only when needed:

```powershell
python -m pip install -e ".[audio]"
```

This installs `sounddevice`; application code may instead use another playback,
network or UI layer.

## Minimal end-to-end smoke

This allocates GPU memory and runs inference:

```python
from fish_s2 import FishS2

with FishS2(r"D:\models\fish-audio-s2-pro-q8_0.gguf") as tts:
    print("runtime:", tts.runtime_info)
    voice = tts.prepare_voice(
        r"D:\voices\reference.wav",
        "Exact reference transcript.",
    )
    audio = tts.generate("Installation test.", voice=voice)
    print("worker pid:", tts.pid)
    print("duration:", audio.duration, "metrics:", dict(audio.metrics))
    audio.save("installation-test.wav")
```

Successful import validates packaging only. A meaningful GPU smoke must load
the actual model, prepare the intended voice, generate non-empty audio and
confirm that `unload()` releases the worker and GPU memory.

## Build the two wheels

Build the pure SDK artifacts:

```powershell
python -m build
```

After compiling and validating the worker, stage a runtime wheel:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File scripts/build_runtime_wheel.ps1 `
  -Runtime native/audio.cpp/build/windows-cuda-portable-release/bin/audiocpp_cli.exe
```

The runtime wheel builder queries the binary manifest, copies licenses into the
wheel and emits a `py3-none-win_amd64` artifact. It does not leave the binary
tracked in the source tree.

Before release, follow [the release checklist](release-checklist.md). In
particular, install built wheels in a fresh environment outside the checkout;
editable installation can hide missing package files.

## Offline installation

Copy both wheels plus the separately licensed GGUF and required runtime DLLs to
the target machine:

```powershell
python -m pip install --no-index `
  .\fish_s2-0.1.0-py3-none-any.whl `
  .\fishs2rt_runtime-0.1.0-py3-none-win_amd64.whl
```

Release notes must state whether CUDA/MSVC redistributable DLLs are bundled.
Never copy DLLs from untrusted download sites.

## Uninstall

```powershell
python -m pip uninstall fish_s2 fishs2rt-runtime
```

This removes installed distributions, not a Git checkout, models or generated
audio.

## Next steps

- [First integration and lifecycle](python-sdk.md)
- [Configuration](configuration.md)
- [Live streaming and playback](streaming.md)
- [Troubleshooting](troubleshooting.md)
