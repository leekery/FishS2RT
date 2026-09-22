# FishS2RT

**English** | [Русский](README.ru.md)

FishS2RT is a Python SDK for accelerated native inference and real-time
streaming with Fish Audio S2 Pro. It combines a persistent optimized CUDA
runtime, live PCM delivery, exact final rendering and reference-voice caching
behind a compact API. Source builds select the target CUDA architecture, while
binary releases record the architectures they support.

The public Python package and import intentionally remain `fish_s2`:

```python
from fish_s2 import FishS2
```

Applications can connect the SDK to their preferred playback, UI, service and
model-management layers. Model weights and voice references are supplied by the
application.

## Highlights

- persistent native process, loaded model and prepared-reference cache;
- blocking exact-final generation and live PCM16 streaming through one API;
- optimized Q8_0 and BF16 paths without rewriting model weights;
- incremental ModifiedDAC streaming with a KV ring and bounded convolution state;
- independent exact-length final render after live playback;
- optional lossless BF16 P2/P3 storage profiles;
- dependency-free core Python API with explicit advanced native options;
- source-selectable CUDA architecture instead of a hard-coded GPU product.

## Install from source

```powershell
git clone https://github.com/leekery/FishS2RT.git
cd FishS2RT
python -m pip install -e .
fish-s2-build
```

`fish-s2-build` compiles the native runtime when a source checkout does not
already contain one. Automatic architecture selection is the default. An
explicit build for the first validation platform is also possible:

```powershell
fish-s2-build --cuda-arch 120a-real
```

Published binary releases must list their compiled CUDA architectures in
`BUILD_INFO.txt`. Installing the library never downloads model weights; pass the
path to a compatible local Q8_0 or BF16 GGUF yourself.

Release users can install the SDK together with its exactly matched Windows
runtime wheel. Both distributions are produced from the same Git tag; the
public import remains `fish_s2`:

```powershell
python -m pip install "fish_s2[runtime]"
```

## First voice clone

```python
from fish_s2 import FishS2

with FishS2(r"C:\models\fish-audio-s2-pro-q8_0.gguf") as tts:
    voice = tts.prepare_voice(
        r"C:\voices\reference.wav",
        "Exact transcript of everything spoken in the reference.",
    )
    audio = tts.generate("The first render is ready.", voice=voice)
    audio.save("result.wav")
```

The model stays loaded between calls. Reuse the same `FishS2` instance and
`Voice`; call `unload()` when GPU memory should be released.

## Stream audio as it is generated

```python
from fish_s2 import FishS2, Sampling

with FishS2(r"C:\models\fish-audio-s2-pro-q8_0.gguf") as tts:
    voice = tts.prepare_voice("reference.wav", "Exact reference transcript.")
    sampling = Sampling(stream_chunk_frames=4)

    with tts.stream(
        "Audio can be played before the complete utterance is ready.",
        voice=voice,
        sampling=sampling,
    ) as stream:
        for chunk in stream:
            audio_sink.write(chunk.pcm)  # PCM16 little-endian

        final = stream.result()
        final.save("exact-final.wav")
```

The SDK deliberately does not impose an audio-output dependency. Feed
`chunk.pcm` to `sounddevice`, a socket, a browser bridge or another sink. Live
chunks prioritize immediate playback and are not bit-identical to the
independent exact-final render; save `stream.result()` when the canonical final
file matters.

## Precision and runtime profiles

| Profile | Meaning | Recommended use |
|---|---|---|
| Q8_0 | Quantized model weights | Lower VRAM use and the practical 12 GB profile |
| BF16/raw | Original BF16 values in the GGUF path | Highest available weight precision |
| BF16/P2 | Exact two-plane BF16 storage for selected Fast-AR weights | Advanced opt-in |
| BF16/P3 | Exact three-plane/HMC-oriented storage with P2 fallback | Advanced opt-in |

P2 and P3 are lossless storage representations, not additional weight
quantization. Model precision, sampling and text segmentation remain explicit
application choices.

## Measured performance

These measurements apply to the retained workloads and the first validation
system only; they are not universal performance promises.

| RTX 5070 12 GB workload | Observed result versus pinned baseline |
|---|---:|
| Q8 ordinary text, matched final | 24.12% lower mean latency |
| Q8 voice clone, matched final | 21.46% lower mean latency |
| BF16 fixed workload, matched final | 16.52% lower mean latency |
| Q8 live streaming | 1.745x real-time throughput |
| BF16/P3 live streaming | 1.304x real-time throughput |

Matched final WAV files were SHA-256-identical within each controlled
comparison. Live-versus-final similarity measured about 46.6 dB SNR for Q8 and
47.1 dB for BF16/P3. These figures describe the retained workloads on one test
system and are not universal performance guarantees.

## Documentation

- [Documentation index and capability matrix](docs/README.md)
- [Installation, wheels and native builds](docs/installation.md)
- [Guided Python SDK tutorial](docs/python-sdk.md)
- [Complete API reference](docs/api-reference.md)
- [Configuration and precision profiles](docs/configuration.md)
- [Streaming, playback, sockets and async integration](docs/streaming.md)
- [Integration recipes](docs/recipes.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Architecture and extension points](docs/architecture.md)
- [Models and licenses](docs/models-and-licenses.md)
- [Development and verification](docs/development.md)
- [Publication and release checklist](docs/publishing.md)
- [Runnable examples](examples/README.md)

## Licensing and upstream

FishS2RT is derived from and built on
[audio.cpp](https://github.com/0xShug0/audio.cpp). Repository code is covered by
the included Apache-2.0 license and third-party notices. Fish Audio model weights
have a separate [research/non-commercial license](licenses/FISH_AUDIO_RESEARCH_LICENSE.md)
and are not redistributed here.

The exact imported commit and update policy are recorded in
[UPSTREAM.md](UPSTREAM.md). This is source attribution, not publication of the
private experiments used while developing FishS2RT.
