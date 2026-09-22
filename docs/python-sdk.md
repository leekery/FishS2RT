# Python SDK guide

This guide takes an application from installation to a persistent, streaming
Fish Audio S2 Pro integration. It documents the supported v0.1 behavior rather
than the much larger native `audio.cpp` CLI surface.

For exact signatures and every field, use the [API reference](api-reference.md).
For copy-and-adapt integrations, use [recipes](recipes.md).

## What the package provides

The `fish_s2` package is a small Python control layer around an optimized native
worker. One `FishS2` object owns one worker process, one loaded GGUF model and its
reference cache.

It provides:

- Q8_0 and BF16 GGUF loading through the same API;
- persistent model and reference-cache lifetime across requests;
- voice cloning from a WAV reference plus its exact transcript;
- blocking final generation and live PCM16 chunk iteration;
- explicit sampling, sentence isolation, BF16 storage and native escape hatches;
- dependency-free WAV writing and in-memory PCM access;
- deterministic cleanup through `unload()` or a context manager.

It deliberately does not provide a GUI, a model downloader, an HTTP server,
automatic MP3/FLAC decoding or a mandatory audio playback backend. Those remain
application choices.

## Five-minute setup

Install the source checkout:

```powershell
git clone https://github.com/leekery/FishS2RT.git
cd FishS2RT
python -m pip install -e .
```

If the checkout does not contain a packaged native worker, build it once:

```powershell
fish-s2-build
```

The build command uses the Windows inbox `powershell.exe`; PowerShell 7 is not
required. A published Windows wheel or binary SDK release can already contain
the worker and its redistributable DLLs. See [installation](installation.md) for
all supported installation layouts and build prerequisites.

The package never downloads model weights. Supply a local Q8_0 or BF16 GGUF
whose use is permitted by its own license.

## First voice clone

```python
from fish_s2 import FishS2

with FishS2(r"C:\models\fish-audio-s2-pro-q8_0.gguf") as tts:
    tts.prepare_voice(
        r"C:\voices\reference.wav",
        "Exact transcript of everything spoken in the reference.",
    )
    audio = tts.generate("The first render is ready.")
    path = audio.save("result.wav")
    print(path, audio.duration, audio.sample_rate, audio.channels)
```

The context manager calls `load()` on entry and `unload()` on exit. The reference
becomes the default voice because `make_default=True` is the default.

Reference requirements in v0.1:

- the input must be a readable WAV file;
- the transcript must be non-empty and match the spoken reference exactly;
- preprocessing is not hidden: convert MP3/FLAC before calling the SDK;
- the native codec performs its own model-specific mono/resampling preparation.

## Explicit lifecycle and reuse

Use an explicit lifetime when integrating with application startup and shutdown:

```python
from fish_s2 import FishS2

tts = FishS2("model.gguf")
tts.load()
try:
    voice = tts.prepare_voice("reference.wav", "Exact transcript.")
    worker_pid = tts.pid

    first = tts.generate("First request.", voice=voice)
    second = tts.generate("Second request.", voice=voice)

    assert tts.loaded
    assert tts.pid == worker_pid
finally:
    tts.unload()
```

`load()` is idempotent. Repeated calls do not create another process while the
worker is alive. `unload()` is also safe to call repeatedly. Unloading clears the
default voice and invalidates every `Voice` created by that model instance.

Do not construct a new `FishS2` for every phrase: doing so pays model-load cost
again and loses the prepared reference cache.

## Selecting Q8_0 or BF16

Precision is determined by the model file passed to the constructor:

```python
q8 = FishS2(r"C:\models\fish-audio-s2-pro-q8_0.gguf")
bf16 = FishS2(r"C:\models\fish-audio-s2-pro-bf16.gguf")
```

The SDK does not quantize, rewrite or convert weights. Q8_0 is the practical
12 GB target used by the local RTX 5070 smoke test. BF16 uses substantially more
VRAM and must be measured on the actual machine. Do not load both at once on a
12 GB card.

The stable BF16 profile is the default:

```python
from fish_s2 import RuntimeConfig

tts = FishS2(
    "fish-audio-s2-pro-bf16.gguf",
    config=RuntimeConfig(bf16_storage="raw"),
)
```

`bf16_storage="p2"` and `"p3"` are experimental exact-storage paths, not
automatic recommendations. See [configuration](configuration.md).

## Multiple voices

`prepare_voice()` returns an immutable handle owned by that loaded instance:

```python
voice_a = tts.prepare_voice(
    "speaker-a.wav",
    "Speaker A reference transcript.",
)
voice_b = tts.prepare_voice(
    "speaker-b.wav",
    "Speaker B reference transcript.",
    make_default=False,
)

default_audio = tts.generate("Uses speaker A.")
explicit_audio = tts.generate("Uses speaker B.", voice=voice_b)
```

The public API exposes one reference per `Voice`. A handle cannot be used with a
different `FishS2` object or after its owner is unloaded. `reference_cache_slots`
controls native cache capacity; it does not turn one handle into a multi-speaker
prompt.

## Final generation

Use `generate()` for an in-memory result:

```python
audio = tts.generate("Return an Audio object.")
print(audio.frames, audio.duration, audio.seed)
wav_bytes = audio.to_wav_bytes()
saved = audio.save("outputs/render.wav")
```

Use `generate_to_file()` for the shortest save path:

```python
saved = tts.generate_to_file(
    "Write the final native render.",
    "outputs/final.wav",
)
```

Both paths run the same inference. `Audio.save()` creates missing parent
directories and returns an absolute `Path`.

## Live streaming

`stream()` returns a synchronous iterator. Generation begins when the iterator
is first advanced or when `result()` drains it:

```python
with tts.stream("Audio arrives before the full request completes.") as stream:
    for chunk in stream:
        print(chunk.sequence, chunk.segment, chunk.frames, chunk.duration)
        audio_sink.write(chunk.pcm)
    final = stream.result()
```

Each `AudioChunk.pcm` is immediately playable interleaved signed PCM16
little-endian. Read `sample_rate` and `channels` from the chunk rather than
hard-coding them.

Live chunks use a stateful native delta decoder. Concatenating them is not
guaranteed to reproduce the independent exact-length final native render
sample-for-sample because their CUDA graph shapes differ. Play or forward
chunks live, but use `stream.result()` for archival WAV, downloads and quality
comparisons. Complete playback, prebuffer, socket and async examples are in
[streaming](streaming.md).

## Sampling and reproducibility

```python
from fish_s2 import Sampling

sampling = Sampling(
    seed=42,
    temperature=0.8,
    top_p=0.8,
    top_k=30,
    max_tokens=1024,
    text_chunk_size=200,
    text_chunk_mode="tag_aware",
    stream_chunk_frames=8,
)

audio = tts.generate("A reproducible request.", sampling=sampling)
```

A fixed seed is necessary for matched comparisons, but runtime build, model,
reference WAV, transcript, text, precision, options and hardware path must also
remain fixed. In sentence mode the SDK derives segment seeds as `seed + index`
modulo 2^32.

## Technical numbers

The lower-entropy preset developed during the local numeric-speech investigation
is explicit:

```python
from fish_s2 import Sampling

text = "The GPU produced 61 FPS. The second run produced 74 FPS."
audio = tts.generate(
    text,
    sampling=Sampling.numbers_safe(seed=42),
    segmentation="sentences",
    pause_ms=80,
)
```

`numbers_safe()` sets temperature 0.65, top-p 0.78 and top-k 20. It does not
rewrite `61` or `74`, change the weights or guarantee correct pronunciation.
Sentence isolation contains many failures to one independently generated
sentence, but it also changes conditioning and random-seed boundaries. Keep
`segmentation="native"` when continuity matters more.

## Long text and Fish tags

```python
sampling = Sampling(
    seed=1234,
    text_chunk_size=240,
    text_chunk_mode="tag_aware",
    max_tokens=1024,
)

audio = tts.generate(
    "<|happy|>The benchmark is complete. <|short pause|>Now inspect the result.",
    sampling=sampling,
    segmentation="native",
)
```

The SDK passes text and Fish tags through without replacing them. In sentence
mode its exact splitter avoids splitting inside `<|...|>` tags. Tag availability
and acoustic behavior belong to the model, not to the Python wrapper.

## Configuration scope

Session settings are fixed when the worker loads:

```python
from fish_s2 import RuntimeConfig

config = RuntimeConfig(
    backend="cuda",
    device=0,
    threads=None,
    reference_cache_slots=1,
    mem_saver=False,
    bf16_storage="raw",
    load_timeout=300.0,
    command_timeout=None,
)

tts = FishS2("model.gguf", config=config)
```

Request settings belong to `Sampling` and may change for every generation.
Advanced `session_options` and `request_options` are intentionally exposed, but
unknown keys are validated only by the native runtime and are not stable across
unrelated builds.

## Metrics

Final native metrics are attached to `Audio.metrics`:

```python
audio = tts.generate("Measure this request.")
for name, value in sorted(audio.metrics.items()):
    print(name, value)

wall_ms = audio.metrics.get("part_0000.wall_ms")
rtf = audio.metrics.get("part_0000.rtf")
print("wall_ms=", wall_ms, "rtf=", rtf)
```

Treat metric names as diagnostics. Guard optional keys because future native
workers may add or rename metrics. For performance reporting, separate cold
load, reference preparation, first SDK chunk, first audible sample, total wall
time, audio duration, RTF and peak/retained VRAM.

## Concurrency and cancellation

One instance accepts one active prepare/generate request. A concurrent request
raises `BusyError`; the SDK never creates an implicit unbounded queue.

Separate instances are possible, but each owns another loaded model and consumes
additional VRAM. A service should serialize work or use a bounded instance pool
whose memory use has been measured.

Closing a stream after it has started is a hard cancellation in v0.1. The owned
worker is terminated, GPU allocations are released, the model becomes unloaded
and old voice handles become invalid. Reload and prepare the voice again before
the next request. Cooperative cancellation that retains CUDA state is a future
native extension.

## Exceptions

Catch `FishS2Error` for all SDK-defined failures, or a specific subclass:

```python
from fish_s2 import FishS2Error, InvalidRequestError, WorkerCrashedError

try:
    audio = tts.generate("Request text.")
except InvalidRequestError as exc:
    print("Invalid input:", exc)
except WorkerCrashedError as exc:
    print("Worker stopped:", exc)
except FishS2Error as exc:
    print("Fish S2 failure:", exc)
```

Timeouts use Python's built-in `TimeoutError`. A generation timeout or worker
crash leaves the instance unloaded because preserving native CUDA state cannot
be guaranteed.

## Runtime discovery

The native worker is resolved in this order:

1. explicit `runtime=` constructor argument;
2. `FISH_S2_RUNTIME` environment variable;
3. packaged `fish_s2/_native/audiocpp_cli.exe`;
4. known source-checkout build directories.

```python
tts = FishS2(
    "model.gguf",
    runtime=r"D:\runtimes\audiocpp_cli.exe",
)
```

The runtime is an isolated subprocess. The SDK hides its console window on
Windows, reads stdout/stderr continuously, uses a bounded event queue and keeps
the parent process environment unchanged.

## Production ownership checklist

An embedding application still owns:

- acquisition, verification and licensing of model weights;
- reference consent, storage, preprocessing and transcript correctness;
- request authentication, rate limits, queues and admission control;
- playback device, volume, buffering and reconnect policy;
- cancellation UX and retry policy;
- output retention, encryption and deletion;
- observability and hardware-specific performance acceptance.

The SDK owns model-process lifetime, request serialization, exact PCM/WAV value
objects, temporary request cleanup, native error translation and explicit
configuration boundaries.

## Complete documentation map

- [Documentation index and capability matrix](README.md)
- [Installation and native build](installation.md)
- [Complete API reference](api-reference.md)
- [Configuration and precision profiles](configuration.md)
- [Streaming, playback, sockets and async](streaming.md)
- [Integration recipes](recipes.md)
- [Troubleshooting](troubleshooting.md)
- [Architecture and extension points](architecture.md)
- [Models, hashes and licenses](models-and-licenses.md)
- [Development and verification](development.md)
- [Publication guide](publishing.md)
- [Release checklist](release-checklist.md)
