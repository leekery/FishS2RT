# Integration recipes

All examples assume an installed package and user-supplied model/reference files.
Replace paths and transcripts with real values.

## Smallest voice clone

```python
from fish_s2 import FishS2

with FishS2(r"C:\models\fish-audio-s2-pro-q8_0.gguf") as tts:
    tts.prepare_voice("reference.wav", "Exact reference transcript.")
    tts.generate("Hello from the cloned voice.").save("hello.wav")
```

`prepare_voice()` makes the voice default, so `voice=` is optional afterward.

## Explicit voice without changing the default

```python
voice = tts.prepare_voice(
    "speaker-b.wav",
    "Speaker B transcript.",
    make_default=False,
)
audio = tts.generate("Only this request uses speaker B.", voice=voice)
```

## Reuse one loaded model

```python
from fish_s2 import FishS2

tts = FishS2("model.gguf").load()
try:
    voice = tts.prepare_voice("reference.wav", "Exact transcript.")
    pid = tts.pid

    for index, text in enumerate(("First.", "Second.", "Third."), 1):
        tts.generate_to_file(text, f"output-{index}.wav", voice=voice)
        assert tts.pid == pid
finally:
    tts.unload()
```

This is the intended high-throughput usage: cold model load and reference
encoding are outside repeated generation calls.

## Return bytes to a web framework

```python
audio = tts.generate("WAV response body.")
body = audio.to_wav_bytes()

# Framework-neutral response fields:
status = 200
headers = {
    "Content-Type": "audio/wav",
    "Content-Length": str(len(body)),
}
```

Do not create a new `FishS2` inside every HTTP handler. Own one long-lived
instance in application startup/shutdown and serialize requests or maintain a
bounded pool whose total models fit VRAM.

## Save final output in a generated directory

```python
from pathlib import Path
from secrets import token_hex

name = f"fish-{token_hex(6)}.wav"
path = tts.generate_to_file(
    "Random output filename.",
    Path("outputs") / name,
)
print(path)  # absolute Path
```

## In-memory PCM processing

```python
import array

audio = tts.generate("Analyze PCM.")
samples = array.array("h")
samples.frombytes(audio.pcm)
if samples.itemsize != 2:
    raise RuntimeError("expected 16-bit host array")
peak = max(abs(value) for value in samples)
print(audio.sample_rate, audio.channels, audio.frames, peak)
```

On the supported little-endian Windows target, `array('h')` matches the SDK's
signed PCM16 little-endian representation.

## Immediate streaming playback

```python
import sounddevice as sd

with tts.stream("Play while generating.") as stream:
    output = None
    try:
        for chunk in stream:
            if output is None:
                output = sd.RawOutputStream(
                    samplerate=chunk.sample_rate,
                    channels=chunk.channels,
                    dtype="int16",
                )
                output.start()
            output.write(chunk.pcm)
        final = stream.result()
    finally:
        if output is not None:
            output.stop()
            output.close()
```

Install with `python -m pip install -e ".[audio]"` for this example.

## Stream without saving

```python
with tts.stream("Playback-only test.") as stream:
    for chunk in stream:
        sink.write(chunk.pcm)
    # Exhaustion is enough. Do not call save().
    final_duration = stream.result().duration
```

The SDK uses and cleans its own temporary request directory. It does not place a
final WAV in the user's output tree unless `save()`/`generate_to_file()` is used.

## Long text with native continuation

```python
from fish_s2 import Sampling

audio = tts.generate(
    article,
    sampling=Sampling(
        seed=1234,
        text_chunk_size=240,
        text_chunk_mode="tag_aware",
        max_tokens=1024,
    ),
    segmentation="native",
)
```

Use this when continuity across native chunks matters more than isolating each
sentence.

## Sentence isolation for technical numbers

```python
from fish_s2 import Sampling

text = (
    "В первой сцене видеокарта показала 61 FPS. "
    "Во второй результат вырос до 74 FPS. "
    "Температура составила 67 градусов."
)

audio = tts.generate(
    text,
    sampling=Sampling.numbers_safe(seed=42),
    segmentation="sentences",
    pause_ms=80,
)
```

The SDK sends the original digits. It does not transliterate `61` into words.

## Fish inline tags

```python
text = "<|happy|>Отличный результат! <|short pause|>Но проверим ещё раз."
audio = tts.generate(
    text,
    segmentation="sentences",
)
```

The exact splitter skips boundaries while inside `<|...|>` and never removes or
rewrites a tag. Whether a particular tag has the intended acoustic effect is a
model behavior, not an SDK guarantee.

## BF16 stable and experimental P3

```python
from fish_s2 import FishS2, RuntimeConfig

stable = FishS2(
    "fish-audio-s2-pro-bf16.gguf",
    config=RuntimeConfig(bf16_storage="raw"),
)

experimental = FishS2(
    "fish-audio-s2-pro-bf16.gguf",
    config=RuntimeConfig(bf16_storage="p3"),
)
```

Do not load both simultaneously on a 12 GB GPU. Run one, unload it, verify VRAM
release, then run the other with identical requests for a matched comparison.

## Select another CUDA device

```python
from fish_s2 import RuntimeConfig

tts = FishS2(
    "model.gguf",
    config=RuntimeConfig(backend="cuda", device=1),
)
```

Device numbering follows the native backend. Validate with the native
`audiocpp_cli --list-devices` if application enumeration matters.

## Use a custom native runtime

```python
tts = FishS2(
    "model.gguf",
    runtime=r"D:\runtimes\audiocpp_cli.exe",
)
```

Or for the whole process:

```powershell
$env:FISH_S2_RUNTIME = "D:\runtimes\audiocpp_cli.exe"
python app.py
```

An explicit constructor value wins because environment discovery is skipped.

## Inspect metrics

```python
audio = tts.generate("Metrics example.")
for name, value in sorted(audio.metrics.items()):
    print(name, value)

wall_ms = float(audio.metrics["part_0000.wall_ms"])
rtf = float(audio.metrics["part_0000.rtf"])
print(f"request wall={wall_ms:.1f} ms, RTF={rtf:.3f}")
```

Metric keys are diagnostic and may grow. Guard optional keys in production code
instead of assuming every native version emits exactly the same set.

## Handle a busy model

```python
from fish_s2 import BusyError

try:
    audio = tts.generate("Second request")
except BusyError:
    # Return 429/queue/retry according to application policy.
    pass
```

The SDK refuses implicit unbounded queues. A service should choose its own bound,
priority and cancellation policy.

## Handle failures and reload after a hard cancel

```python
from fish_s2 import FishS2Error, WorkerCrashedError

try:
    with tts.stream(long_text) as stream:
        for chunk in stream:
            if should_cancel():
                stream.close()
                break
except WorkerCrashedError as exc:
    log_error(exc)
except FishS2Error as exc:
    log_error(exc)

if not tts.loaded:
    tts.load()
    voice = tts.prepare_voice("reference.wav", "Exact transcript.")
```

## Async bridge with a callback

```python
import asyncio

_END = object()


async def synthesize_async(tts, text, on_chunk):
    with tts.stream(text) as stream:
        while True:
            chunk = await asyncio.to_thread(next, stream, _END)
            if chunk is _END:
                break
            await on_chunk(chunk)
        return stream.result()
```

Only one `next()` is outstanding at a time. The instance remains single-request.

## Release VRAM before another GPU application

```python
tts.unload()
assert not tts.loaded
```

Dropping a Python variable and waiting for garbage collection is not the public
lifecycle contract. Call `unload()`/`close()` or use a context manager.

## Explicit MP3/FLAC reference conversion

Core v0.1 accepts WAV. Convert outside the SDK so preprocessing is visible and
reproducible:

```powershell
ffmpeg -i reference.mp3 -ac 1 -ar 44100 -c:a pcm_s16le reference.wav
```

Record the converted WAV hash for matched experiments. Different decoders or
conversion settings can change reference PCM and therefore generated output.

## Advanced native options

```python
from fish_s2 import RuntimeConfig, Sampling

config = RuntimeConfig(
    session_options={
        "fish_audio.weight_type": "native",
    }
)
sampling = Sampling(
    request_options={
        "future.namespaced.option": "value",
    }
)
tts = FishS2("model.gguf", config=config)
audio = tts.generate("Advanced request.", sampling=sampling)
```

Use namespaced advanced options only when the matching native binary documents
them. Python does not silently ignore a native rejection.
