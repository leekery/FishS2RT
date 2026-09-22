# Troubleshooting the Python SDK

## First diagnostic snapshot

Run these from the same Python environment and working directory as the
application:

```powershell
python -c "import sys, fish_s2; print(sys.executable); print(fish_s2.__version__); print(fish_s2.__file__)"
python -c "from fish_s2._runtime import resolve_runtime; print(resolve_runtime(None))"
nvidia-smi
```

Then check the native worker itself:

```powershell
& "PATH_FROM_PREVIOUS_COMMAND" --version
& "PATH_FROM_PREVIOUS_COMMAND" --list-devices
```

Do not publish model paths, private transcripts, references, generated WAVs or
raw logs when asking for help. Runtime version, binary hash, GPU/driver, error
type and bounded stderr tail are usually sufficient for the first diagnosis.

## `ModuleNotFoundError: No module named 'fish_s2'`

The application is using a different interpreter from the one used to install.

```powershell
python -m pip install -e .
python -c "import fish_s2; print(fish_s2.__file__)"
```

Prefer `python -m pip` over a bare `pip` so the interpreter is explicit.

## `RuntimeNotFoundError`

No native worker was found. Choose one:

```python
tts = FishS2("model.gguf", runtime=r"D:\runtime\audiocpp_cli.exe")
```

```powershell
$env:FISH_S2_RUNTIME = "D:\runtime\audiocpp_cli.exe"
```

Or build from a source checkout:

```powershell
fish-s2-build
```

Runtime resolution paths are listed in the exception text.

## Worker fails before `--version`

The executable may be missing a dynamic dependency. The tested worker can depend
on the CUDA runtime/CUBLAS and Microsoft Visual C++ runtime. Use the SDK release
built with `-IncludeCudaRuntime`, or install compatible runtime components. A
source build normally finds DLLs from its CUDA Toolkit environment.

Do not copy arbitrary DLLs from untrusted sites. Keep runtime DLLs and their
redistribution licenses together.

## `ModelLoadError`

Common causes:

- path is not the intended Fish S2 GGUF;
- truncated/corrupt file;
- wrong model conversion or unsupported metadata;
- CUDA driver/runtime mismatch;
- insufficient free VRAM;
- binary built without the Fish model family or CUDA backend.

Verify the documented byte size and SHA-256:

```powershell
Get-Item "C:\models\model.gguf" | Select-Object Length
Get-FileHash "C:\models\model.gguf" -Algorithm SHA256
```

Check [models-and-licenses.md](models-and-licenses.md) for expected values. Test
Q8 first when other GPU applications are using memory.

## CUDA out of memory

```powershell
nvidia-smi
```

Actions, in order:

1. unload other local models and GPU-heavy applications;
2. confirm an old SDK worker is not still alive;
3. call `tts.unload()` before retrying another profile;
4. use Q8 instead of BF16;
5. reduce the number of simultaneous `FishS2` instances;
6. test `RuntimeConfig(mem_saver=True)` and remeasure latency/quality.

Model file size is not a complete VRAM prediction. Record actual load and peak
memory on the target machine.

## `InvalidRequestError: v0.1 accepts WAV references only`

Convert explicitly:

```powershell
ffmpeg -i reference.mp3 -ac 1 -ar 44100 -c:a pcm_s16le reference.wav
```

Then pass the converted file and an exact transcript. The SDK does not perform
hidden conversion because changed reference PCM can change the voice result.

## Reference voice is wrong or weak

Check:

- transcript matches every spoken word and language;
- recording contains one speaker, little noise/reverb/music and no long silence;
- no clipping;
- reference is not accidentally a generated/processed file of the wrong voice;
- the same WAV/transcript are used in comparisons;
- `Voice` belongs to the current loaded `FishS2` instance.

`prepare_voice()` strips only surrounding transcript whitespace. It does not
correct wording or alignment.

## `voice belongs to a different or unloaded FishS2 instance`

`Voice` is an owned live-session handle, not a portable embedding. Prepare it
again:

```python
tts.load()
voice = tts.prepare_voice("reference.wav", "Exact transcript.")
```

This commonly follows `unload()`, a hard stream cancellation, a timeout or a
worker crash.

## `BusyError`

Only one request may use an instance. Serialize calls through an application
queue or create a bounded instance pool if VRAM permits. Do not blindly retry in
a tight loop.

```python
try:
    return tts.generate(text)
except BusyError:
    return queue_or_reject(text)
```

## No live chunks until late

Check separately:

- model/reference were prepared before timing;
- iteration starts immediately after `stream()`;
- `stream_chunk_frames` is not excessively large;
- the consumer is not doing slow work before calling `next()` again;
- the application is not adding a large playback/network prebuffer;
- GPU is not occupied by another workload;
- text/first semantic tokens are not unusually slow.

```python
from time import perf_counter

started = perf_counter()
with tts.stream(text, sampling=Sampling(stream_chunk_frames=8)) as stream:
    first = next(stream)
    print("SDK first chunk seconds:", perf_counter() - started)
    for chunk in stream:
        pass
```

Do not infer SDK delay from the moment headphones start if the application waits
for two chunks or the device adds latency.

## Beginning of playback is clipped

The SDK yields chunk zero; clipping is normally in the application/device start
path. Keep the first chunks and start a persistent output stream before writing.
The [streaming guide](streaming.md) includes a two-chunk safety prebuffer.

Do not launch a new external player for every chunk. Do not discard chunk zero.

## Live playback sounds slightly different from saved WAV

Expected limitation: live events use fixed-shape stateful delta decoding and
are not promised to be sample-identical to the independent exact-length final
native render. Save `stream.result()` and play chunks only for live monitoring.

If the final WAV itself changes between supposedly matched runs, verify seed,
model/runtime hashes, sampler, reference, segmentation and P2/P3 profile.

## Numbers are mispronounced

Try a controlled comparison, without rewriting the input:

```python
sampling = Sampling.numbers_safe(seed=42)
audio = tts.generate(
    original_text,
    sampling=sampling,
    segmentation="sentences",
)
```

Then compare Q8 and BF16 with identical text/reference/settings. Lower entropy
and sentence isolation can help some cases but are not guaranteed. Record the
exact failing token, its sentence position and seed for research.

## Output ends early

Increase `max_tokens` for the request:

```python
sampling = Sampling(max_tokens=1536)
```

Also check whether native chunking or sentence isolation produced the expected
number of segments and whether the application closed the stream early.

## Output has excessive pauses

- set `pause_ms=0`;
- use `segmentation="native"` if independent sentence requests sound too
  separated;
- inspect punctuation and explicit Fish pause tags in the original text;
- ensure the playback sink does not underrun between chunks.

The SDK does not insert silence except explicit `pause_ms` between final
sentence-isolated parts.

## `GenerationError`

The native worker rejected or failed a command but normally remains reusable.
Inspect the exception and bounded native stderr. Check option ranges and remove
unrecognized `request_options`/`session_options` first.

```python
try:
    audio = tts.generate(text)
except GenerationError as exc:
    print(exc)
    print("still loaded:", tts.loaded)
```

## `WorkerCrashedError`

The process exited or its pipe closed. The message includes a bounded stderr
tail. During generation the SDK cancels/cleans the dead worker; reload and
prepare the voice only after diagnosing the cause.

Common causes include CUDA failure, missing DLL, out of memory, incompatible
binary/model and external process termination.

## Timeout

`load_timeout` and `command_timeout` raise Python `TimeoutError`. Generation
timeouts hard-stop the worker because v0.1 has no safe cooperative native
cancellation.

```python
config = RuntimeConfig(load_timeout=420, command_timeout=180)
```

Do not automatically retry without a bound: a pathological input can otherwise
create a restart loop.

## P3 does not activate or is slower

P3 is experimental and hardware/allocation dependent. Confirm:

- BF16 model, not Q8;
- `RuntimeConfig(bf16_storage="p3")` was set before load;
- optimized binary contains P2/P3 support;
- no stale process from a previous profile remains;
- matched warm measurements use identical requests and counterbalanced order.

P3 may fall back to P2 when its allocation path is unavailable. Do not infer
selection only from a requested Python value; research validation should inspect
native debug traces and exact outputs.

## Temporary files remain after an abnormal exit

Normal requests clean `fish-s2-sdk-*` directories. A killed Python process,
power loss or OS crash can leave them under the system temp directory. Close all
SDK processes, verify the target path is a specific stale `fish-s2-sdk-*`
directory, and remove it using normal OS cleanup policy.

Never recursively delete a broad temp, user-profile or workspace directory.

## Paths with spaces or Cyrillic fail

Use `Path`/normal strings and avoid shell-constructed commands:

```python
from pathlib import Path

model = Path(r"D:\Модели Fish\fish s2 q8.gguf")
reference = Path(r"D:\Голоса\референс.wav")
tts = FishS2(model)
voice = tts.prepare_voice(reference, "Точная расшифровка.")
```

The SDK launches without `shell=True` and uses UTF-8 control files. Failures in
a custom wrapper/runtime may still have their own encoding limitations.

## Performance is below realtime

Measure warm generation separately from model load and reference preparation:

```python
from time import perf_counter

tts.load()
tts.prepare_voice("reference.wav", "Exact transcript.")
started = perf_counter()
audio = tts.generate(text, sampling=Sampling(seed=42))
wall = perf_counter() - started
print("wall", wall, "audio", audio.duration, "RTF", wall / audio.duration)
```

Then record GPU utilization/power state, background workloads, model precision,
binary hash, text length, segmentation, sampler and cold/warm status. Do not add
speed percentages from unrelated baselines.
