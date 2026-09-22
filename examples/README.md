# Runnable Python examples

These scripts are application examples for the `fish_s2` package. They require
user-supplied model weights, a WAV reference and its exact transcript. They do
not download weights or contain private audio.

Run any script with `--help` before inference:

```powershell
python examples/python_basic.py --help
```

## One final WAV

```powershell
python examples/python_basic.py `
  --model D:\models\fish-audio-s2-pro-q8_0.gguf `
  --reference D:\voices\reference.wav `
  --transcript-file D:\voices\reference.txt `
  --text "Видеокарта обработала 74 кадра в секунду." `
  --numbers-safe `
  --segmentation sentences `
  --pause-ms 80 `
  --output outputs\basic.wav
```

`python_basic.py` shows explicit paths, voice preparation, optional numeric
sampling, final WAV saving and native metrics.

## Hear chunks during inference

Install the optional playback extra:

```powershell
python -m pip install -e ".[audio]"
```

Then run:

```powershell
python examples/python_streaming.py `
  --model D:\models\fish-audio-s2-pro-q8_0.gguf `
  --reference D:\voices\reference.wav `
  --transcript-file D:\voices\reference.txt `
  --text "Первый звук поступает до завершения всей генерации." `
  --prebuffer-chunks 2 `
  --output outputs\stream-final.wav
```

The console prints every chunk as it is created. Omit `--output` to avoid saving
test generations. Use `--no-playback` to inspect streaming events without an
audio device or `sounddevice`. A two-chunk prebuffer avoids first-block clipping
on affected Windows devices; set `--prebuffer-chunks 1` for minimum application
latency.

## Keep one model loaded for many requests

```powershell
python examples/python_reuse_session.py `
  --model D:\models\fish-audio-s2-pro-q8_0.gguf `
  --reference D:\voices\reference.wav `
  --transcript-file D:\voices\reference.txt `
  --text "Первый тест." `
  --text "Второй тест." `
  --text "Третий тест." `
  --output-dir outputs
```

`python_reuse_session.py` prints the worker PID for every request and fails if it
changes unexpectedly. Each result receives a random suffix; the model is
explicitly unloaded in `finally`.

## Advanced configuration

```powershell
python examples/python_advanced.py `
  --model D:\models\fish-audio-s2-pro-bf16.gguf `
  --reference D:\voices\reference.wav `
  --transcript-file D:\voices\reference.txt `
  --text "В первой сцене 61 FPS. Во второй 74 FPS." `
  --bf16-storage raw `
  --temperature 0.65 `
  --top-p 0.78 `
  --top-k 20 `
  --seed 42 `
  --sentence-isolation `
  --pause-ms 80 `
  --output outputs\advanced.wav
```

Use `raw` for the stable BF16 path. `p2` and `p3` are experimental research
profiles and should only be selected for matched measurements.

## Async application bridge

```powershell
python examples/python_async.py `
  --model D:\models\fish-audio-s2-pro-q8_0.gguf `
  --reference D:\voices\reference.wav `
  --transcript-file D:\voices\reference.txt `
  --text "Асинхронное приложение получает чанки по одному." `
  --output outputs\async-final.wav
```

The script advances the synchronous iterator one step at a time with
`asyncio.to_thread`. Replace `consume_chunk()` with an awaited WebSocket or
network send. This does not make one `FishS2` instance concurrent.

## Fixed A/B text fixtures

- `en_quick_test.txt`: short English sanity phrases;
- `ru_cpu_review.txt`: Russian processor-review language;
- `ru_gpu_numbers.txt`: Russian GPU and numeric stress text.

For a valid A/B comparison, do not edit fixture text between variants. Keep the
reference WAV and transcript, seed, sampling, segmentation, runtime and model
file fixed. Randomize listening order and record hashes plus timing metadata.

More examples are in the [recipe book](../docs/recipes.md), while exact
signatures and constraints are in the [API reference](../docs/api-reference.md).
