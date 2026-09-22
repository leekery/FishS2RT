# Configuration and profiles

## Choosing Q8_0 or BF16

Precision is selected only by the GGUF passed to `FishS2`:

```python
q8 = FishS2(r"C:\models\fish-audio-s2-pro-q8_0.gguf")
bf16 = FishS2(r"C:\models\fish-audio-s2-pro-bf16.gguf")
```

| Profile | File size | Intended use | Quality statement |
|---|---:|---|---|
| Q8_0 | 6,317,911,232 bytes | Lower-memory tested profile with more VRAM headroom | Quantized weights; not mathematically identical to BF16 |
| BF16 | 10,229,278,080 bytes | Highest available GGUF weight precision | Preserves BF16 weight bits in this conversion; still not a claim of parity with every upstream Python path |

The API and reference workflow are identical. Never compare quality or speed by
changing precision and sampler settings simultaneously.

```python
from fish_s2 import FishS2, Sampling

common = Sampling(seed=42, temperature=0.8, top_p=0.8, top_k=30)
with FishS2("q8.gguf") as q8:
    q8_audio = q8.generate("Matched text.", sampling=common)
with FishS2("bf16.gguf") as bf16:
    bf16_audio = bf16.generate("Matched text.", sampling=common)
```

Model hashes and licensing are in [models-and-licenses.md](models-and-licenses.md).

## Stable optimized runtime

The bundled worker contains native changes below the Python SDK. The stable path
includes the accepted Fish-specific decode/runtime work such as fast-token
pruning, exact BF16 rounding fusions, direct attention/KV storage changes,
asynchronous upload and CUDA GEMV work. Those optimizations are part of the
binary and require no Python flag.

The retained matched experiments compare the optimized binary with a pinned
upstream baseline using the same model, text, seed and sampler. They do not prove
that Q8 and BF16 sound identical to one another.

## BF16 storage profiles

```python
from fish_s2 import FishS2, RuntimeConfig

with FishS2(
    "fish-audio-s2-pro-bf16.gguf",
    config=RuntimeConfig(bf16_storage="p3"),
) as tts:
    audio = tts.generate("BF16 P3 test.")
```

| `bf16_storage` | Behavior | Status |
|---|---|---|
| `raw` | Ordinary BF16 storage | Default and broad fallback |
| `p2` | Two-plane exact BF16 representation for selected Fast-AR weights | Experimental opt-in |
| `p3` | Three-plane/HMC-oriented exact representation with P2 fallback | Experimental opt-in |

P2/P3 are storage representations, not new quantization levels: the validated
path reconstructs the same logical BF16 values. P3 relies on opaque CUDA virtual
memory/HMC behavior and therefore remains opt-in even though the retained RTX
5070 cases matched raw output exactly.

These modes are fixed before model loading. To change one, unload and create a
new `FishS2` instance.

## Recommended profiles

### General Q8 voice cloning

```python
sampling = Sampling(
    seed=None,
    temperature=0.8,
    top_p=0.8,
    top_k=30,
    max_tokens=1024,
    stream_chunk_frames=8,
)
```

### Reproducible comparison

```python
sampling = Sampling(
    seed=42,
    temperature=0.8,
    top_p=0.8,
    top_k=30,
    max_tokens=1024,
)
```

Fix the model hash, reference WAV/transcript, runtime hash, text, segmentation,
sampler and seed. For sentence isolation the SDK increments the seed by sentence
index; record that policy in comparisons.

### Numeric-risk text

```python
sampling = Sampling.numbers_safe(seed=42)
audio = tts.generate(
    "Производительность выросла с 61 до 74 FPS.",
    sampling=sampling,
    segmentation="sentences",
)
```

The preset is temperature `0.65`, top-p `0.78`, top-k `20`. Lower entropy can
reduce some malformed trajectories but may reduce variation or expressiveness.
It does not guarantee correct pronunciation and does not replace digits with
words.

### Low-latency live chunks

```python
sampling = Sampling(stream_chunk_frames=4)
```

Smaller values can deliver events more frequently but increase fixed graph,
callback, protocol and temporary-file overhead. They no longer re-decode the
growing prefix. Larger values reduce event overhead but delay the first
playable block. The balanced default is `8` codec frames (roughly 372 ms of
generated audio per native event, though actual delivery timing also includes
semantic generation and decoding). On the tested RTX 5070/Q8 path, `4` frames
provided roughly 184 ms of audio per event while remaining faster than realtime.

## Text segmentation

### Native

```python
audio = tts.generate(long_text, segmentation="native")
```

One native request is sent. `text_chunk_size` and `text_chunk_mode` control the
framework's long-text path; Fish can carry generated codes between internal
chunks. This is the default because it preserves native conditioning semantics.

### Isolated sentences

```python
audio = tts.generate(
    long_text,
    segmentation="sentences",
    pause_ms=100,
)
```

The SDK identifies clear `.`, `!`, `?` and `…` boundaries outside Fish tags.
Decimal dots such as `61.5` are not boundaries. Every character, including
whitespace, digits and tags, remains in the concatenation of segments.

Each sentence becomes an independent request in the same worker. Native
internal chunking is effectively disabled for that sentence by using a large
character budget. If a base seed is supplied, segment `i` uses
`(seed + i) & 0xffffffff`.

This can isolate a pronunciation failure to one sentence, but it also changes
conditioning and random-number boundaries. It is therefore explicit, not a
hidden quality fix.

`pause_ms` adds PCM silence only between final sentence outputs. It does not
delay, crossfade, normalize or edit live chunks.

## Text chunk modes

| Mode | Use |
|---|---|
| `tag_aware` | Default SDK choice; avoids careless boundaries around inline model tags |
| `default` | Generic framework chunking |
| `japanese` | Japanese-oriented boundary rules |
| `endline` | Prefer line boundaries |

```python
sampling = Sampling(text_chunk_size=320, text_chunk_mode="endline")
```

With `segmentation="sentences"`, the SDK overrides only `text_chunk_size` for
each isolated request; the selected mode is still forwarded.

## Reference cache and memory

```python
config = RuntimeConfig(
    reference_cache_slots=2,
    mem_saver=False,
)
```

- `reference_cache_slots=0`: disables retained reference entries.
- `1`: default for one working voice.
- Larger values: useful when an application alternates exact reference WAV/text
  pairs, at additional memory cost.
- `mem_saver=True`: asks Fish to release cached AR runtime graphs after each
  request. It can reduce retained memory but can hurt repeated-request latency.

`prepare_voice()` uses eager preparation so the first generation does not also
pay the reference encoding cost. A cache key depends on reference content and
conditioning, not merely the filename.

## Timeouts

```python
config = RuntimeConfig(
    load_timeout=420.0,
    command_timeout=120.0,
)
```

- `load_timeout`: covers native process start and model readiness.
- `command_timeout`: covers each prepare/generate command; `None` is unlimited.

A generation timeout cannot safely interrupt the synchronous native request
while preserving its CUDA state. v0.1 therefore hard-stops the worker and marks
the model unloaded. Reload explicitly before retrying.

## Backend, device and threads

```python
config = RuntimeConfig(backend="cuda", device=0, threads=8)
```

This release is tested with `backend="cuda"`, device zero. The underlying
audio.cpp CLI accepts other backend names, but this repository's binary and
performance/quality evidence do not certify them. `threads` changes native host
threading and should be measured rather than assumed faster.

## Advanced options

Common settings are typed. Native escape hatches remain available:

```python
config = RuntimeConfig(
    session_options={
        "fish_audio.weight_type": "native",
        "fish_audio.codec_weight_type": "native",
    }
)

sampling = Sampling(
    request_options={
        "custom.future.option": "value",
    }
)
```

Rules:

- Values must be strings, integers, floats or booleans.
- Unknown options are not validated by Python; native code may reject them.
- `Sampling.stream_chunk_frames` wins over the same request-option key.
- Advanced session options are appended last and may override native map keys;
  avoid duplicating typed settings unless intentionally testing behavior.
- Options are not a compatibility promise across unrelated audio.cpp versions.

## Controlled child environment

The SDK creates a copy of the parent environment, removes known experimental
and diagnostic Fish/BF16 flags, then applies the selected storage profile only
to the child. This prevents a stale shell variable from silently changing
results and avoids mutating the embedding application's global environment.

For research requiring diagnostic disable flags, launch the native CLI/harness
directly or extend `RuntimeConfig` deliberately; ordinary SDK code should not
inherit benchmark toggles.
