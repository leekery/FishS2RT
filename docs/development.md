# Development and verification

## Stable SDK invariants

Changes are not ready when they violate any of these conditions:

1. The core library owns no GUI, HTTP server or playback device.
2. `stream()` yields chunk zero as soon as native audio is ready, in contiguous
   sequence order with explicit PCM format metadata.
3. One persistent worker owns the loaded model and reference cache until
   `unload()` or context-manager exit.
4. Text splitting preserves every input character; numbers and Fish tags are
   never rewritten.
5. Final saved audio comes from an independent exact-length native render, not
   concatenated live chunks.
6. Q8 and BF16 are selected by explicit GGUF path through the same API.
7. Experimental BF16 P3/HMC remains opt-in.
8. One request per instance is enforced; early cancellation leaves the instance
   unloaded rather than claiming an unsafe warm recovery.

## Fast checks without loading a model

```powershell
python -m pip install -e . --no-deps
python -m unittest discover -s tests/python -p "test_*.py" -v
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/check_fish_publication.ps1
```

These test installation, persistent-process ownership, streaming order, exact
text splitting, cancellation, wheel-safe publication boundaries and privacy.
They intentionally do not load weights or reserve VRAM.

The native `fish_audio.codec_*_decode_total_ms` counters are diagnostic sums of
measured decode calls, not pure backend-kernel time. Initial streaming-graph
construction is outside the stream-decode sum; the deferred-final measurement
also includes graph release and appending the decoded buffer. Use request wall
time and RTF for end-to-end performance claims.

## Manual release checks

Follow [release-checklist.md](release-checklist.md) in a fresh extracted folder.
GPU integration tests are manual because hosted CI has no matching RTX 5070 and
cannot accept the model license on a user's behalf.

## Code map

- Fish model runtime: `src/models/fish_audio`, `include/engine/models/fish_audio`.
- Decoder/KV path: `src/framework/modules/transformers/qwen_decoder.cpp`.
- CUDA exact BF16 storage: `external/ggml/src/ggml-cuda/bf16-p2.*` and call sites.
- Persistent CLI protocol: `app/cli/batch.*`, `app/cli/main.cpp`.
- Python SDK: `src/fish_s2`.
- Model-free SDK tests: `tests/python`; manual GPU smoke: `tests/manual`.

## Contributions

Keep stable user-facing work separate from experimental kernels. Include a
test command, expected output and rollback/fallback path. Do not add
model files, generated audio, private references, saved application state, absolute home
paths or unsanitized raw profiler logs.
