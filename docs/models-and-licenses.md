# Models, integrity and licenses

## Two runtime profiles

Both files represent Fish S2 Pro. They are GGUF conversions published by the
audio.cpp project at immutable revision
`074bb6d15e0040ac12b07f04f2ae67c591b75d22`.

| ID | File | Bytes | SHA-256 |
|---|---|---:|---|
| Q8 | `fish-audio-s2-pro-q8_0.gguf` | 6,317,911,232 | `4ffc169447b7a26df8bf49e8637adb4000bfa763a22c018b6c03968564259d0b` |
| BF16 | `fish-audio-s2-pro-bf16.gguf` | 10,229,278,080 | `781fdece3ff837838c48f7d5a7b37e37c4d661a6416416ad57fe92fed47d96ff` |

The machine-readable source of truth is
[`config/fish_s2_models.json`](../config/fish_s2_models.json). The SDK does
not download, convert or modify either file. Users provide an existing GGUF path
and should verify its size/hash before use.

Q8 reduces weight precision and therefore is not mathematically identical to
BF16. The exact-output claims in this repository compare baseline and optimized
runtimes **within the same model file**, not Q8 against BF16.

## License boundary

- Runtime source derived from audio.cpp: repository [Apache-2.0 LICENSE](../LICENSE).
- Fish S2 Pro weights/materials: [Fish Audio Research License](../licenses/FISH_AUDIO_RESEARCH_LICENSE.md).
- Required Fish attribution: [NOTICE](../NOTICE).

The Fish license permits research and non-commercial use under its terms.
Commercial use requires a separate written agreement with Fish Audio. The full
license, not this summary, controls.

The repository, SDK ZIP and wheel do not contain Fish weights. Downloading and
using them still binds the user to the model license.

Primary sources:

- [Official Fish S2 Pro model card](https://huggingface.co/fishaudio/s2-pro)
- [Fish Audio Research License](https://huggingface.co/fishaudio/s2-pro/blob/0fd528fa1d2018e80b0ae60a8ae2042f58163a13/LICENSE.md)
- [audio.cpp GGUF conversion revision](https://huggingface.co/audio-cpp/audio.cpp-gguf/commit/074bb6d15e0040ac12b07f04f2ae67c591b75d22)
