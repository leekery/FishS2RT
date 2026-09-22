# Third-party notices

This file describes the third-party components in FishS2RT's native worker.
The corresponding source-dependency license texts are included under
`licenses/third-party`. Binary runtime wheels additionally carry the
redistribution terms under `licenses/runtime`.

## Statically linked into `audiocpp_cli.exe`

- **ggml / llama.cpp components** — MIT License, copyright 2023-2026 the ggml
  authors. Full text: `licenses/third-party/ggml-MIT.txt`.
- **llamafile SGEMM integrated by ggml** — MIT License, copyright 2024 Mozilla
  Foundation. Full text: `licenses/third-party/llamafile-sgemm-MIT.txt`.
- **cJSON** — MIT License, copyright 2009-2017 Dave Gamble and cJSON
  contributors. Full text: `licenses/third-party/cJSON-MIT.txt`.
- **LibYAML** — MIT-style license, copyright 2017-2020 Ingy döt Net and
  2006-2016 Kirill Simonov. Full text: `licenses/third-party/libyaml.txt`.
- **SentencePiece** — Apache License 2.0. Full text:
  `licenses/third-party/sentencepiece-Apache-2.0.txt`.
- **SentencePiece vendored components** — Abseil (Apache-2.0), darts-clone
  (BSD-3-Clause style), esaxx (MIT-style) and protobuf-lite (BSD-3-Clause
  style). Full texts are included beside the SentencePiece license.

The optional native model manager and server are not part of this package;
therefore cpp-httplib/BoringSSL are not linked into this CLI binary.

## Redistributed runtime DLLs

The Windows runtime wheel includes unmodified CUDA redistributable DLLs
(`cudart`, `cublas`, `cublasLt`, `cufft`) from the selected toolkit. Their
redistribution terms are in Attachment A of the NVIDIA CUDA Toolkit EULA.
The wheel carries that toolkit's EULA at
`licenses/runtime/NVIDIA_CUDA_EULA.txt`.

The wheel also includes Microsoft Visual C++ and OpenMP runtime DLLs from the
Visual Studio 2022 redistributable directory. The redist pointer is at
`licenses/runtime/MICROSOFT_VISUAL_STUDIO_REDIST.txt`; current
Microsoft guidance is
<https://learn.microsoft.com/cpp/windows/redistributing-visual-cpp-files>.

`nvcuda.dll` is supplied by the user's NVIDIA driver and is not redistributed.
FFmpeg is optional for converting non-WAV references and is not bundled.

## Model materials

Fish S2 Pro weights are not included. Their separate license and mandatory
notice are in `licenses/FISH_AUDIO_RESEARCH_LICENSE.md` and `NOTICE`.
