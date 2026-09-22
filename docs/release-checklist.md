# FishS2RT release checklist

## Repository and privacy

- [ ] repository is the clean FishS2RT product tree, not the integration checkout
- [ ] `python scripts/check_publication.py` passes
- [ ] no GGUF, safetensors, WAV/MP3, transcript, private path, GUI or raw profile
- [ ] `UPSTREAM.md`, Apache-2.0, NOTICE and third-party license texts are present
- [ ] `runtime-lock.json` versions and upstream commit match the release
- [ ] `git diff --cached` and the complete first-commit file list were reviewed

## SDK artifacts

- [ ] editable install passes from the repository root
- [ ] model-free Python tests pass
- [ ] `python -m build` creates SDK wheel and source archive
- [ ] SDK wheel is `py3-none-any`
- [ ] source archive contains native source, build script, docs and licenses
- [ ] SDK wheel installs and imports in a fresh environment outside the checkout

## Native runtime artifact

- [ ] clean source archive configures without private/ignored files
- [ ] Windows CUDA build compiles `audiocpp_cli`
- [ ] `--build-info-json` reports product, versions, commits and capabilities
- [ ] runtime wheel is `py3-none-win_amd64`
- [ ] runtime wheel contains worker, manifest, NOTICE and required licenses
- [ ] SDK rejects a wrong protocol and accepts the matching runtime wheel
- [ ] required DLLs are bundled or explicitly documented

## API and lifecycle

- [ ] Q8 and BF16 load through the same `FishS2` API
- [ ] sequential requests retain the worker PID and prepared voice cache
- [ ] stream sequence begins at zero and remains contiguous
- [ ] canonical file comes from `stream.result()`, not concatenated live chunks
- [ ] concurrent use raises `BusyError`
- [ ] early close hard-cancels only the owned worker
- [ ] `unload()` releases process and GPU memory
- [ ] sentence segmentation preserves text, digits, whitespace and Fish tags
- [ ] paths containing spaces and non-ASCII characters pass

## Performance and quality

- [ ] release binary, model, settings, seed and reference hashes are recorded
- [ ] cold load, reference preparation, first chunk, audible start and wall time are separate
- [ ] audio duration, RTF, peak and retained GPU memory are reported separately
- [ ] Q8/BF16 comparisons use matched requests and counterbalanced process order
- [ ] final-output equality/quality claims inspect the complete output
- [ ] historical README results were not assumed valid for an unmeasured rebuild

## Publication

- [ ] branch is `main`
- [ ] `origin` is exactly `https://github.com/leekery/FishS2RT.git`
- [ ] tag and both distribution versions are identical
- [ ] release contains no model, reference, generated audio or private evidence
