# Publishing FishS2RT

The public repository is `https://github.com/leekery/FishS2RT`. It is a clean
product monorepo, not the historical integration checkout. The native snapshot
origin and exact imported commit are recorded in [UPSTREAM.md](../UPSTREAM.md).

## Release units

One Git tag, for example `v0.1.0`, produces two coordinated distributions:

- `fish_s2-0.1.0-py3-none-any.whl` — Python SDK;
- `fishs2rt_runtime-0.1.0-py3-none-win_amd64.whl` — native worker, manifest,
  licenses and selected redistributable DLLs.

The SDK version, runtime version and protocol are locked in
`runtime-lock.json`. Never replace a published runtime binary while retaining
the same version.

## Before the first commit

```powershell
python -m pip install -e . --no-deps
python -m unittest discover -s tests/python -p "test_*.py" -v
python scripts/check_publication.py
git status --short
```

The checker reviews the entire prospective Git tree, not only selected Fish
paths. Manually review the resulting `git diff --cached` before committing.

Forbidden release content includes weights, voice references, transcripts,
generated audio, raw profiles/logs, local application state, absolute private
paths, GUI/WebUI code and private investigation material. Public regression
tests that use synthetic data are permitted.

## Native source validation

At minimum, configure the copied source in a clean directory. A release build
must also compile `audiocpp_cli`, run `--build-info-json`, and pass the manual
Q8/BF16 matrix on supported hardware. The source archive must configure without
files ignored by Git.

The optional storage regression target is enabled with:

```powershell
cmake -S native/audio.cpp -B build/native-tests `
  -DENGINE_ENABLE_CUDA=ON `
  -DFISHS2RT_BUILD_NATIVE_TESTS=ON
cmake --build build/native-tests --target fish_bf16_p2_tests
```

It uses `tests/native/bf16_storage_test.cpp`, which is public software test code,
not a dependency on the private investigation tree.

## Build artifacts

```powershell
$repo = (Resolve-Path .).Path
Push-Location $env:TEMP
try { python -m build --outdir (Join-Path $repo "dist") $repo }
finally { Pop-Location }
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File scripts/build_runtime_wheel.ps1 `
  -Runtime native/audio.cpp/build/windows-cuda-portable-release/bin/audiocpp_cli.exe
```

Install both wheels in a fresh virtual environment outside the checkout. Verify
that runtime discovery and `--build-info-json` work before loading a model.
Inspect wheel contents for the executable, CUDA/MSVC DLLs, manifest,
`BUILD_INFO.txt`, NOTICE and third-party licenses. The runtime-wheel script
requires a clean commit and a worker rebuilt from that commit. For a local
packaging check before the first commit, pass `-AllowUncommitted` and keep that
wheel out of public releases.

## Manual GPU release gate

Run Q8 and BF16 through the same public API. Confirm persistent PID reuse,
reference preparation, live chunk ordering, final render, unload/recovery and
the selected P2/P3 fallbacks. Record model/runtime hashes and report cold load,
warm wall time, audible TTFA, audio duration, RTF and GPU memory separately.

The performance table in the README is a retained measurement from one system.
Do not silently apply it to a rebuilt binary: update or keep the table only
after comparing the release candidate with the pinned baseline.

## First push

Only after the checklist is complete:

```powershell
git add .
git diff --cached --check -- README.md README.ru.md docs examples src `
  packaging scripts tests pyproject.toml MANIFEST.in runtime-lock.json `
  UPSTREAM.md THIRD_PARTY_NOTICES.md .github .gitignore .gitattributes AGENTS.md
git diff --cached --stat
git commit -m "Initial FishS2RT SDK and Fish native runtime"
git remote -v
git branch --show-current
git push -u origin main
```

Verify that `origin` is `leekery/FishS2RT` immediately before pushing. Do not
attach model weights, references, generated speech or raw evidence to GitHub.
The scoped whitespace check covers FishS2RT-authored files. A whole-tree
`git diff --cached --check` also reports pre-existing whitespace in copied
third-party source and license text; preserve those upstream files verbatim.
