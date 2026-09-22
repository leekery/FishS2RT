from __future__ import annotations

import json
import unittest
from pathlib import Path

import fish_s2
from fish_s2._runtime import PROTOCOL_VERSION


ROOT = Path(__file__).resolve().parents[2]


class PackagingContractTests(unittest.TestCase):
    def test_runtime_lock_matches_sdk(self) -> None:
        lock = json.loads((ROOT / "runtime-lock.json").read_text(encoding="utf-8"))
        self.assertEqual(lock["sdk_version"], fish_s2.__version__)
        self.assertEqual(lock["runtime_version"], fish_s2.__version__)
        self.assertEqual(lock["protocol_version"], PROTOCOL_VERSION)
        commit = lock["upstream"]["commit"]
        self.assertIn(commit, (ROOT / "UPSTREAM.md").read_text(encoding="utf-8"))
        self.assertIn(commit, (ROOT / "native/audio.cpp/CMakeLists.txt").read_text(encoding="utf-8"))

    def test_third_party_license_references_exist(self) -> None:
        expected = [
            "ggml-MIT.txt",
            "llamafile-sgemm-MIT.txt",
            "cJSON-MIT.txt",
            "libyaml.txt",
            "sentencepiece-Apache-2.0.txt",
            "absl-Apache-2.0.txt",
            "darts-clone-BSD-3-Clause.txt",
            "esaxx-MIT.txt",
            "protobuf-lite-BSD-3-Clause.txt",
        ]
        directory = ROOT / "licenses" / "third-party"
        self.assertEqual([], [name for name in expected if not (directory / name).is_file()])

    def test_source_build_inputs_exist(self) -> None:
        required = [
            ROOT / "native" / "audio.cpp" / "CMakeLists.txt",
            ROOT / "native" / "audio.cpp" / "scripts" / "build_windows.ps1",
            ROOT / "native" / "audio.cpp" / "external" / "ggml" / "CMakeLists.txt",
            ROOT / "native" / "audio.cpp" / "src" / "models" / "fish_audio" / "session.cpp",
            ROOT / "native" / "audio.cpp" / "external" / "ggml" / "src" / "ggml-cuda" / "bf16-p2.cu",
            ROOT / "tests" / "native" / "bf16_storage_test.cpp",
        ]
        self.assertEqual([], [str(path) for path in required if not path.is_file()])

    def test_source_archive_prunes_local_build_outputs(self) -> None:
        manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
        self.assertIn("prune native/audio.cpp/build", manifest)
        self.assertIn("global-exclude", manifest)
        for suffix in ("*.gguf", "*.wav", "*.exe", "*.dll", "*.pdb"):
            self.assertIn(suffix, manifest)


if __name__ == "__main__":
    unittest.main()
