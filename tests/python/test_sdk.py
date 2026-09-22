from __future__ import annotations

import sys
import tempfile
import unittest
import wave
from pathlib import Path

from fish_s2 import BusyError, FishS2, RuntimeConfig, Sampling
from fish_s2._runtime import inspect_runtime
from fish_s2._text import split_sentences_exact


ROOT = Path(__file__).resolve().parents[2]
FAKE_WORKER = ROOT / "tests" / "python" / "fake_fish_worker.py"


def write_reference(path: Path) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(44100)
        wav.writeframes(b"\0\0" * 16)


class SdkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.model = root / "model.gguf"
        self.model.write_bytes(b"fake")
        self.reference = root / "reference.wav"
        write_reference(self.reference)
        self.tts = FishS2(
            self.model,
            runtime=(sys.executable, str(FAKE_WORKER)),
        )

    def tearDown(self) -> None:
        self.tts.close()
        self.temp.cleanup()

    def test_persistent_voice_generate_and_sentence_join(self) -> None:
        self.tts.load()
        self.assertEqual(self.tts.runtime_info.protocol_version, 1)
        self.assertIn("streaming", self.tts.runtime_info.capabilities)
        pid = self.tts.pid
        voice = self.tts.prepare_voice(self.reference, "Точная расшифровка.")
        result = self.tts.generate(
            "61 и 74. Затем 109.",
            voice=voice,
            sampling=Sampling.numbers_safe(seed=7),
            segmentation="sentences",
            pause_ms=10,
        )
        self.assertEqual(pid, self.tts.pid)
        self.assertEqual(result.seed, 7)
        self.assertEqual(result.sample_rate, 44100)
        self.assertGreater(result.frames, 4)
        self.assertIn("part_0000.wall_ms", result.metrics)
        output = Path(self.temp.name) / "result.wav"
        self.assertEqual(result.save(output), output.resolve())
        self.assertTrue(output.is_file())

    def test_stream_yields_before_result_and_rejects_parallel_request(self) -> None:
        stream = self.tts.stream("Первое. Второе.", segmentation="sentences")
        first = next(stream)
        self.assertEqual(first.sequence, 0)
        self.assertEqual(first.segment, 0)
        with self.assertRaises(BusyError):
            self.tts.generate("Параллельный запрос")
        chunks = [first, *list(stream)]
        self.assertEqual([chunk.sequence for chunk in chunks], [0, 1, 2, 3])
        self.assertEqual(stream.result().frames, 4)

    def test_early_close_hard_cancels_owned_worker(self) -> None:
        stream = self.tts.stream("Остановить после первого чанка.")
        next(stream)
        stream.close()
        self.assertFalse(self.tts.loaded)

    def test_exact_split_keeps_digits_whitespace_and_tags(self) -> None:
        text = "FPS 61.5 — норма.  <|happy|>А 74?\nДа!"
        parts = split_sentences_exact(text)
        self.assertEqual("".join(parts), text)
        self.assertEqual(len(parts), 3)

    def test_runtime_handshake_checks_selected_bf16_profile(self) -> None:
        command = (sys.executable, str(FAKE_WORKER))
        info = inspect_runtime(command, bf16_storage="p3")
        self.assertEqual(info.runtime_version, "0.1.0-test")
        self.assertIn("bf16_p3", info.capabilities)

    def test_runtime_config_accepts_raw_p2_and_p3(self) -> None:
        for profile in ("raw", "p2", "p3"):
            with self.subTest(profile=profile):
                self.assertEqual(RuntimeConfig(bf16_storage=profile).bf16_storage, profile)


if __name__ == "__main__":
    unittest.main()
