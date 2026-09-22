"""Manual end-to-end SDK smoke test; outputs remain in memory/temp files."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from fish_s2 import FishS2, RuntimeConfig, Sampling


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--transcript-file", required=True)
    parser.add_argument("--text", default="SDK streaming smoke test.")
    parser.add_argument("--bf16-storage", choices=("raw", "p2", "p3"), default="raw")
    args = parser.parse_args()

    transcript = Path(args.transcript_file).read_text(encoding="utf-8").strip()
    started = time.perf_counter()
    with FishS2(args.model, config=RuntimeConfig(bf16_storage=args.bf16_storage)) as tts:
        loaded = time.perf_counter()
        pid = tts.pid
        voice = tts.prepare_voice(args.reference, transcript)
        prepared = time.perf_counter()
        chunks = []
        first_chunk_at = None
        with tts.stream(
            args.text,
            voice=voice,
            sampling=Sampling.numbers_safe(seed=42),
            segmentation="sentences",
        ) as stream:
            for chunk in stream:
                if first_chunk_at is None:
                    first_chunk_at = time.perf_counter()
                chunks.append(chunk)
            result = stream.result()
        finished = time.perf_counter()
        assert tts.pid == pid, "model process changed during the request"

    print(f"pid={pid}")
    print(f"load_s={loaded - started:.3f}")
    print(f"prepare_voice_s={prepared - loaded:.3f}")
    print(f"audible_chunk_s={(first_chunk_at - prepared) if first_chunk_at else -1:.3f}")
    print(f"generation_s={finished - prepared:.3f}")
    print(f"chunks={len(chunks)}")
    print(f"final_duration_s={result.duration:.3f}")
    print(f"final_frames={result.frames}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
