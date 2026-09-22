"""Reuse one loaded model and prepared voice for several sequential requests."""

from __future__ import annotations

import argparse
from pathlib import Path
from secrets import token_hex

from fish_s2 import FishS2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--transcript-file", required=True, type=Path)
    parser.add_argument(
        "--text",
        action="append",
        required=True,
        help="repeat this option for each sequential generation",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    transcript = args.transcript_file.read_text(encoding="utf-8").strip()

    tts = FishS2(args.model).load()
    try:
        voice = tts.prepare_voice(args.reference, transcript)
        original_pid = tts.pid
        print(f"loaded once: pid={original_pid}", flush=True)

        for index, text in enumerate(args.text, 1):
            name = f"fish-{index:03d}-{token_hex(4)}.wav"
            path = tts.generate_to_file(
                text,
                args.output_dir / name,
                voice=voice,
            )
            if tts.pid != original_pid:
                raise RuntimeError("worker PID changed unexpectedly")
            print(f"request={index} pid={tts.pid} saved={path}", flush=True)
    finally:
        tts.unload()
        print(f"unloaded: loaded={tts.loaded} pid={tts.pid}", flush=True)


if __name__ == "__main__":
    main()
