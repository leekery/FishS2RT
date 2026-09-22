"""Generate one final voice-cloned WAV with explicit command-line inputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from fish_s2 import FishS2, Sampling


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="Q8_0 or BF16 GGUF")
    parser.add_argument("--reference", required=True, type=Path, help="reference WAV")
    transcript = parser.add_mutually_exclusive_group(required=True)
    transcript.add_argument("--transcript", help="exact reference transcript")
    transcript.add_argument(
        "--transcript-file",
        type=Path,
        help="UTF-8 text file containing the exact reference transcript",
    )
    parser.add_argument("--text", required=True, help="text to synthesize")
    parser.add_argument("--output", type=Path, default=Path("result.wav"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--numbers-safe", action="store_true")
    parser.add_argument(
        "--segmentation",
        choices=("native", "sentences"),
        default="native",
    )
    parser.add_argument("--pause-ms", type=int, default=0)
    return parser.parse_args()


def read_transcript(args: argparse.Namespace) -> str:
    if args.transcript_file is not None:
        return args.transcript_file.read_text(encoding="utf-8").strip()
    return args.transcript.strip()


def main() -> None:
    args = parse_args()
    sampling = (
        Sampling.numbers_safe(seed=args.seed)
        if args.numbers_safe
        else Sampling(seed=args.seed)
    )

    with FishS2(args.model) as tts:
        print(f"model loaded: pid={tts.pid}", flush=True)
        voice = tts.prepare_voice(args.reference, read_transcript(args))
        print("reference prepared", flush=True)
        audio = tts.generate(
            args.text,
            voice=voice,
            sampling=sampling,
            segmentation=args.segmentation,
            pause_ms=args.pause_ms,
        )
        output = audio.save(args.output)
        print(
            f"saved={output} duration={audio.duration:.3f}s "
            f"rate={audio.sample_rate}Hz channels={audio.channels}",
            flush=True,
        )
        for name, value in sorted(audio.metrics.items()):
            print(f"metric {name}={value}", flush=True)


if __name__ == "__main__":
    main()
