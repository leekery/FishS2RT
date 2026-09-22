"""Configure runtime, sampling, sentence isolation and metric reporting."""

from __future__ import annotations

import argparse
from pathlib import Path

from fish_s2 import FishS2, RuntimeConfig, Sampling


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--transcript-file", required=True, type=Path)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--bf16-storage", choices=("raw", "p2", "p3"), default="raw")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stream-chunk-frames", type=int, default=8)
    parser.add_argument("--sentence-isolation", action="store_true")
    parser.add_argument("--pause-ms", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RuntimeConfig(
        backend="cuda",
        device=args.device,
        threads=args.threads,
        bf16_storage=args.bf16_storage,
    )
    sampling = Sampling(
        seed=args.seed,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        stream_chunk_frames=args.stream_chunk_frames,
    )
    transcript = args.transcript_file.read_text(encoding="utf-8").strip()

    with FishS2(args.model, config=config) as tts:
        voice = tts.prepare_voice(args.reference, transcript)
        audio = tts.generate(
            args.text,
            voice=voice,
            sampling=sampling,
            segmentation="sentences" if args.sentence_isolation else "native",
            pause_ms=args.pause_ms,
        )
        path = audio.save(args.output)
        print(f"saved={path}")
        print(f"frames={audio.frames} duration={audio.duration:.3f}s seed={audio.seed}")
        for key, value in sorted(audio.metrics.items()):
            print(f"{key}={value}")


if __name__ == "__main__":
    main()
