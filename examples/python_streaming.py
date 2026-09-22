"""Play PCM chunks during inference and optionally save the final render."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from fish_s2 import FishS2, Sampling


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    transcript = parser.add_mutually_exclusive_group(required=True)
    transcript.add_argument("--transcript")
    transcript.add_argument("--transcript-file", type=Path)
    parser.add_argument("--text", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        help="optional final WAV; omit to avoid saving test generations",
    )
    parser.add_argument("--device", help="sounddevice output device id or name")
    parser.add_argument("--prebuffer-chunks", type=int, default=2)
    parser.add_argument("--no-playback", action="store_true")
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


def output_device(value: str | None) -> int | str | None:
    if value is not None and value.isdecimal():
        return int(value)
    return value


def main() -> None:
    args = parse_args()
    if args.prebuffer_chunks < 1:
        raise SystemExit("--prebuffer-chunks must be at least 1")

    sounddevice: Any = None
    if not args.no_playback:
        try:
            import sounddevice
        except ImportError as exc:
            raise SystemExit(
                'playback requires: python -m pip install -e ".[audio]"'
            ) from exc

    sampling = (
        Sampling.numbers_safe(seed=args.seed)
        if args.numbers_safe
        else Sampling(seed=args.seed)
    )

    with FishS2(args.model) as tts:
        print(f"model loaded: pid={tts.pid}", flush=True)
        voice = tts.prepare_voice(args.reference, read_transcript(args))
        print("reference prepared; starting request", flush=True)

        started = time.perf_counter()
        first_chunk_at: float | None = None
        pending = []
        output_stream = None
        try:
            with tts.stream(
                args.text,
                voice=voice,
                sampling=sampling,
                segmentation=args.segmentation,
                pause_ms=args.pause_ms,
            ) as stream:
                for chunk in stream:
                    elapsed = time.perf_counter() - started
                    if first_chunk_at is None:
                        first_chunk_at = elapsed
                    print(
                        f"chunk={chunk.sequence} segment={chunk.segment} "
                        f"frames={chunk.frames} duration={chunk.duration:.3f}s "
                        f"ready_at={elapsed:.3f}s",
                        flush=True,
                    )

                    if args.no_playback:
                        continue
                    pending.append(chunk)
                    if output_stream is None and len(pending) >= args.prebuffer_chunks:
                        first = pending[0]
                        output_stream = sounddevice.RawOutputStream(
                            device=output_device(args.device),
                            samplerate=first.sample_rate,
                            channels=first.channels,
                            dtype="int16",
                        )
                        output_stream.start()
                        print("playback started", flush=True)
                    if output_stream is not None:
                        while pending:
                            output_stream.write(pending.pop(0).pcm)

                if not args.no_playback and output_stream is None and pending:
                    first = pending[0]
                    output_stream = sounddevice.RawOutputStream(
                        device=output_device(args.device),
                        samplerate=first.sample_rate,
                        channels=first.channels,
                        dtype="int16",
                    )
                    output_stream.start()
                    print("playback started", flush=True)
                while output_stream is not None and pending:
                    output_stream.write(pending.pop(0).pcm)

                final = stream.result()
        finally:
            if output_stream is not None:
                output_stream.stop()
                output_stream.close()

        elapsed = time.perf_counter() - started
        first_text = "n/a" if first_chunk_at is None else f"{first_chunk_at:.3f}s"
        print(
            f"complete wall={elapsed:.3f}s first_chunk={first_text} "
            f"audio={final.duration:.3f}s",
            flush=True,
        )
        if args.output is not None:
            print(f"saved={final.save(args.output)}", flush=True)


if __name__ == "__main__":
    main()
