"""Measure Fish live-chunk cadence without playback or output WAV files."""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

from fish_s2 import FishS2, RuntimeConfig, Sampling


DEFAULT_TEXT = (
    "Пришло время обеда, он нашёл в столовой тихий столик в углу, где никто "
    "не станет его тревожить, где можно будет жевать и глотать неаномальную "
    "еду, пить термоядерной крепости кофе и усваивать непростые уроки "
    "сегодняшнего утра."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--transcript-file", required=True, type=Path)
    parser.add_argument("--frames", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--max-tokens", type=int, nargs="+", default=[1024])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16-storage", choices=("raw", "p2", "p3"), default="raw")
    parser.add_argument("--text-chunk-size", type=int, default=200)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def main() -> None:
    args = parse_args()
    transcript = args.transcript_file.read_text(encoding="utf-8").strip()
    report: dict[str, object] = {"runs": []}

    with FishS2(args.model, config=RuntimeConfig(bf16_storage=args.bf16_storage)) as tts:
        voice = tts.prepare_voice(args.reference, transcript)
        tts.generate(
            "Проверка прогрева модели.",
            voice=voice,
            sampling=Sampling(seed=args.seed, stream_chunk_frames=8),
        )

        for max_tokens, frames in (
            (max_tokens, frames)
            for max_tokens in args.max_tokens
            for frames in args.frames
        ):
            started = time.perf_counter()
            arrivals: list[float] = []
            durations: list[float] = []
            live_parts: list[bytes] = []
            with tts.stream(
                args.text,
                voice=voice,
                sampling=Sampling(
                    seed=args.seed,
                    max_tokens=max_tokens,
                    stream_chunk_frames=frames,
                    text_chunk_size=args.text_chunk_size,
                ),
                segmentation="native",
            ) as stream:
                for chunk in stream:
                    arrivals.append(time.perf_counter() - started)
                    durations.append(chunk.duration)
                    live_parts.append(chunk.pcm)
                final = stream.result()
            wall = time.perf_counter() - started
            live_pcm = b"".join(live_parts)
            live_samples = array.array("h")
            final_samples = array.array("h")
            live_samples.frombytes(live_pcm)
            final_samples.frombytes(final.pcm)
            compared = min(len(live_samples), len(final_samples))
            errors = [int(live_samples[i]) - int(final_samples[i]) for i in range(compared)]
            signal_energy = sum(int(final_samples[i]) ** 2 for i in range(compared))
            error_energy = sum(error * error for error in errors)
            intervals = [right - left for left, right in zip(arrivals, arrivals[1:])]
            worst_intervals = sorted(
                ((value, index + 1) for index, value in enumerate(intervals)),
                reverse=True,
            )[:5]
            run = {
                "max_tokens": max_tokens,
                "stream_chunk_frames": frames,
                "chunks": len(arrivals),
                "first_chunk_s": arrivals[0] if arrivals else None,
                "wall_s": wall,
                "audio_s": final.duration,
                "rtf": wall / final.duration,
                "mean_interval_s": statistics.fmean(intervals) if intervals else 0.0,
                "p50_interval_s": percentile(intervals, 0.50),
                "p95_interval_s": percentile(intervals, 0.95),
                "max_interval_s": max(intervals, default=0.0),
                "worst_intervals_after_chunk": [
                    {"chunk": chunk, "seconds": value} for value, chunk in worst_intervals
                ],
                "mean_chunk_audio_s": statistics.fmean(durations) if durations else 0.0,
                "final_pcm_sha256": hashlib.sha256(final.pcm).hexdigest(),
                "live_pcm_sha256": hashlib.sha256(live_pcm).hexdigest(),
                "live_final_same_bytes": live_pcm == final.pcm,
                "live_final_length_delta_bytes": len(live_pcm) - len(final.pcm),
                "live_final_max_abs_pcm16": max((abs(error) for error in errors), default=0),
                "live_final_snr_db": (
                    10.0 * math.log10(signal_energy / error_energy)
                    if signal_energy > 0 and error_energy > 0
                    else (math.inf if error_energy == 0 else -math.inf)
                ),
                "metrics": dict(final.metrics),
            }
            report["runs"].append(run)
            print(json.dumps(run, ensure_ascii=False), flush=True)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
