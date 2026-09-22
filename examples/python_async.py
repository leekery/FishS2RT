"""Bridge the synchronous chunk iterator into an asyncio application."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from fish_s2 import AudioChunk, FishS2

_END = object()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--transcript-file", required=True, type=Path)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


async def consume_chunk(chunk: AudioChunk) -> None:
    """Replace this body with an awaited WebSocket/network send."""
    print(
        f"chunk={chunk.sequence} bytes={len(chunk.pcm)} "
        f"rate={chunk.sample_rate} channels={chunk.channels}",
        flush=True,
    )


async def synthesize(tts: FishS2, text: str):
    with tts.stream(text) as stream:
        while True:
            item = await asyncio.to_thread(next, stream, _END)
            if item is _END:
                break
            await consume_chunk(item)
        return stream.result()


async def run() -> None:
    args = parse_args()
    transcript = args.transcript_file.read_text(encoding="utf-8").strip()

    with FishS2(args.model) as tts:
        tts.prepare_voice(args.reference, transcript)
        audio = await synthesize(tts, args.text)
        print(f"saved={audio.save(args.output)}", flush=True)


if __name__ == "__main__":
    asyncio.run(run())
