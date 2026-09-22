"""Tiny request-loop stand-in used by model-free SDK tests."""

from __future__ import annotations

import json
import sys
import wave
from pathlib import Path


def write_wav(path: Path, samples: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = b"".join(sample.to_bytes(2, "little", signed=True) for sample in samples)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(44100)
        wav.writeframes(pcm)


if "--build-info-json" in sys.argv:
    print(json.dumps({
        "product": "FishS2RT",
        "runtime_version": "0.1.0-test",
        "protocol_version": 1,
        "source_commit": "test",
        "upstream_commit": "f2b4937306daa25f5c78520f3c626ed31495a37a",
        "capabilities": [
            "streaming", "reference_cache", "q8", "bf16", "bf16_p2", "bf16_p3"
        ],
    }))
    raise SystemExit(0)

print("request_loop_ready=1", flush=True)
for line in sys.stdin:
    command = json.loads(line)
    action = command.get("command")
    command_id = command.get("id", "unknown")
    if action == "quit":
        print("request_loop_bye=1", flush=True)
        break
    if action == "prepare":
        print(f"request_loop_done={command_id}", flush=True)
        continue
    if action != "run":
        print(f"request_loop_error={command_id}\tbad command", flush=True)
        continue
    request_file = Path(command["request_sequence"])
    output_dir = Path(command["out_dir"])
    requests = json.loads(request_file.read_text(encoding="utf-8"))["requests"]
    for request_index, request in enumerate(requests):
        request_id = request["id"]
        for chunk_index in range(2):
            path = output_dir / f"{request_id}_stream_{chunk_index}.wav"
            write_wav(path, [100 + request_index * 10 + chunk_index])
            print(f"audio_out[{request_id}_stream_{chunk_index}]={path}", flush=True)
        final = output_dir / f"{request_id}.wav"
        write_wav(final, [1000 + request_index, 2000 + request_index])
        print(f"audio_out={final}", flush=True)
        print(f"metrics[{request_id}].wall_ms={10 + request_index}", flush=True)
    print(f"request_loop_done={command_id}", flush=True)
