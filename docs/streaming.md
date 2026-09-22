# Streaming contract and playback

## What streaming means

Fish generation remains synchronous inside the native worker, but its generator
pushes newly decoded audio into a stream-event callback. The DAC path consumes
only delta codec frames, retains a 128-frame transformer KV window on the device,
and replays the ten post-transformer frames required by the causal CNN. The
Python SDK receives each completed event and yields an `AudioChunk` before the
complete request is finished.

```text
Python stream() -> persistent worker -> semantic generation -> stateful delta DAC
      ^                                                         |
      |-------------------- PCM16 AudioChunk -------------------|

all live segments complete -> exact-length final render -> AudioStream.result()
```

Calling `stream()` loads the model if required. The request itself starts only
when the iterator advances or `result()` drains it.

## Minimal consumer

```python
with tts.stream("Audio arrives during generation.") as stream:
    for chunk in stream:
        print(chunk.sequence, chunk.duration)
    final = stream.result()
```

Every chunk is interleaved signed 16-bit little-endian PCM. Never hard-code the
rate or channels; use `chunk.sample_rate` and `chunk.channels`.

## Immediate playback with sounddevice

Install the optional playback dependency:

```powershell
python -m pip install -e ".[audio]"
```

```python
import sounddevice as sd

with tts.stream("Immediate playback example.") as stream:
    output = None
    try:
        for chunk in stream:
            if output is None:
                output = sd.RawOutputStream(
                    samplerate=chunk.sample_rate,
                    channels=chunk.channels,
                    dtype="int16",
                )
                output.start()
            output.write(chunk.pcm)
        final = stream.result()
    finally:
        if output is not None:
            output.stop()
            output.close()
```

`RawOutputStream.write()` accepts buffer objects, so NumPy is not required. The
SDK itself deliberately does not select a device, volume, latency or prebuffer.

## Two-chunk safety prebuffer

Some Windows devices/drivers clip the first block when playback starts from an
empty buffer. An application can retain two SDK chunks without changing the SDK
contract:

```python
import sounddevice as sd

with tts.stream("Buffered playback.") as stream:
    pending = []
    output = None
    try:
        for chunk in stream:
            pending.append(chunk)
            if output is None and len(pending) >= 2:
                first = pending[0]
                output = sd.RawOutputStream(
                    samplerate=first.sample_rate,
                    channels=first.channels,
                    dtype="int16",
                )
                output.start()
            if output is not None:
                while pending:
                    output.write(pending.pop(0).pcm)
        if output is None and pending:
            first = pending[0]
            output = sd.RawOutputStream(
                samplerate=first.sample_rate,
                channels=first.channels,
                dtype="int16",
            )
            output.start()
        while output is not None and pending:
            output.write(pending.pop(0).pcm)
        final = stream.result()
    finally:
        if output is not None:
            output.stop()
            output.close()
```

Chunk zero is always written first. A prebuffer improves device robustness at
the cost of delaying audible start by the time needed for the second chunk.

## Live audio versus final audio

Live chunks are stateful delta decodes optimized for early playback. Their
concatenation is not promised to equal the final render sample-for-sample:
fixed live graph shapes can select different CUDA reduction paths from the
independent exact-length final graph. Retained local tests measured small
differences between live PCM and final output.

Therefore:

- send each `AudioChunk.pcm` to live playback/network consumers once;
- use `stream.result()` for archival WAV, downloads and objective comparisons;
- do not play the final `Audio` again after live playback unless replay is
  explicitly desired;
- do not claim exactness by hashing concatenated live chunks.

## Saving while playing

The SDK retains the final result independently, so no live reconstruction is
needed:

```python
with tts.stream(text) as stream:
    for chunk in stream:
        playback.write(chunk.pcm)
    saved_path = stream.result().save("final.wav")
```

## Forwarding over a socket

Send format metadata once, then length-frame each PCM block. The framing below
is only an application example, not an SDK wire protocol:

```python
import json
import struct

with tts.stream(text) as stream:
    first = True
    for chunk in stream:
        if first:
            header = json.dumps({
                "format": "s16le",
                "sample_rate": chunk.sample_rate,
                "channels": chunk.channels,
            }).encode("utf-8")
            sock.sendall(struct.pack("!I", len(header)) + header)
            first = False
        sock.sendall(struct.pack("!I", len(chunk.pcm)) + chunk.pcm)
    final = stream.result()
```

Applications must add authentication, size limits, disconnect handling and
backpressure suitable for their network protocol.

## Using an async application

The public iterator is synchronous. Bridge one blocking `next()` at a time to a
thread; do not run multiple `next()` calls concurrently:

```python
import asyncio

_END = object()


async def next_chunk(stream):
    return await asyncio.to_thread(next, stream, _END)


async def synthesize(tts, text):
    with tts.stream(text) as stream:
        while True:
            item = await next_chunk(stream)
            if item is _END:
                break
            await websocket.send_bytes(item.pcm)
        return stream.result()
```

This does not make one `FishS2` instance concurrent: a second request still
raises `BusyError`.

## Backpressure and slow consumers

The SDK does not drop or reorder chunks. Its stdout reader uses a bounded queue;
if the consumer stops reading for long enough, worker output eventually
backpressures instead of growing Python memory without limit. Native v0.1 also
writes each live chunk to a temporary WAV before Python reads and deletes it.

Consequences:

- consume chunks promptly;
- put network/device buffering after the SDK iterator, with an application-level
  bound and policy;
- a permanently blocked sink should close the stream, which hard-cancels the
  worker;
- temporary chunk I/O is an implementation cost and a future dedicated binary
  protocol is an optimization opportunity.

## Cancellation

```python
with tts.stream(long_text) as stream:
    for chunk in stream:
        if user_pressed_stop():
            stream.close()
            break
```

After generation has begun, `close()` terminates only the worker owned by that
`FishS2` instance. The model becomes unloaded, GPU allocations are released and
old `Voice` handles are invalid. Call `load()` and `prepare_voice()` again before
the next request.

If clean cancellation while keeping the model loaded is required, it needs a
future native cooperative cancellation token checked between generation steps;
v0.1 does not pretend that stdin `cancel` can interrupt a worker blocked inside
synchronous generation.

## Failure handling

Always use the stream context manager. If playback code raises, `__exit__` closes
the unfinished stream and releases its native worker rather than leaving hidden
generation running.

```python
try:
    with tts.stream(text) as stream:
        for chunk in stream:
            device.write(chunk.pcm)
except Exception:
    # tts.loaded is False if generation had started and was hard-cancelled.
    raise
```

For timeouts and native crashes, the SDK also moves to an unloaded state.

## Measuring streaming

Measure these separately:

1. cold model load;
2. reference preparation;
3. request start to first yielded chunk (SDK TTFA);
4. request start to first audible sample (device TTFA, including prebuffer);
5. total generation wall time;
6. final audio duration and RTF;
7. peak and retained GPU memory.

Do not label the first chunk timestamp "audible" if the application waits for a
prebuffer or the device adds latency.
