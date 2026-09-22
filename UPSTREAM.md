# Upstream provenance

FishS2RT's native runtime is derived from
[`0xShug0/audio.cpp`](https://github.com/0xShug0/audio.cpp) at commit
`f2b4937306daa25f5c78520f3c626ed31495a37a`.

The source-only snapshot is stored in `native/audio.cpp`. It intentionally omits
the upstream Web UI, model weights, generated media, benchmark evidence and
unrelated release artifacts. The snapshot retains the framework and vendored
source dependencies needed to configure and build the Fish Audio worker.
Upstream model registration declarations remain in `CMakeLists.txt` for easier
source comparisons; this snapshot creates only the `fish_audio` model target.

FishS2RT changes include the persistent request-loop integration, incremental
audio streaming, bounded codec/KV state, Fish S2 CUDA paths, exact final render,
BF16 storage profiles and the Python SDK. The Apache-2.0 license, upstream
copyright notices and third-party licenses remain in the repository.

When updating the snapshot:

1. record the new exact upstream commit in this file and `runtime-lock.json`;
2. import source changes separately from FishS2RT-specific changes;
3. rebuild and run model-free, native and manual Q8/BF16 release checks;
4. never copy upstream model files, generated audio, Web UI assets or private
   investigation material into the public tree.
