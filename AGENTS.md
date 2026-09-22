# FishS2RT contributor guidance

- Keep the public SDK independent of playback devices, GUIs and HTTP servers.
- Keep one persistent native worker per loaded `FishS2` instance.
- Never rewrite user text, digits or Fish tags during segmentation.
- Preserve Q8 and BF16 support through the same public API.
- Treat P2/P3 as optional exact-storage profiles with documented fallback.
- Do not commit weights, voice references, transcripts, generated audio, raw
  profiles, absolute user paths or private investigation material.
- Run model-free tests and `python scripts/check_publication.py` before release.
- GPU evidence must report cold load, warm inference, TTFA, RTF and memory
  separately. Historical measurements do not validate a rebuilt binary.
