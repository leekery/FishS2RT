#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/models/fish_audio/assets.h"
#include "engine/models/fish_audio/types.h"

#include <functional>
#include <memory>

namespace engine::models::fish_audio {

class FishAudioARRuntime {
public:
    using StreamCallback = std::function<void(const engine::codecs::FishDacCodes &, bool final)>;

    FishAudioARRuntime(
        std::shared_ptr<const FishAudioAssets> assets,
        core::BackendConfig backend,
        int threads,
        size_t graph_arena_bytes,
        size_t weight_context_bytes,
        assets::TensorStorageType weight_storage_type);
    ~FishAudioARRuntime();

    engine::codecs::FishDacCodes generate(const FishAudioPrompt & prompt, const FishAudioGenerationOptions & options);
    engine::codecs::FishDacCodes generate_streaming(
        const FishAudioPrompt & prompt,
        const FishAudioGenerationOptions & options,
        int64_t frames_per_chunk,
        const StreamCallback & callback);
    void release_runtime_graphs();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::fish_audio
