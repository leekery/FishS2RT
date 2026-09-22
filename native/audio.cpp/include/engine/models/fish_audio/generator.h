#pragma once

#include "engine/framework/codecs/fish_dac_codec_runtime.h"
#include "engine/models/fish_audio/ar.h"
#include "engine/models/fish_audio/prompt_builder.h"
#include "engine/models/fish_audio/tokenizer_text.h"

#include <functional>
#include <memory>
#include <optional>

namespace engine::models::fish_audio {

struct FishAudioGenerationResult {
    runtime::AudioBuffer audio;
    engine::codecs::FishDacCodes codes;
    double codec_stream_decode_ms = 0.0;
    double codec_final_decode_ms = 0.0;
};

class FishAudioGenerator {
public:
    using AudioChunkCallback = std::function<void(const runtime::AudioBuffer &)>;

    FishAudioGenerator(
        std::shared_ptr<const FishAudioAssets> assets,
        std::unique_ptr<FishAudioARRuntime> ar,
        std::unique_ptr<engine::codecs::FishDacCodecRuntime> codec);
    ~FishAudioGenerator();

    engine::codecs::FishDacCodes encode_reference(const runtime::AudioBuffer & audio);
    FishAudioGenerationResult generate(
        const FishAudioRequest & request,
        const std::vector<engine::codecs::FishDacCodes> & reference_codes,
        const std::optional<FishAudioConversationTurn> & previous_turn,
        bool mem_saver);
    FishAudioGenerationResult generate_streaming(
        const FishAudioRequest & request,
        const std::vector<engine::codecs::FishDacCodes> & reference_codes,
        const std::optional<FishAudioConversationTurn> & previous_turn,
        bool mem_saver,
        int64_t frames_per_chunk,
        const AudioChunkCallback & callback,
        bool defer_final_audio = false);
    runtime::AudioBuffer decode_generated_codes(const engine::codecs::FishDacCodes & codes);

private:
    FishAudioGenerationResult generate_impl(
        const FishAudioRequest & request,
        const std::vector<engine::codecs::FishDacCodes> & reference_codes,
        const std::optional<FishAudioConversationTurn> & previous_turn,
        bool mem_saver,
        int64_t frames_per_chunk,
        const AudioChunkCallback & callback,
        bool defer_final_audio);
    void release_runtime_graphs_noexcept() noexcept;

    std::shared_ptr<const FishAudioAssets> assets_;
    FishAudioTextTokenizer tokenizer_;
    FishAudioPromptBuilder prompt_builder_;
    std::unique_ptr<FishAudioARRuntime> ar_;
    std::unique_ptr<engine::codecs::FishDacCodecRuntime> codec_;
};

}  // namespace engine::models::fish_audio
