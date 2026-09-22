#include "engine/models/fish_audio/generator.h"

#include "engine/framework/debug/profiler.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <utility>

namespace engine::models::fish_audio {
namespace {

using Clock = std::chrono::steady_clock;

}  // namespace

FishAudioGenerator::FishAudioGenerator(
    std::shared_ptr<const FishAudioAssets> assets,
    std::unique_ptr<FishAudioARRuntime> ar,
    std::unique_ptr<engine::codecs::FishDacCodecRuntime> codec)
    : assets_(std::move(assets)),
      tokenizer_(assets_),
      prompt_builder_(assets_, tokenizer_),
      ar_(std::move(ar)),
      codec_(std::move(codec)) {
    if (assets_ == nullptr || ar_ == nullptr || codec_ == nullptr) {
        throw std::runtime_error("Fish Audio generator requires assets, AR runtime, and codec runtime");
    }
}

FishAudioGenerator::~FishAudioGenerator() = default;

engine::codecs::FishDacCodes FishAudioGenerator::encode_reference(const runtime::AudioBuffer & audio) {
    auto codes = codec_->encode_codes(audio);
    codec_->release_encode_graph();
    return codes;
}

FishAudioGenerationResult FishAudioGenerator::generate(
    const FishAudioRequest & request,
    const std::vector<engine::codecs::FishDacCodes> & reference_codes,
    const std::optional<FishAudioConversationTurn> & previous_turn,
    bool mem_saver) {
    return generate_impl(request, reference_codes, previous_turn, mem_saver, 0, {}, false);
}

FishAudioGenerationResult FishAudioGenerator::generate_streaming(
    const FishAudioRequest & request,
    const std::vector<engine::codecs::FishDacCodes> & reference_codes,
    const std::optional<FishAudioConversationTurn> & previous_turn,
    bool mem_saver,
    int64_t frames_per_chunk,
    const AudioChunkCallback & callback,
    bool defer_final_audio) {
    if (frames_per_chunk <= 0) {
        throw std::runtime_error("Fish Audio streaming frames_per_chunk must be positive");
    }
    if (!callback) {
        throw std::runtime_error("Fish Audio streaming requires an audio callback");
    }
    return generate_impl(
        request,
        reference_codes,
        previous_turn,
        mem_saver,
        frames_per_chunk,
        callback,
        defer_final_audio);
}

runtime::AudioBuffer FishAudioGenerator::decode_generated_codes(
    const engine::codecs::FishDacCodes & codes) {
    if (codes.frames <= 0) {
        throw std::runtime_error("Fish Audio generated no audio frames");
    }
    try {
        auto audio = codec_->decode_codes(codes);
        codec_->release_runtime_graphs();
        return audio;
    } catch (...) {
        release_runtime_graphs_noexcept();
        throw;
    }
}

void FishAudioGenerator::release_runtime_graphs_noexcept() noexcept {
    try {
        codec_->release_runtime_graphs();
    } catch (...) {
    }
    try {
        ar_->release_runtime_graphs();
    } catch (...) {
    }
}

FishAudioGenerationResult FishAudioGenerator::generate_impl(
    const FishAudioRequest & request,
    const std::vector<engine::codecs::FishDacCodes> & reference_codes,
    const std::optional<FishAudioConversationTurn> & previous_turn,
    bool mem_saver,
    int64_t frames_per_chunk,
    const AudioChunkCallback & callback,
    bool defer_final_audio) {
    engine::debug::trace_log_scalar("fish_audio.request.has_reference", !request.references.empty());
    engine::debug::trace_log_scalar("fish_audio.request.reference_count", static_cast<int64_t>(request.references.size()));
    engine::debug::trace_log_scalar("fish_audio.request.text_chars", static_cast<int64_t>(request.text.size()));
    engine::debug::trace_log_scalar("fish_audio.request.has_previous_turn", previous_turn.has_value());
    engine::debug::trace_log_scalar("fish_audio.sampler.seed", request.generation.seed);
    const auto prompt_start = Clock::now();
    const auto prompt = prompt_builder_.build(request, reference_codes, previous_turn);
    engine::debug::timing_log_scalar(
        "fish_audio.prompt_build_ms",
        engine::debug::elapsed_ms(prompt_start, Clock::now()));

    const auto ar_start = Clock::now();
    FishAudioGenerationResult result;
    int64_t streamed_frames = 0;
    double codec_decode_ms = 0.0;
    double codec_stream_decode_ms = 0.0;
    double codec_final_decode_ms = 0.0;
    if (callback) {
        try {
            codec_->begin_decode_stream({frames_per_chunk});
            result.codes = ar_->generate_streaming(
                prompt,
                request.generation,
                frames_per_chunk,
                [&](const engine::codecs::FishDacCodes & prefix_codes, bool final) {
                    if (prefix_codes.frames < streamed_frames) {
                        throw std::runtime_error("Fish Audio streaming codec prefix shrank unexpectedly");
                    }
                    const int64_t delta_frames = prefix_codes.frames - streamed_frames;
                    if (delta_frames <= 0) {
                        return;
                    }
                    engine::codecs::FishDacCodes delta_codes;
                    delta_codes.codebooks = prefix_codes.codebooks;
                    delta_codes.frames = delta_frames;
                    delta_codes.codes.resize(static_cast<size_t>(delta_codes.codebooks * delta_frames));
                    for (int64_t codebook = 0; codebook < delta_codes.codebooks; ++codebook) {
                        const auto source = prefix_codes.codes.begin() + static_cast<std::ptrdiff_t>(
                            codebook * prefix_codes.frames + streamed_frames);
                        const auto destination = delta_codes.codes.begin() + static_cast<std::ptrdiff_t>(
                            codebook * delta_frames);
                        std::copy_n(source, delta_frames, destination);
                    }
                    const auto decode_start = Clock::now();
                    auto chunk = codec_->decode_stream_chunk(delta_codes, final);
                    codec_stream_decode_ms += engine::debug::elapsed_ms(decode_start, Clock::now());
                    streamed_frames = prefix_codes.frames;
                    if (!chunk.audio.samples.empty()) {
                        callback(chunk.audio);
                    }
                });
            codec_->end_decode_stream();
        } catch (...) {
            // A callback, allocation, or backend failure must not leave a
            // half-populated cache/graph in the long-lived worker.
            release_runtime_graphs_noexcept();
            throw;
        }
    } else {
        result.codes = ar_->generate(prompt, request.generation);
    }
    engine::debug::trace_log_scalar("fish_audio.generated.frames", result.codes.frames);
    engine::debug::trace_log_scalar("fish_audio.generated.codebooks", result.codes.codebooks);
    engine::debug::timing_log_scalar(
        "fish_audio.ar_generate_ms",
        engine::debug::elapsed_ms(ar_start, Clock::now()));

    if (result.codes.frames <= 0) {
        release_runtime_graphs_noexcept();
        throw std::runtime_error("Fish Audio generated no audio frames");
    }

    try {
        if (callback && !defer_final_audio) {
            const auto decode_start = Clock::now();
            result.audio = codec_->decode_codes(result.codes);
            codec_final_decode_ms = engine::debug::elapsed_ms(decode_start, Clock::now());
            codec_decode_ms = codec_stream_decode_ms + codec_final_decode_ms;
        } else if (callback) {
            codec_decode_ms = codec_stream_decode_ms;
        } else {
            const auto decode_start = Clock::now();
            result.audio = codec_->decode_codes(result.codes);
            codec_decode_ms = engine::debug::elapsed_ms(decode_start, Clock::now());
            codec_final_decode_ms = codec_decode_ms;
        }
    } catch (...) {
        release_runtime_graphs_noexcept();
        throw;
    }
    result.codec_stream_decode_ms = codec_stream_decode_ms;
    result.codec_final_decode_ms = codec_final_decode_ms;
    engine::debug::timing_log_scalar("fish_audio.codec_stream_decode_ms", codec_stream_decode_ms);
    engine::debug::timing_log_scalar("fish_audio.codec_final_decode_ms", codec_final_decode_ms);
    engine::debug::timing_log_scalar("fish_audio.codec_decode_ms", codec_decode_ms);
    codec_->release_runtime_graphs();
    if (mem_saver) {
        ar_->release_runtime_graphs();
    }
    return result;
}

}  // namespace engine::models::fish_audio
