#pragma once
#include "common.cuh"

// Experimental immutable, single-device BF16 storage. It is NOT raw CUDA BF16.
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && !defined(GGML_CUDA_NO_VMM)
#define GGML_CUDA_BF16_P2_AVAILABLE
bool ggml_cuda_buft_is_bf16_p2(ggml_backend_buffer_type_t buft);
bool ggml_cuda_is_bf16_p2(const ggml_tensor * tensor);
bool ggml_cuda_bf16_p2_on_device(const ggml_tensor * tensor, int device);
bool ggml_cuda_bf16_p2_supports(const ggml_tensor * weight, const ggml_tensor * input, const ggml_tensor * dst);
bool ggml_cuda_bf16_p2_can_materialize(const ggml_tensor * weight, const ggml_tensor * input, const ggml_tensor * dst);
bool ggml_cuda_bf16_p2_prepare_scratch(ggml_backend_t backend, size_t bytes);
ggml_tensor ggml_cuda_bf16_p2_materialize(ggml_backend_cuda_context & ctx, const ggml_tensor * weight);
ggml_backend_buffer_t ggml_cuda_bf16_p2_create(ggml_backend_t backend, ggml_tensor * tensor, const void * raw, size_t bytes);
// Fast-only P3; failure/disable tries P2 before ordinary raw upload.
ggml_backend_buffer_t ggml_cuda_bf16_p3_create(ggml_backend_t backend, ggml_tensor * tensor, const void * raw, size_t bytes);
// mode 0: ordinary projection; 1: explicit final BF16 round; 2: rounded packed SwiGLU.
void ggml_cuda_bf16_p2_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * weight,
                             const ggml_tensor * input, ggml_tensor * dst, int mode);
#else
inline bool ggml_cuda_buft_is_bf16_p2(ggml_backend_buffer_type_t) { return false; }
inline bool ggml_cuda_is_bf16_p2(const ggml_tensor *) { return false; }
inline bool ggml_cuda_bf16_p2_on_device(const ggml_tensor *, int) { return false; }
#endif
