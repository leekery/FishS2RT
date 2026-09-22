#include "bf16-p2.cuh"
#ifdef GGML_CUDA_BF16_P2_AVAILABLE
#include "unary.cuh"
#include "ggml-backend-impl.h"
#include <array>
#include <atomic>
#include <climits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
bool flag(const char * name) { const char * v = getenv(name); return v && atoi(v) != 0; }
void driver(CUresult e) {
    if (e != CUDA_SUCCESS) { const char * s = nullptr; cuGetErrorString(e, &s); throw std::runtime_error(s ? s : "VMM error"); }
}
void runtime(cudaError_t e) { if (e != cudaSuccess) { throw std::runtime_error(cudaGetErrorString(e)); } }
uint16_t inverse(uint8_t sm, uint8_t delta, unsigned base) {
    return uint16_t((sm & 127) | ((sm & 128) << 8) | (((unsigned(delta) + base) & 255) << 7));
}
struct Storage {
    ggml_backend_buffer_type buft{};
    int device = 0;
    unsigned base = 0;
    int version = 2;
    ggml_backend_cuda_context * owner = nullptr;
    size_t count = 0, mapped_bytes = 0;
    size_t slots[3]{}, offsets[3]{};
    CUdeviceptr address = 0;
    CUmemGenericAllocationHandle handles[3]{};
    bool mapped[3]{};
    ~Storage() {
        if (!address) { return; }
        cudaSetDevice(device);
        // Buffers may still be referenced by queued kernels/captured graph replay.
        // Store destruction must follow runner destruction; synchronize as a last guard.
        cudaDeviceSynchronize();
        for (int i = 0; i < version; ++i) {
            if (mapped[i]) { cuMemUnmap(address + offsets[i], slots[i]); }
            if (handles[i]) { cuMemRelease(handles[i]); }
        }
        cuMemAddressFree(address, mapped_bytes);
    }
    uint8_t * plane(int i) const { return reinterpret_cast<uint8_t *>(address + offsets[i]); }
    size_t plane_bytes(int i) const { return version == 3 && i ? count/2 : count; }
};
const char * buffer_name(ggml_backend_buffer_type_t) { return "CUDA_BF16_P2_V1"; }
const char * buffer_name_p3(ggml_backend_buffer_type_t) { return "CUDA_BF16_P3_V1"; }
size_t alignment(ggml_backend_buffer_type_t) { return 256; }
ggml_backend_buffer_t no_alloc(ggml_backend_buffer_type_t, size_t) { return nullptr; }
void free_buffer(ggml_backend_buffer_t b) { delete static_cast<Storage *>(b->context); }
void * get_base(ggml_backend_buffer_t b) { return static_cast<Storage *>(b->context)->plane(0); }
ggml_status init_tensor(ggml_backend_buffer_t, ggml_tensor *) { return GGML_STATUS_SUCCESS; }
size_t logical_offset(const Storage & s, const ggml_tensor * t, size_t offset, size_t bytes) {
    const size_t n = uintptr_t(t->data) - uintptr_t(s.plane(0)) + offset;
    GGML_ASSERT(n <= 2*s.count && bytes <= 2*s.count - n);
    return n;
}
// Rare diagnostics/copies reconstruct logical bytes, including odd-byte ranges.
// No raw memcpy is allowed across this storage boundary. They synchronize all
// streams because the buffer API does not carry the producing stream.
void get_tensor(ggml_backend_buffer_t b, const ggml_tensor * t, void * out, size_t off, size_t bytes) {
    auto & s = *static_cast<Storage *>(b->context); if (!bytes) { return; }
    ggml_cuda_set_device(s.device); CUDA_CHECK(cudaDeviceSynchronize());
    off = logical_offset(s, t, off, bytes);
    const size_t first = off/2, count = (off + bytes + 1)/2 - first;
    std::vector<uint8_t> sm(count), delta(count); std::vector<uint16_t> raw(count);
    CUDA_CHECK(cudaMemcpy(sm.data(), s.plane(0) + first, count, cudaMemcpyDeviceToHost));
    if (s.version == 2) {
        CUDA_CHECK(cudaMemcpy(delta.data(), s.plane(1) + first, count, cudaMemcpyDeviceToHost));
    } else {
        const size_t start_pair=first/2, pairs=(first+count+1)/2-start_pair;
        std::vector<uint8_t> lo(pairs), hi(pairs);
        CUDA_CHECK(cudaMemcpy(lo.data(),s.plane(1)+start_pair,pairs,cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hi.data(),s.plane(2)+start_pair,pairs,cudaMemcpyDeviceToHost));
        for(size_t i=0;i<count;++i) { const size_t j=(first+i)/2-start_pair;const unsigned shift=((first+i)&1)*4;
            delta[i]=uint8_t(((lo[j]>>shift)&15)|(((hi[j]>>shift)&15)<<4)); }
    }
    for (size_t i = 0; i < count; ++i) { raw[i] = inverse(sm[i], delta[i], s.base); }
    memcpy(out, reinterpret_cast<uint8_t *>(raw.data()) + off%2, bytes);
}
__global__ void initialize_plane(const uint8_t * src, uint8_t * dst, size_t n) {
    for (size_t i = size_t(blockIdx.x)*256 + threadIdx.x; i < n; i += size_t(gridDim.x)*256) { dst[i] = src[i]; }
}
// The representation/base is fixed. Updates are deliberately synchronous and
// preserve boundary BF16 words; intended for diagnostics, not concurrent inference.
void set_tensor(ggml_backend_buffer_t b, ggml_tensor * t, const void * input, size_t off, size_t bytes) {
    auto & s = *static_cast<Storage *>(b->context); if (!bytes) { return; }
    ggml_cuda_set_device(s.device); CUDA_CHECK(cudaDeviceSynchronize());
    const size_t abs = logical_offset(s, t, off, bytes);
    // P3 updates whole pairs, preserving a neighbouring BF16 word and both
    // packed nibbles for arbitrary odd-byte/odd-word logical writes.
    const size_t first = s.version == 3 ? (abs/4)*2 : abs/2;
    const size_t end = s.version == 3 ? ((abs+bytes+3)/4)*2 : (abs+bytes+1)/2;
    const size_t count = end-first;
    std::vector<uint16_t> raw(count);
    ggml_tensor root = *t; root.data = s.plane(0);
    get_tensor(b, &root, raw.data(), first*2, count*2);
    memcpy(reinterpret_cast<uint8_t *>(raw.data()) + abs-first*2, input, bytes);
    std::vector<uint8_t> sm(count), delta(count);
    for (size_t i = 0; i < count; ++i) { sm[i] = uint8_t((raw[i]&127)|((raw[i]>>8)&128)); delta[i] = uint8_t((raw[i]>>7)-s.base); }
    uint8_t * staging = nullptr; CUDA_CHECK(cudaMalloc(&staging, count));
    std::vector<uint8_t> packed(count/2);
    for (int p = 0; p < s.version; ++p) {
        const uint8_t * source=p?delta.data():sm.data();size_t n=count,start=first;
        if(s.version==3 && p) { n=count/2;start=first/2;const unsigned shift=p==1?0:4;
            for(size_t i=0;i<n;++i) { packed[i]=uint8_t(((delta[2*i]>>shift)&15)|(((delta[2*i+1]>>shift)&15)<<4)); }
            source=packed.data(); }
        CUDA_CHECK(cudaMemcpy(staging, source, n, cudaMemcpyHostToDevice));
        initialize_plane<<<256,256>>>(staging, s.plane(p) + start, n);
        CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    }
    CUDA_CHECK(cudaFree(staging));
}
void memset_tensor(ggml_backend_buffer_t b, ggml_tensor * t, uint8_t value, size_t off, size_t n) {
    std::vector<uint8_t> bytes(n, value); set_tensor(b, t, bytes.data(), off, n);
}
void clear_buffer(ggml_backend_buffer_t b, uint8_t value) {
    auto & s = *static_cast<Storage *>(b->context); ggml_tensor t{}; t.data = s.plane(0);
    memset_tensor(b, &t, value, 0, 2*s.count);
}
__device__ __forceinline__ nv_bfloat162 pair_at(const uint16_t * sm, const uint16_t * delta, size_t i, unsigned base) {
    const unsigned mixed = __byte_perm(sm[i], __vadd4(delta[i], base*0x0101u), 0x5140);
    const unsigned bits = (mixed&0x007f007f) | ((mixed&0x00800080)<<8) | ((mixed&0xff00ff00)>>1);
    nv_bfloat162 v; v.x = __ushort_as_bfloat16(uint16_t(bits)); v.y = __ushort_as_bfloat16(uint16_t(bits>>16)); return v;
}
__global__ void inverse_pairs(const uint16_t * sm, const uint16_t * delta, nv_bfloat162 * out, size_t offset, size_t count, unsigned base) {
    for (size_t i = size_t(blockIdx.x)*256 + threadIdx.x; i < count; i += size_t(gridDim.x)*256) { out[i] = pair_at(sm, delta, offset+i, base); }
}
__device__ __forceinline__ nv_bfloat162 pair_at_p3(const uint16_t * sm, const uint8_t * lo, const uint8_t * hi, size_t i, unsigned base) {
    const unsigned l=lo[i], h=hi[i];
    const unsigned deltas=(l&15u)|((h&15u)<<4)|((l&240u)<<4)|((h&240u)<<8);
    const unsigned mixed=__byte_perm(sm[i],__vadd4(deltas,base*0x0101u),0x5140);
    const unsigned bits=(mixed&0x007f007fu)|((mixed&0x00800080u)<<8)|((mixed&0xff00ff00u)>>1);
    nv_bfloat162 v;v.x=__ushort_as_bfloat16(uint16_t(bits));v.y=__ushort_as_bfloat16(uint16_t(bits>>16));return v;
}
__global__ void inverse_pairs_p3(const uint16_t * sm,const uint8_t * lo,const uint8_t * hi,nv_bfloat162 * out,size_t offset,size_t count,unsigned base) {
    for(size_t i=size_t(blockIdx.x)*256+threadIdx.x;i<count;i+=size_t(gridDim.x)*256) { out[i]=pair_at_p3(sm,lo,hi,offset+i,base); }
}
template<int mode>
__global__ void matvec(const uint16_t * sm, const uint16_t * delta, const float2 * y, float * dst, int cols, int rows, unsigned base) {
    const int tid = threadIdx.x, row = blockIdx.x;
    __shared__ float buf[mode == 2 ? 64 : 32];
    if (tid < 32) { buf[tid] = 0.0f; if constexpr(mode == 2) { buf[32+tid] = 0.0f; } }
    __syncthreads(); float sum = 0.0f, gate = 0.0f;
    for (int col2 = tid; col2 < cols/2; col2 += 256) {
        const size_t i = size_t(row)*(cols/2)+col2;
        const nv_bfloat162 a = pair_at(sm, delta, i + (mode == 2 ? size_t(rows)*(cols/2) : 0), base);
        const float2 x = y[col2];
        ggml_cuda_mad(sum, a.x, x.x); ggml_cuda_mad(sum, a.y, x.y);
        if constexpr(mode == 2) { const nv_bfloat162 g = pair_at(sm, delta, i, base); ggml_cuda_mad(gate, g.x, x.x); ggml_cuda_mad(gate, g.y, x.y); }
    }
    sum = warp_reduce_sum<32>(sum); if constexpr(mode == 2) { gate = warp_reduce_sum<32>(gate); }
    buf[tid/32] = sum; if constexpr(mode == 2) { buf[32+tid/32] = gate; }
    __syncthreads();
    if (tid < 32) { sum = warp_reduce_sum<32>(buf[tid]); if constexpr(mode == 2) { gate = warp_reduce_sum<32>(buf[32+tid]); } }
    __syncthreads();
    if (tid == 0) {
        if constexpr(mode == 1) { sum = __bfloat162float(__float2bfloat16(sum)); }
        if constexpr(mode == 2) {
            const float g = __bfloat162float(__float2bfloat16(gate)), u = __bfloat162float(__float2bfloat16(sum));
            const float s = __bfloat162float(__float2bfloat16(ggml_cuda_op_silu_single(g)));
            sum = __bfloat162float(__float2bfloat16(s*u));
        }
        dst[row] = sum;
    }
}
// Separate instantiation retains the accepted P2 kernel unchanged. Only BF16
// pair reconstruction differs; every FMA, reduction and explicit round is equal.
template<int mode>
__global__ void matvec_p3(const uint16_t * sm,const uint8_t * lo,const uint8_t * hi,const float2 * y,float * dst,int cols,int rows,unsigned base) {
    const int tid=threadIdx.x,row=blockIdx.x;
    __shared__ float buf[mode==2?64:32];
    if(tid<32) {buf[tid]=0.0f;if constexpr(mode==2){buf[32+tid]=0.0f;}}
    __syncthreads();float sum=0.0f,gate=0.0f;
    for(int col2=tid;col2<cols/2;col2+=256) {
        const size_t i=size_t(row)*(cols/2)+col2;
        const nv_bfloat162 a=pair_at_p3(sm,lo,hi,i+(mode==2?size_t(rows)*(cols/2):0),base);
        const float2 x=y[col2];
        ggml_cuda_mad(sum,a.x,x.x);ggml_cuda_mad(sum,a.y,x.y);
        if constexpr(mode==2) {const nv_bfloat162 g=pair_at_p3(sm,lo,hi,i,base);ggml_cuda_mad(gate,g.x,x.x);ggml_cuda_mad(gate,g.y,x.y);}
    }
    sum=warp_reduce_sum<32>(sum);if constexpr(mode==2){gate=warp_reduce_sum<32>(gate);}
    buf[tid/32]=sum;if constexpr(mode==2){buf[32+tid/32]=gate;}
    __syncthreads();
    if(tid<32){sum=warp_reduce_sum<32>(buf[tid]);if constexpr(mode==2){gate=warp_reduce_sum<32>(buf[32+tid]);}}
    __syncthreads();
    if(tid==0) {
        if constexpr(mode==1){sum=__bfloat162float(__float2bfloat16(sum));}
        if constexpr(mode==2){
            const float g=__bfloat162float(__float2bfloat16(gate)),u=__bfloat162float(__float2bfloat16(sum));
            const float s=__bfloat162float(__float2bfloat16(ggml_cuda_op_silu_single(g)));
            sum=__bfloat162float(__float2bfloat16(s*u));
        }
        dst[row]=sum;
    }
}
}

// Historical P2-family API also recognizes the explicitly versioned P3 buffer.
bool ggml_cuda_buft_is_bf16_p2(ggml_backend_buffer_type_t b) { return b && (b->iface.get_name == buffer_name || b->iface.get_name == buffer_name_p3); }
bool ggml_cuda_is_bf16_p2(const ggml_tensor * t) { return t && t->buffer && ggml_cuda_buft_is_bf16_p2(t->buffer->buft); }
bool ggml_cuda_bf16_p2_on_device(const ggml_tensor * t, int device) {
    return ggml_cuda_is_bf16_p2(t) && static_cast<Storage *>(t->buffer->context)->device == device;
}
bool ggml_cuda_bf16_p2_supports(const ggml_tensor * w, const ggml_tensor * x, const ggml_tensor * dst) {
    return ggml_cuda_is_bf16_p2(w) && !w->view_src && w->type == GGML_TYPE_BF16 && w->ne[0] > 0 && w->ne[0]%512 == 0 &&
        w->ne[0] <= INT_MAX && w->ne[1] > 0 && w->ne[1] <= INT_MAX/w->ne[0] && w->ne[2] == 1 && w->ne[3] == 1 &&
        x && dst && x->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 && x->ne[0] == w->ne[0] &&
        ggml_nrows(x) == 1 && ggml_nrows(dst) == 1 && ggml_is_contiguous(w) && ggml_is_contiguous(x) && ggml_is_contiguous(dst);
}
bool ggml_cuda_bf16_p2_can_materialize(const ggml_tensor * w, const ggml_tensor * x, const ggml_tensor * dst) {
    if (!ggml_cuda_is_bf16_p2(w) || w->view_src || !x || !dst) { return false; }
    const auto & s = *static_cast<Storage *>(w->buffer->context);
    return s.version == 2 && s.owner && s.owner->bf16_p2_scratch && s.owner->bf16_p2_scratch_bytes >= ggml_nbytes(w) &&
        x->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 && ggml_is_contiguous(w) && ggml_is_contiguous(x) &&
        ggml_is_contiguous(dst) && x->ne[0] == w->ne[0] && x->ne[1] > 0 && x->ne[2] == 1 && x->ne[3] == 1 &&
        dst->ne[0] == w->ne[1] && dst->ne[1] == x->ne[1] && dst->ne[2] == 1 && dst->ne[3] == 1;
}
bool ggml_cuda_bf16_p2_prepare_scratch(ggml_backend_t backend, size_t bytes) {
    // No resize after publication/capture. Caller must opt in before loading.
    if (!backend || !ggml_backend_is_cuda(backend) || !bytes || bytes > 128*1024*1024 || flag("GGML_CUDA_DISABLE_BF16_P2")) { return false; }
    auto & ctx = *static_cast<ggml_backend_cuda_context *>(backend->context);
    if (ctx.bf16_p2_scratch) { return bytes <= ctx.bf16_p2_scratch_bytes; }
    ggml_cuda_set_device(ctx.device);
    auto b = ggml_backend_buft_alloc_buffer(ggml_backend_cuda_buffer_type(ctx.device),bytes);
    if (!b) { cudaGetLastError(); GGML_LOG_WARN("BF16 P2 scratch allocation failed; Slow weights stay raw\n"); return false; }
    ctx.bf16_p2_scratch=b;ctx.bf16_p2_scratch_bytes=bytes;
    GGML_LOG_INFO("BF16 P2 prefill scratch: bytes=%zu single-stream=1\n",bytes);return true;
}
ggml_tensor ggml_cuda_bf16_p2_materialize(ggml_backend_cuda_context & ctx, const ggml_tensor * w) {
    GGML_ASSERT(ggml_cuda_is_bf16_p2(w) && !w->view_src && ctx.curr_stream_no == 0);
    const auto & s = *static_cast<Storage *>(w->buffer->context);
    GGML_ASSERT(s.version == 2 && s.owner == &ctx && ctx.bf16_p2_scratch && ctx.bf16_p2_scratch_bytes >= ggml_nbytes(w));
    ggml_tensor shadow = *w;
    shadow.buffer=ctx.bf16_p2_scratch;shadow.data=ggml_backend_buffer_get_base(shadow.buffer);
    shadow.view_src=nullptr;shadow.view_offs=0;shadow.extra=nullptr;
    inverse_pairs<<<1024,256,0,ctx.stream()>>>(reinterpret_cast<const uint16_t *>(s.plane(0)),reinterpret_cast<const uint16_t *>(s.plane(1)),
        static_cast<nv_bfloat162 *>(shadow.data),0,s.count/2,s.base);
    static const bool debug=flag("GGML_CUDA_DEBUG_BF16_P2");
    if(debug) { GGML_LOG_INFO("BF16 P2 materialize: bytes=%zu stream=0\n",ggml_nbytes(w)); }
    return shadow;
}
void ggml_cuda_bf16_p2_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * w, const ggml_tensor * x, ggml_tensor * dst, int mode) {
    GGML_ASSERT(ggml_cuda_bf16_p2_supports(w,x,dst) && ggml_cuda_bf16_p2_on_device(w,ctx.device));
    GGML_ASSERT((mode == 2 ? 2*dst->ne[0] : dst->ne[0]) == w->ne[1]);
    const auto & s = *static_cast<Storage *>(w->buffer->context);
    const auto a = reinterpret_cast<const uint16_t *>(s.plane(0)), b = reinterpret_cast<const uint16_t *>(s.plane(1));
    const auto input = static_cast<const float2 *>(x->data); auto output = static_cast<float *>(dst->data);
    const int cols = int(w->ne[0]), rows = int(dst->ne[0]);
    if (s.version == 3) {
        if(mode==0){matvec_p3<0><<<rows,256,0,ctx.stream()>>>(a,s.plane(1),s.plane(2),input,output,cols,rows,s.base);}
        else if(mode==1){matvec_p3<1><<<rows,256,0,ctx.stream()>>>(a,s.plane(1),s.plane(2),input,output,cols,rows,s.base);}
        else {GGML_ASSERT(mode==2);matvec_p3<2><<<rows,256,0,ctx.stream()>>>(a,s.plane(1),s.plane(2),input,output,cols,rows,s.base);}
    }
    else if (mode == 0) { matvec<0><<<rows,256,0,ctx.stream()>>>(a,b,input,output,cols,rows,s.base); }
    else if (mode == 1) { matvec<1><<<rows,256,0,ctx.stream()>>>(a,b,input,output,cols,rows,s.base); }
    else { GGML_ASSERT(mode == 2); matvec<2><<<rows,256,0,ctx.stream()>>>(a,b,input,output,cols,rows,s.base); }
    static const bool debug = flag("GGML_CUDA_DEBUG_BF16_P2");
    if (debug) { GGML_LOG_INFO("BF16 P%d dispatch: mode=%d rows=%d cols=%d (capture dispatch, not replay count)\n",s.version,mode,rows,cols); }
}

static ggml_backend_buffer_t create_version(ggml_backend_t backend, ggml_tensor * t, const void * raw, size_t bytes,int version) {
    if (!backend || !ggml_backend_is_cuda(backend) || !t || !raw || t->data || t->buffer || t->view_src || t->type != GGML_TYPE_BF16 ||
        t->ne[0] <= 0 || t->ne[0]%512 || t->ne[1] <= 0 || t->ne[1] > INT_MAX/t->ne[0] ||
        t->ne[2] != 1 || t->ne[3] != 1 || !ggml_is_contiguous(t) || bytes != ggml_nbytes(t) ||
        flag("GGML_CUDA_DISABLE_BF16_P2")) { return nullptr; }
    try {
        auto s = std::make_unique<Storage>();s->version=version;s->owner = static_cast<ggml_backend_cuda_context *>(backend->context);s->device = s->owner->device;
        runtime(cudaSetDevice(s->device)); runtime(cudaFree(nullptr)); driver(cuInit(0));
        int compression = 0, vmm = 0; driver(cuDeviceGetAttribute(&compression,CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED,s->device));
        driver(cuDeviceGetAttribute(&vmm,CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED,s->device));
        if (!compression || !vmm || ggml_cuda_info().devices[s->device].cc < GGML_CUDA_CC_ADA_LOVELACE) { return nullptr; }
        s->count = bytes/2; const auto words = static_cast<const uint16_t *>(raw);
        std::array<size_t,256> histogram{}; for (size_t i=0;i<s->count;++i) { ++histogram[(words[i]>>7)&255]; }
        size_t best = 0; for (unsigned e=0;e<=241;++e) { size_t n=0; for(unsigned j=0;j<15;++j) { n+=histogram[e+j]; } if(n>best) {best=n;s->base=e;} }
        for(unsigned bits=0;bits<65536;++bits) { if(inverse(uint8_t((bits&127)|((bits>>8)&128)),uint8_t((bits>>7)-s->base),s->base)!=bits) { throw std::runtime_error("codebook inverse mismatch"); } }
        CUmemAllocationProp prop{}; prop.type=CU_MEM_ALLOCATION_TYPE_PINNED;prop.location.type=CU_MEM_LOCATION_TYPE_DEVICE;prop.location.id=s->device;
        prop.allocFlags.compressionType=CU_MEM_ALLOCATION_COMP_GENERIC;
        size_t gran=0;driver(cuMemGetAllocationGranularity(&gran,&prop,CU_MEM_ALLOC_GRANULARITY_MINIMUM));
        for(int p=0;p<version;++p) {
            s->offsets[p]=s->mapped_bytes;s->slots[p]=(s->plane_bytes(p)+gran-1)/gran*gran;s->mapped_bytes+=s->slots[p];
        }
        driver(cuMemAddressReserve(&s->address,s->mapped_bytes,0,0,0));
        for(int p=0;p<version;++p) {
            driver(cuMemCreate(&s->handles[p],s->slots[p],&prop,0)); CUmemAllocationProp actual{};
            driver(cuMemGetAllocationPropertiesFromHandle(&actual,s->handles[p]));
            if(actual.allocFlags.compressionType!=CU_MEM_ALLOCATION_COMP_GENERIC) { throw std::runtime_error("GENERIC not granted"); }
            driver(cuMemMap(s->address+s->offsets[p],s->slots[p],0,s->handles[p],0));s->mapped[p]=true;
            CUmemAccessDesc access{};access.location=prop.location;access.flags=CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            driver(cuMemSetAccess(s->address+s->offsets[p],s->slots[p],&access,1));
        }
        // Bounded 4 MiB startup staging, never a full raw device duplicate.
        constexpr size_t chunk=4*1024*1024; uint8_t * staging=nullptr;
        runtime(cudaMalloc(&staging,chunk));
        std::unique_ptr<uint8_t,decltype(&cudaFree)> guard(staging,cudaFree);
        std::vector<uint8_t> host(chunk);
        for(int p=0;p<version;++p) { for(size_t off=0;off<s->plane_bytes(p);off+=chunk) {
            const size_t n=std::min(chunk,s->plane_bytes(p)-off);
            if(version==3 && p) {
                const unsigned shift=p==1?0:4;
                for(size_t i=0;i<n;++i) {const uint8_t d0=uint8_t((words[2*(off+i)]>>7)-s->base),d1=uint8_t((words[2*(off+i)+1]>>7)-s->base);
                    host[i]=uint8_t(((d0>>shift)&15)|(((d1>>shift)&15)<<4));}
            } else {
                for(size_t i=0;i<n;++i) { unsigned b=words[off+i];host[i]=p?uint8_t((b>>7)-s->base):uint8_t((b&127)|((b>>8)&128)); }
            }
            runtime(cudaMemcpy(staging,host.data(),n,cudaMemcpyHostToDevice));
            initialize_plane<<<256,256>>>(staging,s->plane(p)+off,n);runtime(cudaGetLastError());runtime(cudaDeviceSynchronize());
        } }
        if(flag("GGML_CUDA_VERIFY_BF16_P2")) {
            for(size_t off=0;off<bytes;off+=chunk) {
                const size_t n=std::min(chunk,bytes-off);
                if(version==3) {inverse_pairs_p3<<<256,256>>>(reinterpret_cast<uint16_t *>(s->plane(0)),s->plane(1),s->plane(2),reinterpret_cast<nv_bfloat162 *>(staging),off/4,n/4,s->base);}
                else {inverse_pairs<<<256,256>>>(reinterpret_cast<uint16_t *>(s->plane(0)),reinterpret_cast<uint16_t *>(s->plane(1)),reinterpret_cast<nv_bfloat162 *>(staging),off/4,n/4,s->base);}
                runtime(cudaGetLastError());runtime(cudaMemcpy(host.data(),staging,n,cudaMemcpyDeviceToHost));
                if(memcmp(host.data(),static_cast<const uint8_t *>(raw)+off,n)) { throw std::runtime_error("GPU weight inverse mismatch"); }
            }
        }
        ggml_backend_buffer_type_i ti{};ti.get_name=version==3?buffer_name_p3:buffer_name;ti.alloc_buffer=no_alloc;ti.get_alignment=alignment;
        s->buft={ti,ggml_backend_get_device(backend),nullptr};
        ggml_backend_buffer_i bi{};bi.free_buffer=free_buffer;bi.get_base=get_base;bi.init_tensor=init_tensor;
        bi.memset_tensor=memset_tensor;bi.set_tensor=set_tensor;bi.get_tensor=get_tensor;bi.clear=clear_buffer;
        auto b=ggml_backend_buffer_init(&s->buft,bi,s.get(),s->mapped_bytes);
        if(!b) { throw std::runtime_error("buffer init failed"); }
        t->buffer=b;t->data=s->plane(0);ggml_backend_buffer_set_usage(b,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        GGML_LOG_INFO("BF16 P%d storage: rows=%lld cols=%lld base=%u logical=%zu mapped=%zu grants=%s codebook=65536/65536 weight_verify=%d\n",version,(long long)t->ne[1],(long long)t->ne[0],s->base,bytes,s->mapped_bytes,version==3?"1,1,1":"1,1",int(flag("GGML_CUDA_VERIFY_BF16_P2")));
        s.release();return b;
    } catch(const std::exception & e) {
        // Allocation failure is recoverable; do not leak the sticky runtime
        // error into a later ordinary-buffer CUDA_CHECK(cudaGetLastError()).
        cudaGetLastError();
        GGML_LOG_WARN("BF16 P%d factory declined; caller may fall back: %s\n",version,e.what()); return nullptr;
    }
}
ggml_backend_buffer_t ggml_cuda_bf16_p2_create(ggml_backend_t backend,ggml_tensor * t,const void * raw,size_t bytes) {
    return create_version(backend,t,raw,bytes,2);
}
ggml_backend_buffer_t ggml_cuda_bf16_p3_create(ggml_backend_t backend,ggml_tensor * t,const void * raw,size_t bytes) {
    if(!flag("GGML_CUDA_DISABLE_BF16_P3")) {
        if(auto b=create_version(backend,t,raw,bytes,3)) {return b;}
    }
    // The unsuccessful factory leaves tensor/data unchanged and its partial
    // VMM state is RAII-released before the accepted P2 path is attempted.
    return ggml_cuda_bf16_p2_create(backend,t,raw,bytes);
}
#endif
