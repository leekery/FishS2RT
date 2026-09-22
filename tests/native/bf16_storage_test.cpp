// Software regression test for the actual backend buffer and graph dispatch.
// Exit nonzero on the first mismatch.
#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cuda.h>
#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using Factory = ggml_backend_buffer_t (*)(ggml_backend_t, ggml_tensor *, const void *, size_t);
void require(bool b, const char * why) { if (!b) { throw std::runtime_error(why); } }
uint32_t next(uint32_t & s) { s^=s<<13;s^=s>>17;s^=s<<5;return s; }
void check_bytes(ggml_tensor * t, const std::vector<uint16_t> & ref) {
    std::vector<uint16_t> got(ref.size());ggml_backend_tensor_get(t,got.data(),0,got.size()*2);
    require(got==ref,"logical buffer read mismatch");
}
void test_buffer(ggml_backend_t backend, Factory factory,const char * expected) {
    auto ctx=ggml_init({4*1024*1024,nullptr,true});
    auto t=ggml_new_tensor_2d(ctx,GGML_TYPE_BF16,512,128);
    std::vector<uint16_t> ref(65536);for(unsigned i=0;i<65536;++i)ref[i]=uint16_t(i);
    auto b=factory(backend,t,ref.data(),ref.size()*2);require(b!=nullptr,"P2 factory unavailable");
    require(strcmp(ggml_backend_buffer_name(b),expected)==0,"unexpected storage version/fallback");
    check_bytes(t,ref); // includes NaNs/infinities as bit patterns, not arithmetic
    const uint8_t edit[]={0x34,0x12,0xfe,0xdc,0,0x80,0x7f};
    ggml_backend_tensor_set_async(backend,t,edit,3,sizeof(edit));
    memcpy(reinterpret_cast<uint8_t *>(ref.data())+3,edit,sizeof(edit));check_bytes(t,ref);
    uint8_t readback[7]{};ggml_backend_tensor_get_async(backend,t,readback,3,sizeof(readback));ggml_backend_synchronize(backend);
    require(memcmp(edit,readback,sizeof(edit))==0,"odd async partial mismatch");
    uint8_t patch[3][5]={{1,2,3,4,5},{6,7,8,9,10},{11,12,13,14,15}};
    ggml_backend_tensor_set_2d_async(backend,t,patch,31,5,3,23,5);
    for(size_t i=0;i<3;++i)memcpy(reinterpret_cast<uint8_t *>(ref.data())+31+i*23,patch[i],5);
    check_bytes(t,ref);uint8_t copy[3][5]{};
    ggml_backend_tensor_get_2d_async(backend,t,copy,31,5,3,23,5);ggml_backend_synchronize(backend);
    require(memcmp(copy,patch,sizeof(copy))==0,"2D logical mismatch");
    auto view=ggml_view_1d(ctx,t,64,18);require(ggml_backend_view_init(view)==GGML_STATUS_SUCCESS,"view init failed");
    std::vector<uint16_t> vr(64);ggml_backend_tensor_get(view,vr.data(),0,128);
    require(memcmp(vr.data(),ref.data()+9,128)==0,"view logical offset mismatch");
    auto raw=ggml_new_tensor_2d(ctx,GGML_TYPE_BF16,512,128);auto ordinary=ggml_backend_alloc_ctx_tensors(ctx,backend);
    require(ordinary!=nullptr,"ordinary buffer allocation failed");
    ggml_backend_tensor_copy(t,raw);check_bytes(raw,ref);
    ggml_backend_tensor_copy(raw,t);check_bytes(t,ref);
    ggml_backend_tensor_memset(t,0xa5,19,101);memset(reinterpret_cast<uint8_t *>(ref.data())+19,0xa5,101);check_bytes(t,ref);
    ggml_backend_buffer_clear(b,0x6b);std::fill(ref.begin(),ref.end(),uint16_t(0x6b6b));check_bytes(t,ref);
    for(size_t offset:{size_t(0),size_t(1),size_t(2),size_t(3),size_t(4),size_t(131071)}) {
        const uint8_t value=uint8_t(offset^0xa7);ggml_backend_tensor_set(t,&value,offset,1);
        reinterpret_cast<uint8_t *>(ref.data())[offset]=value;check_bytes(t,ref);
    }
    if(strcmp(expected,"CUDA_BF16_P3_V1")==0) {
        auto multi=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,512,2);
        require(!ggml_backend_supports_op(backend,ggml_mul_mat(ctx,t,multi)),"P3 multi-token must be rejected");
    }
    ggml_backend_buffer_free(ordinary);ggml_backend_buffer_free(b);ggml_free(ctx);
    std::cout<<"buffer PASS: 65536 codes, odd get/set, async/2D, view, both copy directions, memset/clear\n";
}
ggml_tensor * projection(ggml_context * ctx,ggml_tensor * w,ggml_tensor * x,int rows,int mode) {
    auto mm=ggml_mul_mat(ctx,w,x);
    if(mode==0)return mm;
    if(mode==1)return ggml_round_bf16(ctx,ggml_reshape_1d(ctx,mm,rows));
    auto r=ggml_reshape_1d(ctx,mm,2*rows);
    auto g=ggml_round_bf16(ctx,ggml_view_1d(ctx,r,rows,0));
    auto u=ggml_round_bf16(ctx,ggml_view_1d(ctx,r,rows,rows*sizeof(float)));
    return ggml_round_bf16(ctx,ggml_mul(ctx,ggml_round_bf16(ctx,ggml_silu(ctx,g)),u));
}
size_t test_projection(ggml_backend_t backend,Factory factory,int cols,int rows,int mode) {
    auto ctx=ggml_init({8*1024*1024,nullptr,true});
    const int wr=mode==2?rows*2:rows;
    auto raw=ggml_new_tensor_2d(ctx,GGML_TYPE_BF16,cols,wr);
    auto p2=ggml_new_tensor_2d(ctx,GGML_TYPE_BF16,cols,wr);
    auto x=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,cols);
    uint32_t rng=0x5070c007u;std::vector<uint16_t> weights(size_t(cols)*wr);
    for(auto & w:weights) {unsigned u=next(rng);w=uint16_t((u&0x807f)|((110+(u>>8)%22)<<7));}
    auto pb=factory(backend,p2,weights.data(),weights.size()*2);require(pb!=nullptr,"projection factory failed");
    auto a=projection(ctx,raw,x,rows,mode),b=projection(ctx,p2,x,rows,mode);
    ggml_set_output(a);ggml_set_output(b);
    auto graph=ggml_new_graph_custom(ctx,128,false);ggml_build_forward_expand(graph,a);ggml_build_forward_expand(graph,b);
    auto arena=ggml_backend_alloc_ctx_tensors(ctx,backend);require(arena!=nullptr,"arena failed");
    ggml_backend_tensor_set(raw,weights.data(),0,weights.size()*2);
    std::vector<float> input(cols),out_a(rows),out_b(rows);
    for(int v=0;v<32;++v) {
        for(int i=0;i<cols;++i) {input[i]=v==0?0.0f:v==1?1.0f:v==2?(i%2?1.0f:-1.0f):float(int(next(rng)%65537)-32768)/32768.0f;}
        ggml_backend_tensor_set(x,input.data(),0,input.size()*4);
        require(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"compute failed");
        ggml_backend_tensor_get(a,out_a.data(),0,rows*4);ggml_backend_tensor_get(b,out_b.data(),0,rows*4);
        if(memcmp(out_a.data(),out_b.data(),rows*4)) {for(int r=0;r<rows;++r)if(memcmp(&out_a[r],&out_b[r],4)){std::cerr<<"first divergence mode="<<mode<<" K="<<cols<<" vector="<<v<<" row="<<r<<"\n";break;}throw std::runtime_error("F32 output mismatch");}
    }
    ggml_backend_synchronize(backend);ggml_backend_cuda_clear_graph(backend,graph);
    ggml_backend_buffer_free(arena);ggml_backend_buffer_free(pb);ggml_free(ctx);
    std::cout<<"projection PASS: mode="<<mode<<" K="<<cols<<" M="<<rows<<" vectors=32 bits="<<rows*32<<"\n";
    return size_t(rows)*32;
}
size_t test_prefill(ggml_backend_t backend,Factory factory,int cols,int rows,int tokens) {
    auto ctx=ggml_init({8*1024*1024,nullptr,true});
    auto raw=ggml_new_tensor_2d(ctx,GGML_TYPE_BF16,cols,rows),p2=ggml_new_tensor_2d(ctx,GGML_TYPE_BF16,cols,rows);
    auto raw_second=ggml_new_tensor_2d(ctx,GGML_TYPE_BF16,cols,rows);
    auto p2_second=ggml_new_tensor_2d(ctx,GGML_TYPE_BF16,cols,rows);
    auto x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,cols,tokens);
    uint32_t rng=0x50701234;std::vector<uint16_t> weights(size_t(cols)*rows),neg(weights.size());
    for(size_t i=0;i<weights.size();++i) {auto u=next(rng);weights[i]=uint16_t((u&0x807f)|((110+(u>>8)%22)<<7));neg[i]=weights[i]^0x8000;}
    auto pb=factory(backend,p2,weights.data(),weights.size()*2),pb2=factory(backend,p2_second,neg.data(),neg.size()*2);
    require(pb && pb2,"prefill factory failed");
    auto a=ggml_mul_mat(ctx,raw,x),b=ggml_mul_mat(ctx,p2,x),c=ggml_mul_mat(ctx,p2_second,x),d=ggml_mul_mat(ctx,raw_second,x);
    ggml_set_output(a);ggml_set_output(b);ggml_set_output(c);ggml_set_output(d);
    auto graph=ggml_new_graph_custom(ctx,128,false);ggml_build_forward_expand(graph,a);ggml_build_forward_expand(graph,b);ggml_build_forward_expand(graph,c);ggml_build_forward_expand(graph,d);
    auto arena=ggml_backend_alloc_ctx_tensors(ctx,backend);require(arena!=nullptr,"prefill arena failed");
    ggml_backend_tensor_set(raw,weights.data(),0,weights.size()*2);
    ggml_backend_tensor_set(raw_second,neg.data(),0,neg.size()*2);
    std::vector<float> input(size_t(cols)*tokens),oa(size_t(rows)*tokens),ob(oa.size()),oc(oa.size()),od(oa.size());
    for(int v=0;v<4;++v) {
        for(auto & f:input) { f=float(int(next(rng)%65537)-32768)/32768.0f; }
        ggml_backend_tensor_set(x,input.data(),0,input.size()*4);
        require(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"prefill compute failed");
        ggml_backend_tensor_get(a,oa.data(),0,oa.size()*4);ggml_backend_tensor_get(b,ob.data(),0,ob.size()*4);ggml_backend_tensor_get(c,oc.data(),0,oc.size()*4);ggml_backend_tensor_get(d,od.data(),0,od.size()*4);
        require(memcmp(oa.data(),ob.data(),oa.size()*4)==0,"prefill raw/P2 mismatch");
        // A second differently encoded matrix overwrites the same scratch. This
        // catches decode/consumer overlap. Compare to its own raw GEMM, also
        // preserving signed zero behavior without an algebraic sign shortcut.
        require(memcmp(oc.data(),od.data(),oc.size()*4)==0,"prefill scratch sequencing mismatch");
    }
    ggml_backend_synchronize(backend);ggml_backend_cuda_clear_graph(backend,graph);
    ggml_backend_buffer_free(arena);ggml_backend_buffer_free(pb);ggml_backend_buffer_free(pb2);ggml_free(ctx);
    std::cout<<"prefill PASS K="<<cols<<" M="<<rows<<" N="<<tokens<<" vectors=4 outputs="<<oa.size()*8<<"\n";
    return oa.size()*8;
}
int main(int argc,char ** argv) try {
    const std::string option=argc>1?argv[1]:"--p2";
    require(argc<=2 && (option=="--p2" || option=="--p3" || option=="--p3-fallback"),"expected --p2/--p3/--p3-fallback");
    if(option=="--p3-fallback") {
#ifdef _WIN32
        _putenv_s("GGML_CUDA_DISABLE_BF16_P3","1");
#else
        setenv("GGML_CUDA_DISABLE_BF16_P3","1",1);
#endif
    }
    const bool p3=option!="--p2",direct_p3=option=="--p3";
    auto backend=ggml_backend_cuda_init(0);require(backend!=nullptr,"CUDA init failed");
    auto factory=reinterpret_cast<Factory>(ggml_backend_reg_get_proc_address(ggml_backend_cuda_reg(),p3?"ggml_backend_cuda_bf16_p3_create_tensor":"ggml_backend_cuda_bf16_p2_create_tensor"));
    require(factory!=nullptr,"factory export absent");test_buffer(backend,factory,direct_p3?"CUDA_BF16_P3_V1":"CUDA_BF16_P2_V1");
    size_t compared=0;for(int cols:{2560,4096,9728})for(int mode=0;mode<3;++mode)compared+=test_projection(backend,factory,cols,128,mode);
    if(!direct_p3) {
    auto prepare=reinterpret_cast<bool (*)(ggml_backend_t,size_t)>(ggml_backend_reg_get_proc_address(ggml_backend_cuda_reg(),"ggml_backend_cuda_bf16_p2_prepare_scratch"));
    require(prepare && prepare(backend,2560*6144*2),"prefill scratch unavailable");
    for(int cols:{2560,4096,9728})for(int tokens:{2,7,8,9,32,129,513})compared+=test_prefill(backend,factory,cols,128,tokens);
    for(int tokens:{2,9,129})compared+=test_prefill(backend,factory,2560,6144,tokens);
    }
    ggml_backend_free(backend);std::cout<<"PASS F32 comparisons="<<compared<<" mismatches=0\n";return 0;
} catch(const std::exception & e) {std::cerr<<"FAIL: "<<e.what()<<"\n";return 1;}
