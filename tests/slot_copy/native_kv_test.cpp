#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/core/conversation_snapshot.hpp"
#include "strata/core/conversation_file.hpp"
#include "strata/kernels/kv_q4.hpp"

#include <array>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <cstdlib>
#include <limits>
#include <vector>

using namespace strata::core;
using namespace strata::kernels;

namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}
void cuda_check(dpct::err0 e) {
    check(e == 0,"SYCL operation completed");
}

struct Fixture {
    ModelGeometry g;
    QsaState state;
    std::vector<void*> device, host;
    std::array<void*,5> sources{};
    std::array<size_t,5> sizes{};

    template<class T> void alloc(T*& p, size_t n, bool pinned = false) {
        if (!n) return;
        void* raw = nullptr;
        if (pinned) {
            /*
            DPCT1048: The original value cudaHostAllocMapped is not
            meaningful in the migrated code and was removed or replaced with 0.
            You may need to check the migrated code.
            */
            cuda_check(DPCT_CHECK_ERROR(raw = (void *)sycl::malloc_host(
                                            n, dpct::get_in_order_queue())));
            host.push_back(raw);
            void* mapped = nullptr;
            cuda_check(DPCT_CHECK_ERROR(mapped = (void *)raw));
            p = static_cast<T*>(mapped);
        } else {
            cuda_check(DPCT_CHECK_ERROR(raw = (void *)sycl::malloc_device(
                                            n, dpct::get_in_order_queue())));
                device.push_back(raw); p = static_cast<T *>(raw);
        }
    }
    Fixture(int fmt, int mode) {
        auto& st = state;
        st.kv_mode = mode; st.kv_int8 = fmt==kKvInt8; st.kv_q4 = fmt==kKvQ4; st.kv_hybrid = fmt==3;
        st.n_pages = 24; st.n_slots = mode ? 4 : st.n_pages;
        st.max_cells = st.n_pages * 4; st.idx_pooled_rows = mode==2 ? 2 : st.max_cells/4+2;
        const size_t per = fmt==kKvQ4 ? kv_q4_bytes_per_head((int)g.head_dim) : g.head_dim*((st.kv_int8 || st.kv_hybrid) ? 1:2);
        const size_t rows = st.max_cells*g.n_head_kv, slot_rows = st.n_slots*4*g.n_head_kv;
        sizes = {rows*per, rows*per, fmt==kKvInt8 ? rows*(g.head_dim/64)*2:0,
                 fmt==kKvInt8 ? rows*(g.head_dim/64)*2:0, (size_t)st.idx_pooled_rows*g.idx_key_dim*4};
        if (fmt==3) {
            sizes = {rows*per, rows*kv_q4_bytes_per_head((int)g.head_dim), rows*(g.head_dim/64)*2, 0,
                     (size_t)st.idx_pooled_rows*g.idx_key_dim*4};
            alloc(st.k_q,sizes[0]); alloc(st.v_q4,sizes[1]); alloc(st.k_scale,sizes[2]);
            sources[0]=st.k_q; sources[1]=st.v_q4; sources[2]=st.k_scale;
        } else if (fmt==kKvQ4) {
            alloc(st.k_q4,slot_rows*per); alloc(st.v_q4,slot_rows*per);
            if (mode) {alloc(st.host.k_q4,sizes[0],true); alloc(st.host.v_q4,sizes[1],true);}
            sources[0]=mode?st.host.k_q4:st.k_q4; sources[1]=mode?st.host.v_q4:st.v_q4;
        } else if (fmt==kKvInt8) {
            alloc(st.k_q,slot_rows*per); alloc(st.v_q,slot_rows*per);
            alloc(st.k_scale,slot_rows*(g.head_dim/64)*2); alloc(st.v_scale,slot_rows*(g.head_dim/64)*2);
            if (mode) {
                alloc(st.host.k_q,sizes[0],true); alloc(st.host.v_q,sizes[1],true);
                alloc(st.host.k_scale,sizes[2],true); alloc(st.host.v_scale,sizes[3],true);
            }
            sources[0]=mode?st.host.k_q:st.k_q; sources[1]=mode?st.host.v_q:st.v_q;
            sources[2]=mode?st.host.k_scale:st.k_scale; sources[3]=mode?st.host.v_scale:st.v_scale;
        } else {
            alloc(st.k_pool,slot_rows*per); alloc(st.v_pool,slot_rows*per);
            if (mode) {alloc(st.host.k_pool,sizes[0],true); alloc(st.host.v_pool,sizes[1],true);}
            sources[0]=mode?st.host.k_pool:st.k_pool; sources[1]=mode?st.host.v_pool:st.v_pool;
        }
        alloc(st.idx_pooled,sizes[4]); sources[4]=st.idx_pooled;
        alloc(st.page_table,st.n_pages*4);
        if (mode==1) {
            auto& m=st.map;
            m.page_table=st.page_table; m.n_blocks=st.n_pages; m.n_slots=st.n_slots;
            alloc(m.slot_block,st.n_slots*4); alloc(m.slot_stamp,st.n_slots*4); alloc(m.slot_ref,st.n_slots*4);
            alloc(m.miss_block,st.n_slots*4); alloc(m.miss_slot,st.n_slots*4); alloc(m.ctl,kKvCtlInts*4);
        } else if (mode==2) {
            kv_ring_table(st.page_table,st.n_pages,st.n_slots,nullptr);
        }
    }
    void fill(uint8_t salt) {
        for (size_t i=0;i<sources.size();++i) {
            std::vector<uint8_t> data(sizes[i]);
            for (size_t j=0;j<data.size();++j) data[j]=(uint8_t)(salt+i*31+j*7+j/257);
            if (!data.empty()) cuda_check(DPCT_CHECK_ERROR(
                dpct::get_in_order_queue()
                    .memcpy(sources[i], data.data(), data.size())
                    .wait()));
        }
    }
    void fill_after(uint8_t salt, int64_t first_dirty) {
        for (size_t i=0;i<sources.size();++i) {
            const size_t offset = i == 4 ? size_t(first_dirty/4)*g.idx_key_dim*4
                                         : (sizes[i]/size_t(state.max_cells))*size_t((first_dirty/4)*4);
            if (sizes[i] > offset)
                cuda_check(DPCT_CHECK_ERROR(
                    dpct::get_in_order_queue()
                        .memset(static_cast<uint8_t *>(sources[i]) + offset,
                                salt, sizes[i] - offset)
                        .wait()));
        }
    }
    ~Fixture() {
        for (void *p : device)
            DPCT_CHECK_ERROR(sycl::free(p, dpct::get_in_order_queue()));
        for (void *p : host) DPCT_CHECK_ERROR(sycl::free(p, dpct::get_in_order_queue()));
    }
};
bool equal(const ConversationKv& a,const ConversationKv& b) {
    return a.k==b.k && a.v==b.v && a.k_scale==b.k_scale && a.v_scale==b.v_scale && a.pooled==b.pooled;
}

} // namespace
int main() {
    const char* approved=std::getenv("STRATA_SLOT_COPY_GPU_APPROVED");
    if(!approved || std::string(approved)!="1") { std::fprintf(stderr,"GPU approval gate closed; no device queried\n"); return 2; }
    // This executable is built during preparation and must not be run before owner approval.
    for(int fmt : std::array<int,4>{kKvF16,kKvInt8,kKvQ4,3}) for(int mode : {0,1,2}) {
        if(fmt==3 && mode!=0) continue;
        Fixture source(fmt,mode), baseline(fmt,mode), candidate(fmt,mode);
        const bool index=mode!=2;
        for(int64_t upto : {0,1,3,4,5,63,64,65,95,96}) {
            std::string error;source.fill(17);baseline.fill(91);candidate.fill(123);
            dpct::get_current_device().queues_wait_and_throw();
            ConversationKv want,reference,got;
            check(conversation_kv_save(want,source.state,source.g,upto,index,error),"capture source");
            check(conversation_kv_restore(want,baseline.state,baseline.g,upto,index,error),"baseline restore");
            check(conversation_kv_copy(source.state,candidate.state,candidate.g,upto,index,error),"direct copy");
            dpct::get_current_device().queues_wait_and_throw();
            check(conversation_kv_save(reference,baseline.state,baseline.g,upto,index,error),"read baseline");
            check(conversation_kv_save(got,candidate.state,candidate.g,upto,index,error),"read candidate");
            check(equal(want,reference) && equal(reference,got),"all authoritative KV parts bitwise equal");
            uint64_t fingerprint=0;
            check(conversation_kv_verify(want,candidate.state,candidate.g,upto,index,fingerprint,error),"resident ring and authoritative bytes verify");
            check(!conversation_kv_copy(source.state,candidate.state,candidate.g,upto,index,error,true),"unsupported queued mode rejected");
            check(!conversation_kv_copy(source.state,candidate.state,candidate.g,-1,index,error),"negative extent rejected");
            if(fmt==kKvInt8) {
                candidate.state.kv_rot=!candidate.state.kv_rot;
                check(!conversation_kv_copy(source.state,candidate.state,candidate.g,upto,index,error),"different layout rejected");
                candidate.state.kv_rot=!candidate.state.kv_rot;
            }
        }
    }
    std::printf("direct KV parity: %d checks passed\n",checks);
}
