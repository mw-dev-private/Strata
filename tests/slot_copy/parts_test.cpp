#include "strata/core/slot_copy_parts.hpp"
#include "strata/core/slot_copy_trace.hpp"
#include <cassert>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>
using namespace strata::core;
int main() {
    int checks=0; std::string error;
    std::array<std::array<unsigned char,257>,5> a{},b{};
    slot_copy::Parts src{},dst{}; slot_copy::Sizes sizes{};
    for(int i=0;i<5;++i){src[i]=a[i].data();dst[i]=b[i].data();}
    auto copy=[](void*d,const void*s,size_t n,std::string&){std::memcpy(d,s,n);return true;};
    std::mt19937 rng(19);
    for(int k=0;k<200;++k) {
        for(int i=0;i<5;++i){sizes[i]=rng()%258; for(auto&v:a[i])v=rng(); b[i].fill(0xAB);}
        assert(slot_copy::transfer(src,dst,sizes,copy,error));++checks;
        for(int i=0;i<5;++i)for(size_t j=0;j<257;++j)assert(b[i][j]==(j<sizes[i]?a[i][j]:0xAB));
    }
    sizes.fill(4);auto bad=dst;bad[4]=nullptr;b[0].fill(0xAB);int calls=0;
    auto count=[&](void*d,const void*s,size_t n,std::string&e){++calls;return copy(d,s,n,e);};
    assert(!slot_copy::transfer(src,bad,sizes,count,error));assert(calls==0 && b[0][0]==0xAB);++checks;
    bad=dst;bad[0]=a[0].data()+1;assert(!slot_copy::transfer(src,bad,sizes,count,error));assert(calls==0);++checks;
    bad=dst;bad[0]=reinterpret_cast<void*>(std::numeric_limits<uintptr_t>::max()-1);
    assert(!slot_copy::transfer(src,bad,sizes,count,error));assert(calls==0);++checks;
    assert(slot_copy::transfer(src,src,sizes,count,error));assert(calls==0);++checks;
    sizes.fill(0);assert(slot_copy::transfer({}, {},sizes,count,error));assert(calls==0);++checks;
    sizes.fill(4);calls=0;
    auto fail=[&](void*d,const void*s,size_t n,std::string&e){if(++calls==2){e="injected failure";return false;}return copy(d,s,n,e);};
    assert(!slot_copy::transfer(src,dst,sizes,fail,error));assert(calls==2 && error=="injected failure");++checks;
    slot_trace::begin_leg("VRAM");assert(slot_trace::leg==0);
    slot_trace::begin_leg("GEN 2 1,2");
    {slot_trace::Context context(2,"test");slot_trace::Scope phase("host_test",4);phase.ok=true;}
    assert(slot_trace::slot==-1);++checks;
    std::cout<<"PASS "<<checks<<" host-only transfer/trace checks\n";
}
