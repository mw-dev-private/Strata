#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
namespace strata::core::slot_copy {
using Parts = std::array<void*,5>;
using Sizes = std::array<size_t,5>;
inline bool validate(const Parts& src, const Parts& dst, const Sizes& sizes, std::string& error) {
    for (size_t i=0;i<5;++i) {
        if (!sizes[i]) continue;
        const uintptr_t s=reinterpret_cast<uintptr_t>(src[i]), d=reinterpret_cast<uintptr_t>(dst[i]);
        if (!s || !d || sizes[i]>std::numeric_limits<uintptr_t>::max()-s || sizes[i]>std::numeric_limits<uintptr_t>::max()-d) {
            error="slot copy: missing buffer or address overflow"; return false;
        }
        for(size_t j=0;j<5;++j) {
            if (!sizes[j]) continue;
            const uintptr_t q=reinterpret_cast<uintptr_t>(src[j]);
            if (!q || sizes[j]>std::numeric_limits<uintptr_t>::max()-q) { error="slot copy: invalid source"; return false; }
            if (d<q+sizes[j] && q<d+sizes[i] && !(i==j && d==q)) { error="slot copy: overlapping state buffers"; return false; }
        }
    }
    return true;
}
template<class Copier> bool transfer(const Parts& src, const Parts& dst, const Sizes& sizes, Copier&& copy, std::string& error) {
    if(!validate(src,dst,sizes,error)) return false;
    for(size_t i=0;i<5;++i) if(sizes[i] && src[i]!=dst[i] && !copy(dst[i],src[i],sizes[i],error)) return false;
    return true;
}
} // namespace strata::core::slot_copy
