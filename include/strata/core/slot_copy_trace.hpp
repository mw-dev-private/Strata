#pragma once
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <unistd.h>
namespace strata::core::slot_trace {
inline bool enabled() { static const bool on = [] { const char* e = std::getenv("STRATA_SLOT_COPY_TRACE"); return e && e[0] == '1'; }(); return on; }
inline uint64_t leg = 0;
inline int slot = -1;
inline const char* direction = "none";
inline bool direct() { static const bool on = [] { const char* e = std::getenv("STRATA_SLOT_COPY_DIRECT"); return e && e[0] == '1'; }(); return on; }
inline void point(const char* event) {
    if(!enabled()) return;
    const auto ns=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    std::fprintf(stderr,"SLOT_LEG pid=%d leg=%llu event=%s ns=%lld\n",int(getpid()),(unsigned long long)leg,event,(long long)ns);
}
inline void begin_leg(const std::string& command) {
    if (!enabled()) return;
    if (command.rfind("GEN ",0) && command.rfind("GENI ",0) && command.rfind("BGEN ",0) && command.rfind("BGENI ",0)) return;
    ++leg; slot = -1; direction = "none";
    point("begin");
}
class Context {
    int previous_slot; const char* previous_direction;
public:
    Context(int b, const char* d): previous_slot(slot), previous_direction(direction) { slot=b; direction=d; }
    ~Context() { slot=previous_slot; direction=previous_direction; }
};
class Scope {
    using Clock = std::chrono::steady_clock;
    const char* phase_; size_t bytes_; bool on_; Clock::time_point start_;
public:
    bool ok = false;
    Scope(const char* phase, size_t bytes=0): phase_(phase), bytes_(bytes), on_(enabled()) { if(on_) start_=Clock::now(); }
    void bytes(size_t n) { bytes_=n; }
    ~Scope() {
        if (!on_) return;
        const auto end=Clock::now();
        const double ms=std::chrono::duration<double,std::milli>(end-start_).count();
        const auto ns=std::chrono::duration_cast<std::chrono::nanoseconds>(start_.time_since_epoch()).count();
        std::fprintf(stderr,"SLOT_COPY pid=%d leg=%llu slot=%d direction=%s phase=%s bytes=%zu ms=%.4f start_ns=%lld ok=%d direct=%d\n",int(getpid()),(unsigned long long)leg,slot,direction,phase_,bytes_,ms,(long long)ns,int(ok),int(direct()));
    }
};
template<class F> bool measured(const char* phase, F&& f) { Scope s(phase); s.ok=f(); return s.ok; }
} // namespace strata::core::slot_trace
