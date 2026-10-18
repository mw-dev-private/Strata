#pragma once
#include <atomic>
#include <cstdio>
#include <stdexcept>

namespace strata::batch_share {
enum class Owner : int { None = 0, Prefill = 1, Decode = 2 };
struct PoolState {
    bool enabled = false;
    bool trace = false;
    std::atomic<int> owner{0};
    std::atomic<unsigned long long> sequence{0};
    std::atomic<unsigned long long> prefill_calls{0}, decode_calls{0}, collisions{0};
};
// One pool on the supported single-GPU path, shared across these two translation units.
inline PoolState pool_state;
class PoolCall {
    bool held_ = false;
    Owner owner_;
    unsigned long long sequence_ = 0;
public:
    explicit PoolCall(Owner owner) : owner_(owner) {
        if (!pool_state.enabled) return;
        int idle = 0;
        if (!pool_state.owner.compare_exchange_strong(idle, (int) owner, std::memory_order_acq_rel)) {
            ++pool_state.collisions;
            std::fprintf(stderr, "BATCH_SHARE COLLISION current=%d requested=%d\n", idle, (int) owner);
            throw std::runtime_error("batch CPU share: concurrent expert-pool submission");
        }
        held_ = true;
        sequence_ = ++pool_state.sequence;
        if (owner == Owner::Prefill) ++pool_state.prefill_calls;
        else ++pool_state.decode_calls;
        if (pool_state.trace) std::fprintf(stderr, "BATCH_SHARE POOL_BEGIN seq=%llu owner=%d\n", sequence_, (int) owner_);
    }
    ~PoolCall() {
        if (!held_) return;
        if (pool_state.trace) std::fprintf(stderr, "BATCH_SHARE POOL_END seq=%llu owner=%d\n", sequence_, (int) owner_);
        pool_state.owner.store(0, std::memory_order_release);
    }
    PoolCall(const PoolCall&) = delete;
    PoolCall& operator=(const PoolCall&) = delete;
};
}
