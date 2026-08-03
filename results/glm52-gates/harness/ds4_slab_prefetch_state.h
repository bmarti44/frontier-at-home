#ifndef DS4_SLAB_PREFETCH_STATE_H
#define DS4_SLAB_PREFETCH_STATE_H

// Default-fail interface frozen by test before the asynchronous implementation.
// The completed implementation is the sole authority for slot publication,
// ownership, completion, recycling, and integrity invalidation in ds4_cuda.cu.

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ds4_slab_prefetch {

enum class State : uint8_t { empty, reading, ready, main_owned, copying };

struct Identity {
    uint64_t key = 0;
    uint64_t gate_offset = 0;
    uint64_t up_offset = 0;
    uint64_t down_offset = 0;
    uint64_t expected_length = 0;
    uint64_t model_generation = 0;
    std::array<uint8_t, 32> expected_sha256{};
};

struct Lease {
    size_t slot = 0;
    uint64_t token = 0;
    bool valid = false;
};

struct Telemetry {
    uint64_t attempts = 0;
    uint64_t sha_successes = 0;
    uint64_t sha_failures = 0;
    uint64_t ready = 0;
    uint64_t late = 0;
    uint64_t stale = 0;
    uint64_t fallback = 0;
    uint64_t copies = 0;
    uint64_t validated_bytes = 0;
    uint64_t copied_bytes = 0;
    uint64_t publications = 0;
};

struct ReadResult { bool ok = false; size_t actual_length = 0; };

class Backend {
public:
    virtual ~Backend() = default;
    virtual ReadResult read(const Identity &, uint8_t *, size_t) = 0;
    virtual bool sha256(const uint8_t *, size_t, uint8_t out[32]) = 0;
    virtual bool copy_sync(const uint8_t *, size_t) = 0;
    virtual bool copy_async(const uint8_t *, size_t, uint64_t *event) = 0;
    virtual bool event_complete(uint64_t event) = 0;
    virtual void invalidate_slab() = 0;
};

class Ring {
public:
    Ring(size_t, uint64_t, const std::vector<uint8_t *> &,
         const std::vector<size_t> &) {}
    Lease issue(const Identity &) { return {}; }
    bool complete_read(Lease, Backend &) { return false; }
    Lease claim(const Identity &) { return {}; }
    bool copy_sync(Lease, Backend &) { return false; }
    bool copy_async(Lease, Backend &) { return false; }
    bool poll(Lease, Backend &) { return false; }
    void reload(uint64_t) {}
    bool invalidated() const { return false; }
    bool demand_fallback_allowed() const { return true; }
    State state(size_t) const { return State::empty; }
    uint64_t lease_token(size_t) const { return 0; }
    Telemetry telemetry() const { return {}; }
};

}  // namespace ds4_slab_prefetch

#endif
