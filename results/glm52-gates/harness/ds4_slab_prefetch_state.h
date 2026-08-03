#ifndef DS4_SLAB_PREFETCH_STATE_H
#define DS4_SLAB_PREFETCH_STATE_H

// Sole authority for slot publication, ownership, completion, recycling, and
// integrity invalidation in the collision-resistant slab-prefetch path.

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
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
    Ring(size_t count, uint64_t generation,
         const std::vector<uint8_t *> &buffers,
         const std::vector<size_t> &capacities)
        : generation_(generation), usable_(count >= 4 && count <= 8 &&
              buffers.size() == count && capacities.size() == count) {
        if (!usable_) return;
        slots_.resize(count);
        for (size_t i = 0; i < count; ++i) {
            slots_[i].buffer = buffers[i];
            slots_[i].capacity = capacities[i];
            if (!buffers[i] || capacities[i] == 0) usable_ = false;
        }
    }

    Ring(const Ring &) = delete;
    Ring &operator=(const Ring &) = delete;

    Lease issue(const Identity &identity) {
        std::lock_guard<std::mutex> guard(mu_);
        if (!usable_ || invalidated_ || identity.model_generation != generation_)
            return {};
        for (size_t i = 0; i < slots_.size(); ++i) {
            Slot &slot = slots_[i];
            if (slot.state != State::empty) continue;
            slot.state = State::reading;
            slot.identity = identity;
            slot.event = 0;
            slot.lease = next_lease_++;
            ++telemetry_.attempts;
            return {i, slot.lease, true};
        }
        ++telemetry_.late;
        return {};
    }

    bool complete_read(Lease lease, Backend &backend) {
        uint8_t *buffer = nullptr;
        size_t capacity = 0;
        Identity identity;
        {
            std::lock_guard<std::mutex> guard(mu_);
            if (!matches_locked(lease, State::reading)) return false;
            Slot &slot = slots_[lease.slot];
            buffer = slot.buffer;
            capacity = slot.capacity;
            identity = slot.identity;
        }
        const ReadResult result = backend.read(identity, buffer, capacity);
        if (!result.ok || result.actual_length != identity.expected_length ||
            result.actual_length > capacity) {
            integrity_failure(lease, backend);
            return false;
        }
        uint8_t actual[32];
        if (!backend.sha256(buffer, result.actual_length, actual) ||
            std::memcmp(actual, identity.expected_sha256.data(), sizeof(actual)) != 0) {
            integrity_failure(lease, backend);
            return false;
        }
        std::lock_guard<std::mutex> guard(mu_);
        if (!matches_locked(lease, State::reading) ||
            slots_[lease.slot].identity.model_generation != generation_) {
            ++telemetry_.stale;
            if (matches_locked(lease, State::reading))
                recycle_locked(slots_[lease.slot]);
            return false;
        }
        slots_[lease.slot].actual_length = result.actual_length;
        slots_[lease.slot].state = State::ready;
        ++telemetry_.sha_successes;
        ++telemetry_.ready;
        telemetry_.validated_bytes += result.actual_length;
        return true;
    }

    Lease claim(const Identity &identity) {
        std::lock_guard<std::mutex> guard(mu_);
        if (!usable_ || invalidated_) return {};
        for (size_t i = 0; i < slots_.size(); ++i) {
            Slot &slot = slots_[i];
            if (slot.state == State::ready && same_identity(slot.identity, identity)) {
                slot.state = State::main_owned;
                return {i, slot.lease, true};
            }
        }
        ++telemetry_.late;
        ++telemetry_.fallback;
        return {};
    }

    bool copy_sync(Lease lease, Backend &backend) {
        BufferSnapshot snapshot;
        if (!snapshot_for_copy(lease, snapshot)) return false;
        if (!revalidate_before_copy(lease, snapshot, backend)) return false;
        const bool ok = backend.copy_sync(snapshot.buffer, snapshot.length);
        std::lock_guard<std::mutex> guard(mu_);
        if (!matches_locked(lease, State::main_owned)) return false;
        if (ok) account_copy_locked(snapshot.length);
        recycle_locked(slots_[lease.slot]);
        return ok;
    }

    bool copy_async(Lease lease, Backend &backend) {
        BufferSnapshot snapshot;
        if (!snapshot_for_copy(lease, snapshot)) return false;
        if (!revalidate_before_copy(lease, snapshot, backend)) return false;
        uint64_t event = 0;
        if (!backend.copy_async(snapshot.buffer, snapshot.length, &event) || event == 0)
            return false;
        std::lock_guard<std::mutex> guard(mu_);
        if (!matches_locked(lease, State::main_owned)) return false;
        slots_[lease.slot].event = event;
        slots_[lease.slot].state = State::copying;
        return true;
    }

    bool poll(Lease lease, Backend &backend) {
        uint64_t event = 0;
        size_t length = 0;
        {
            std::lock_guard<std::mutex> guard(mu_);
            if (!matches_locked(lease, State::copying)) return false;
            event = slots_[lease.slot].event;
            length = slots_[lease.slot].actual_length;
        }
        if (!backend.event_complete(event)) return false;
        std::lock_guard<std::mutex> guard(mu_);
        if (!matches_locked(lease, State::copying) ||
            slots_[lease.slot].event != event) return false;
        account_copy_locked(length);
        recycle_locked(slots_[lease.slot]);
        return true;
    }

    void reload(uint64_t generation) {
        std::lock_guard<std::mutex> guard(mu_);
        generation_ = generation;
        invalidated_ = false;
        for (Slot &slot : slots_) {
            if (slot.state == State::ready) {
                ++telemetry_.stale;
                recycle_locked(slot);
            }
            // READING and COPYING retain their physical buffers until the
            // read returns or the CUDA event completes. MAIN_OWNED is
            // rejected and recycled by snapshot_for_copy below.
        }
    }

    void discard_ready() {
        std::lock_guard<std::mutex> guard(mu_);
        for (Slot &slot : slots_) {
            if (slot.state == State::ready) {
                ++telemetry_.stale;
                recycle_locked(slot);
            }
        }
    }

    bool invalidated() const {
        std::lock_guard<std::mutex> guard(mu_);
        return invalidated_;
    }
    bool demand_fallback_allowed() const { return !invalidated(); }
    State state(size_t index) const {
        std::lock_guard<std::mutex> guard(mu_);
        return index < slots_.size() ? slots_[index].state : State::empty;
    }
    uint64_t lease_token(size_t index) const {
        std::lock_guard<std::mutex> guard(mu_);
        return index < slots_.size() ? slots_[index].lease : 0;
    }
    Telemetry telemetry() const {
        std::lock_guard<std::mutex> guard(mu_);
        return telemetry_;
    }

private:
    struct Slot {
        State state = State::empty;
        uint64_t lease = 0;
        uint64_t event = 0;
        uint8_t *buffer = nullptr;
        size_t capacity = 0;
        size_t actual_length = 0;
        Identity identity{};
    };
    struct BufferSnapshot {
        uint8_t *buffer = nullptr;
        size_t length = 0;
        std::array<uint8_t, 32> digest{};
    };

    static bool same_identity(const Identity &left, const Identity &right) {
        return left.key == right.key &&
            left.gate_offset == right.gate_offset &&
            left.up_offset == right.up_offset &&
            left.down_offset == right.down_offset &&
            left.expected_length == right.expected_length &&
            left.model_generation == right.model_generation &&
            left.expected_sha256 == right.expected_sha256;
    }
    bool matches_locked(Lease lease, State expected) const {
        return lease.valid && lease.slot < slots_.size() &&
            slots_[lease.slot].lease == lease.token &&
            slots_[lease.slot].state == expected;
    }
    bool snapshot_for_copy(Lease lease, BufferSnapshot &snapshot) {
        std::lock_guard<std::mutex> guard(mu_);
        if (invalidated_ || !matches_locked(lease, State::main_owned)) return false;
        Slot &slot = slots_[lease.slot];
        if (slot.identity.model_generation != generation_) {
            ++telemetry_.stale;
            recycle_locked(slot);
            return false;
        }
        snapshot.buffer = slot.buffer;
        snapshot.length = slot.actual_length;
        snapshot.digest = slot.identity.expected_sha256;
        return true;
    }
    bool revalidate_before_copy(Lease lease, const BufferSnapshot &snapshot,
                                Backend &backend) {
        uint8_t actual[32];
        if (!backend.sha256(snapshot.buffer, snapshot.length, actual) ||
            std::memcmp(actual, snapshot.digest.data(), sizeof(actual)) != 0) {
            integrity_failure(lease, backend);
            return false;
        }
        return true;
    }
    void integrity_failure(Lease lease, Backend &backend) {
        bool notify = false;
        {
            std::lock_guard<std::mutex> guard(mu_);
            if (!invalidated_) notify = true;
            invalidated_ = true;
            ++telemetry_.sha_failures;
            for (Slot &slot : slots_) recycle_locked(slot);
        }
        if (notify) backend.invalidate_slab();
        (void)lease;
    }
    static void recycle_locked(Slot &slot) {
        slot.state = State::empty;
        slot.event = 0;
        slot.actual_length = 0;
        slot.identity = {};
        // Deliberately retain the lease token: issue() must advance it before
        // this physical buffer can be owned again.
    }
    void account_copy_locked(size_t bytes) {
        ++telemetry_.copies;
        ++telemetry_.publications;
        telemetry_.copied_bytes += bytes;
    }

    mutable std::mutex mu_;
    std::vector<Slot> slots_;
    uint64_t generation_ = 0;
    uint64_t next_lease_ = 1;
    bool usable_ = false;
    bool invalidated_ = false;
    Telemetry telemetry_{};
};

}  // namespace ds4_slab_prefetch

#endif
