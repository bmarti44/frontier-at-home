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

enum class State : uint8_t { empty, reading, ready, main_owned, retired };

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
    uint64_t current_ready = 0;
    uint64_t read_ns = 0;
    uint64_t sha_ns = 0;
    uint64_t wait_ns = 0;
    uint64_t copy_ns = 0;
};

struct ReadResult { bool ok = false; size_t actual_length = 0; };

class Backend {
public:
    virtual ~Backend() = default;
    virtual ReadResult read(const Identity &, uint8_t *, size_t) = 0;
    virtual bool sha256(const uint8_t *, size_t, uint8_t out[32]) = 0;
    virtual bool copy_sync(const uint8_t *, size_t) = 0;
    virtual void invalidate_slab() = 0;
    virtual uint64_t now_ns() = 0;
    virtual bool publication_expected() const = 0;
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
        const uint64_t read_start = backend.now_ns();
        const ReadResult result = backend.read(identity, buffer, capacity);
        const uint64_t read_end = backend.now_ns();
        if (!add_elapsed(read_start, read_end, &Telemetry::read_ns)) {
            integrity_failure(lease, backend);
            return false;
        }
        if (!result.ok || result.actual_length != identity.expected_length ||
            result.actual_length > capacity) {
            integrity_failure(lease, backend);
            return false;
        }
        uint8_t actual[32];
        const uint64_t sha_start = backend.now_ns();
        const bool sha_ok = backend.sha256(buffer, result.actual_length, actual);
        const uint64_t sha_end = backend.now_ns();
        if (!add_elapsed(sha_start, sha_end, &Telemetry::sha_ns) || !sha_ok ||
            std::memcmp(actual, identity.expected_sha256.data(), sizeof(actual)) != 0) {
            integrity_failure(lease, backend);
            return false;
        }
        std::lock_guard<std::mutex> guard(mu_);
        if (matches_locked(lease, State::retired)) {
            recycle_locked(slots_[lease.slot]);
            return false;
        }
        if (invalidated_ || !matches_locked(lease, State::reading)) return false;
        ++telemetry_.sha_successes;
        telemetry_.validated_bytes += result.actual_length;
        if (slots_[lease.slot].identity.model_generation != generation_) {
            ++telemetry_.stale;
            recycle_locked(slots_[lease.slot]);
            return false;
        }
        slots_[lease.slot].actual_length = result.actual_length;
        slots_[lease.slot].state = State::ready;
        ++telemetry_.ready;
        ++telemetry_.current_ready;
        return true;
    }

    Lease claim(const Identity &identity) {
        std::lock_guard<std::mutex> guard(mu_);
        if (!usable_ || invalidated_) return {};
        for (size_t i = 0; i < slots_.size(); ++i) {
            Slot &slot = slots_[i];
            if (slot.state == State::ready && same_identity(slot.identity, identity)) {
                slot.state = State::main_owned;
                --telemetry_.current_ready;
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
        /* Publication is a side effect. Serialize the authorization check and
         * the production copy so an integrity failure cannot become visible
         * between them and permit a post-failure cache insertion. */
        std::lock_guard<std::mutex> guard(mu_);
        if (matches_locked(lease, State::retired)) {
            recycle_locked(slots_[lease.slot]);
            return false;
        }
        if (invalidated_ || !matches_locked(lease, State::main_owned)) return false;
        const uint64_t copy_start = backend.now_ns();
        const bool ok = backend.copy_sync(snapshot.buffer, snapshot.length);
        const uint64_t copy_end = backend.now_ns();
        if (copy_end < copy_start) {
            recycle_locked(slots_[lease.slot]);
            return false;
        }
        telemetry_.copy_ns += copy_end - copy_start;
        if (ok) account_copy_locked(
            snapshot.length, backend.publication_expected());
        recycle_locked(slots_[lease.slot]);
        return ok;
    }

    bool reload(uint64_t generation) {
        std::lock_guard<std::mutex> guard(mu_);
        for (const Slot &slot : slots_)
            if (slot.state != State::empty) return false;
        generation_ = generation;
        invalidated_ = false;
        return true;
    }

    void discard_ready() {
        std::lock_guard<std::mutex> guard(mu_);
        for (Slot &slot : slots_) {
            if (slot.state == State::ready) {
                ++telemetry_.stale;
                --telemetry_.current_ready;
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
        const uint64_t sha_start = backend.now_ns();
        const bool sha_ok = backend.sha256(snapshot.buffer, snapshot.length, actual);
        const uint64_t sha_end = backend.now_ns();
        if (!add_elapsed(sha_start, sha_end, &Telemetry::sha_ns) || !sha_ok ||
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
            for (size_t index = 0; index < slots_.size(); ++index) {
                Slot &slot = slots_[index];
                const bool current = lease.valid && lease.slot == index &&
                    slot.lease == lease.token;
                if (current || slot.state == State::ready) {
                    recycle_locked(slot);
                } else if (slot.state == State::reading ||
                           slot.state == State::main_owned) {
                    /* The buffer is still physically owned by a read or by a
                     * pre-copy digest. It becomes reusable only when that
                     * exact lease returns through complete_read/copy_sync. */
                    slot.state = State::retired;
                }
            }
            telemetry_.current_ready = 0;
        }
        if (notify) backend.invalidate_slab();
        (void)lease;
    }
    static void recycle_locked(Slot &slot) {
        slot.state = State::empty;
        slot.actual_length = 0;
        slot.identity = {};
        // Deliberately retain the lease token: issue() must advance it before
        // this physical buffer can be owned again.
    }
    void account_copy_locked(size_t bytes, bool published) {
        ++telemetry_.copies;
        telemetry_.publications += published ? 1 : 0;
        telemetry_.copied_bytes += bytes;
    }
    bool add_elapsed(uint64_t start, uint64_t end,
                     uint64_t Telemetry::*field) {
        if (end < start) return false;
        std::lock_guard<std::mutex> guard(mu_);
        telemetry_.*field += end - start;
        return true;
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
