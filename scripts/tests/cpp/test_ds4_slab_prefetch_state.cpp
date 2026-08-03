#include "ds4_slab_prefetch_state.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <thread>
#include <condition_variable>
#include <mutex>

using namespace ds4_slab_prefetch;

#define REQUIRE(condition) do { if (!(condition)) { \
    std::fprintf(stderr, "REQUIRE failed at line %d: %s\n", __LINE__, #condition); \
    std::exit(1); } } while (0)

static std::array<uint8_t, 32> good_digest() {
    std::array<uint8_t, 32> value{};
    value.fill(0x42);
    return value;
}

struct FakeBackend : Backend {
    std::vector<uint8_t> canonical;
    std::vector<uint8_t> source;
    size_t reported_length = 0;
    bool read_ok = true;
    bool copy_ok = true;
    bool invalidated = false;
    uint64_t copy_calls = 0;
    uint64_t copied_bytes = 0;
    uint64_t clock_ns = 1000;

    explicit FakeBackend(size_t bytes) : canonical(bytes), source(bytes), reported_length(bytes) {
        for (size_t i = 0; i < bytes; ++i) canonical[i] = source[i] = uint8_t(i * 37u + 11u);
    }
    ReadResult read(const Identity &, uint8_t *dst, size_t capacity) override {
        if (!read_ok || source.size() > capacity) return {false, 0};
        std::memcpy(dst, source.data(), source.size());
        return {true, reported_length};
    }
    bool sha256(const uint8_t *data, size_t bytes, uint8_t out[32]) override {
        const bool exact = bytes == canonical.size() &&
            std::equal(data, data + bytes, canonical.begin());
        std::memset(out, exact ? 0x42 : 0x99, 32);
        return true;
    }
    bool copy_sync(const uint8_t *, size_t bytes) override {
        ++copy_calls; copied_bytes += bytes; return copy_ok;
    }
    void invalidate_slab() override { invalidated = true; }
    uint64_t now_ns() override { clock_ns += 10; return clock_ns; }
    bool publication_expected() const override { return true; }
};

static Identity identity(size_t bytes, uint64_t generation = 7) {
    Identity value;
    value.key = (uint64_t(12) << 32) | 34;
    value.gate_offset = 4096;
    value.up_offset = 8192;
    value.down_offset = 12288;
    value.expected_length = bytes;
    value.model_generation = generation;
    value.expected_sha256 = good_digest();
    return value;
}

static Ring make_ring(size_t count, uint64_t generation,
                      std::vector<std::vector<uint8_t>> &storage) {
    std::vector<uint8_t *> buffers;
    std::vector<size_t> capacities;
    for (auto &slot : storage) { buffers.push_back(slot.data()); capacities.push_back(slot.size()); }
    return Ring(count, generation, buffers, capacities);
}

static void valid_sync_path() {
    for (size_t count = 4; count <= 8; ++count) {
        std::vector<std::vector<uint8_t>> storage(count, std::vector<uint8_t>(256));
        Ring ring = make_ring(count, 7, storage);
        FakeBackend backend(256);
        const Identity id = identity(256);
        Lease worker = ring.issue(id);
        REQUIRE(worker.valid);
        REQUIRE(ring.complete_read(worker, backend));
        REQUIRE(ring.state(worker.slot) == State::ready);
        Lease owner = ring.claim(id);
        REQUIRE(owner.valid && owner.token == worker.token);
        REQUIRE(ring.copy_sync(owner, backend));
        REQUIRE(ring.state(owner.slot) == State::empty);
        Telemetry t = ring.telemetry();
        REQUIRE(t.attempts == 1 && t.sha_successes == 1 && t.ready == 1);
        REQUIRE(t.copies == 1 && t.validated_bytes == 256 && t.copied_bytes == 256);

    }
}

struct BlockingBackend final : FakeBackend {
    std::mutex mutex;
    std::condition_variable condition;
    bool entered = false;
    bool release = false;

    explicit BlockingBackend(size_t bytes) : FakeBackend(bytes) {}
    ReadResult read(const Identity &id, uint8_t *dst, size_t capacity) override {
        {
            std::unique_lock<std::mutex> lock(mutex);
            entered = true;
            condition.notify_all();
            condition.wait(lock, [&] { return release; });
        }
        return FakeBackend::read(id, dst, capacity);
    }
};

static void invalidation_retires_inflight_buffers_until_physical_completion() {
    std::vector<std::vector<uint8_t>> storage(4, std::vector<uint8_t>(256));
    Ring ring = make_ring(4, 7, storage);
    BlockingBackend blocked(256);
    FakeBackend corrupt(256);
    corrupt.source[17] ^= 1;
    Lease inflight = ring.issue(identity(256));
    Lease failing = ring.issue(identity(256));
    REQUIRE(inflight.valid && failing.valid && inflight.slot != failing.slot);
    std::thread reader([&] { REQUIRE(!ring.complete_read(inflight, blocked)); });
    {
        std::unique_lock<std::mutex> lock(blocked.mutex);
        blocked.condition.wait(lock, [&] { return blocked.entered; });
    }
    REQUIRE(!ring.complete_read(failing, corrupt));
    REQUIRE(ring.invalidated());
    REQUIRE(ring.state(inflight.slot) == State::retired);
    REQUIRE(!ring.reload(8));
    REQUIRE(!ring.issue(identity(256, 8)).valid);
    {
        std::lock_guard<std::mutex> lock(blocked.mutex);
        blocked.release = true;
    }
    blocked.condition.notify_all();
    reader.join();
    REQUIRE(ring.state(inflight.slot) == State::empty);
    REQUIRE(ring.reload(8));
    REQUIRE(ring.issue(identity(256, 8)).valid);
}

static void corruption_fails_closed() {
    const size_t positions[] = {0, 128, 255};
    for (size_t position : positions) {
        std::vector<std::vector<uint8_t>> storage(4, std::vector<uint8_t>(256));
        Ring ring = make_ring(4, 7, storage);
        FakeBackend backend(256);
        backend.source[position] ^= 1;
        Lease lease = ring.issue(identity(256));
        REQUIRE(lease.valid && !ring.complete_read(lease, backend));
        REQUIRE(ring.invalidated() && backend.invalidated);
        REQUIRE(!ring.demand_fallback_allowed());
        REQUIRE(backend.copy_calls == 0);
        Telemetry t = ring.telemetry();
        REQUIRE(t.sha_failures == 1 && t.ready == 0 && t.copies == 0 &&
                t.publications == 0 && t.fallback == 0 && t.copied_bytes == 0);
    }

    // A compensating edit defeats additive checks but not the digest oracle.
    std::vector<std::vector<uint8_t>> storage(4, std::vector<uint8_t>(256));
    Ring ring = make_ring(4, 7, storage);
    FakeBackend backend(256);
    backend.source[10] += 1; backend.source[11] -= 1;
    Lease lease = ring.issue(identity(256));
    REQUIRE(!ring.complete_read(lease, backend) && ring.invalidated());

    // Short, overlong, and failed reads are integrity failures, never lateness.
    for (size_t length : {size_t(0), size_t(255), size_t(257)}) {
        std::vector<std::vector<uint8_t>> local(4, std::vector<uint8_t>(256));
        Ring candidate = make_ring(4, 7, local);
        FakeBackend short_read(256); short_read.reported_length = length;
        Lease item = candidate.issue(identity(256));
        REQUIRE(!candidate.complete_read(item, short_read));
        REQUIRE(candidate.invalidated() && !candidate.demand_fallback_allowed());
    }
}

static void identity_lease_and_reload_are_aba_safe() {
    std::vector<std::vector<uint8_t>> storage(4, std::vector<uint8_t>(256));
    Ring ring = make_ring(4, 7, storage);
    FakeBackend backend(256);
    Identity id = identity(256);
    Lease worker = ring.issue(id);
    REQUIRE(ring.complete_read(worker, backend));

    for (int field = 0; field < 5; ++field) {
        Identity wrong = id;
        if (field == 0) ++wrong.key;
        if (field == 1) ++wrong.gate_offset;
        if (field == 2) ++wrong.up_offset;
        if (field == 3) ++wrong.down_offset;
        if (field == 4) ++wrong.expected_length;
        REQUIRE(!ring.claim(wrong).valid);
        REQUIRE(!ring.invalidated());  // a clean prediction miss may demand-read
        REQUIRE(ring.demand_fallback_allowed());
    }

    Lease owner = ring.claim(id);
    REQUIRE(owner.valid);
    REQUIRE(!ring.reload(8));
    REQUIRE(ring.copy_sync(owner, backend));
    REQUIRE(ring.reload(8));
    REQUIRE(backend.copy_calls == 1);
    REQUIRE(ring.state(owner.slot) == State::empty);
    REQUIRE(!ring.issue(identity(256, 7)).valid);
    REQUIRE(ring.issue(identity(256, 8)).valid);
}

static void mutation_after_validation_is_caught_before_copy() {
    std::vector<std::vector<uint8_t>> storage(4, std::vector<uint8_t>(256));
    Ring ring = make_ring(4, 7, storage);
    FakeBackend backend(256);
    Identity id = identity(256);
    Lease worker = ring.issue(id);
    REQUIRE(ring.complete_read(worker, backend));
    Lease owner = ring.claim(id);
    REQUIRE(owner.valid);
    storage[owner.slot][127] ^= 0x80;
    REQUIRE(!ring.copy_sync(owner, backend));
    REQUIRE(ring.invalidated() && backend.invalidated);
    REQUIRE(backend.copy_calls == 0);
    Telemetry t = ring.telemetry();
    REQUIRE(t.copies == 0 && t.publications == 0 && t.copied_bytes == 0 &&
            t.fallback == 0);
}

static void duplicate_consumers_and_ring_exhaustion_are_safe() {
    std::vector<std::vector<uint8_t>> storage(4, std::vector<uint8_t>(256));
    Ring ring = make_ring(4, 7, storage);
    FakeBackend backend(256);
    Identity id = identity(256);
    Lease worker = ring.issue(id);
    REQUIRE(ring.complete_read(worker, backend));
    std::atomic<bool> start{false};
    std::array<Lease, 2> claims{};
    std::thread left([&] { while (!start.load(std::memory_order_acquire)) {}
                           claims[0] = ring.claim(id); });
    std::thread right([&] { while (!start.load(std::memory_order_acquire)) {}
                            claims[1] = ring.claim(id); });
    start.store(true, std::memory_order_release);
    left.join(); right.join();
    REQUIRE(int(claims[0].valid) + int(claims[1].valid) == 1);

    std::vector<std::vector<uint8_t>> full_storage(4, std::vector<uint8_t>(256));
    Ring full = make_ring(4, 7, full_storage);
    std::array<Lease, 4> leases{};
    for (Lease &lease : leases) { lease = full.issue(id); REQUIRE(lease.valid); }
    REQUIRE(!full.issue(id).valid);
    for (size_t i = 0; i < leases.size(); ++i)
        REQUIRE(full.lease_token(leases[i].slot) == leases[i].token);
}

static void repeated_publication_stress() {
    std::vector<std::vector<uint8_t>> storage(4, std::vector<uint8_t>(256));
    Ring ring = make_ring(4, 7, storage);
    FakeBackend backend(256);
    Identity id = identity(256);
    for (int iteration = 0; iteration < 2000; ++iteration) {
        Lease worker = ring.issue(id);
        REQUIRE(worker.valid && ring.complete_read(worker, backend));
        Lease owner = ring.claim(id);
        REQUIRE(owner.valid && ring.copy_sync(owner, backend));
    }
    Telemetry t = ring.telemetry();
    REQUIRE(t.attempts == 2000 && t.sha_successes == 2000 &&
            t.ready == 2000 && t.copies == 2000 && t.publications == 2000 &&
            t.validated_bytes == 2000 * 256 && t.copied_bytes == 2000 * 256);
}

int main() {
    valid_sync_path();
    invalidation_retires_inflight_buffers_until_physical_completion();
    corruption_fails_closed();
    identity_lease_and_reload_are_aba_safe();
    mutation_after_validation_is_caught_before_copy();
    duplicate_consumers_and_ring_exhaustion_are_safe();
    repeated_publication_stress();
    return 0;
}
