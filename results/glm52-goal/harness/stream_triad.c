#define _GNU_SOURCE
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double seconds(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &ts) != 0) return -1.0;
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(void) {
    const size_t count = 64u * 1024u * 1024u;
    const int iterations = 50;
    const double scalar = 3.0;
    double *a = aligned_alloc(4096, count * sizeof(*a));
    double *b = aligned_alloc(4096, count * sizeof(*b));
    double *c = aligned_alloc(4096, count * sizeof(*c));
    if (!a || !b || !c) return 2;

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < count; i++) {
        a[i] = 0.0;
        b[i] = 1.0;
        c[i] = 2.0;
    }

    const double start = seconds();
    for (int iteration = 0; iteration < iterations; iteration++) {
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < count; i++) a[i] = b[i] + scalar * c[i];
    }
    const double elapsed = seconds() - start;

    long double checksum = 0.0;
#pragma omp parallel for reduction(+ : checksum) schedule(static)
    for (size_t i = 0; i < count; i++) checksum += a[i];
    const long double expected = (long double)count * 7.0L;
    if (!(elapsed > 0.0) || fabsl(checksum - expected) > 0.5L) return 3;

    const uint64_t bytes =
        (uint64_t)count * sizeof(double) * 3u * (uint64_t)iterations;
    printf(
        "{\"threads\":%d,\"array_elements\":%zu,\"iterations\":%d,"
        "\"bytes\":%llu,\"seconds\":%.9f,\"gb_per_second\":%.6f,"
        "\"checksum\":%.0Lf}\n",
        omp_get_max_threads(), count, iterations,
        (unsigned long long)bytes, elapsed, (double)bytes / elapsed / 1e9,
        checksum);
    free(c);
    free(b);
    free(a);
    return 0;
}
