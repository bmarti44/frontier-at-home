#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static double monotonic_seconds(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) return -1.0;
    return (double)value.tv_sec + (double)value.tv_nsec / 1000000000.0;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s MODEL SECONDS\n", argv[0]);
        return 2;
    }
    char *end = NULL;
    errno = 0;
    const unsigned long duration = strtoul(argv[2], &end, 10);
    if (errno != 0 || end == argv[2] || *end != '\0' ||
        duration < 1 || duration > 300) {
        fprintf(stderr, "invalid duration\n");
        return 2;
    }

    const int descriptor = open(argv[1], O_RDONLY | O_DIRECT | O_CLOEXEC);
    if (descriptor < 0) {
        perror("open O_DIRECT");
        return 3;
    }
    struct stat details;
    const int flags = fcntl(descriptor, F_GETFL);
    if (flags < 0 || (flags & O_DIRECT) == 0 ||
        fstat(descriptor, &details) != 0 || details.st_size < 8 * 1024 * 1024) {
        fprintf(stderr, "direct-I/O identity check failed\n");
        close(descriptor);
        return 4;
    }

    const size_t block = 4 * 1024 * 1024;
    void *buffer = NULL;
    if (posix_memalign(&buffer, 4096, block) != 0 || !buffer) {
        fprintf(stderr, "aligned allocation failed\n");
        close(descriptor);
        return 5;
    }

    const double started = monotonic_seconds();
    const double deadline = started + (double)duration;
    uint64_t bytes = 0;
    off_t offset = 0;
    while (monotonic_seconds() < deadline) {
        if (offset + (off_t)block > details.st_size) offset = 0;
        const ssize_t count = pread(descriptor, buffer, block, offset);
        if (count <= 0) {
            perror("pread");
            free(buffer);
            close(descriptor);
            return 6;
        }
        bytes += (uint64_t)count;
        offset += count;
    }
    const double elapsed = monotonic_seconds() - started;
    printf(
        "{\"bytes_read\":%" PRIu64
        ",\"direct_io\":true,\"elapsed_s\":%.9f,\"fcntl_flags\":%d,"
        "\"pid\":%ld}\n",
        bytes, elapsed, flags, (long)getpid());
    fflush(stdout);
    free(buffer);
    close(descriptor);
    return 0;
}
