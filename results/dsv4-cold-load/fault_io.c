#define _GNU_SOURCE

#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* Small LD_PRELOAD mutation helper. It is never linked into serving code. */

#define MAX_TRACKED_FD 1048576

static unsigned char partial_done[MAX_TRACKED_FD];
static int selected_read_fd = -1;

static bool ends_with(const char * value, const char * suffix) {
    if (suffix == NULL || *suffix == '\0') {
        return true;
    }
    const size_t value_len = strlen(value);
    const size_t suffix_len = strlen(suffix);
    return value_len >= suffix_len &&
           memcmp(value + value_len - suffix_len, suffix, suffix_len) == 0;
}

static int selected_errno(void) {
    const char * setting = getenv("DSV4_FAULT_READ_ERRNO");
    if (setting == NULL || *setting == '\0' || strcmp(setting, "EINVAL") == 0) {
        return EINVAL;
    }
    if (strcmp(setting, "EFAULT") == 0) {
        return EFAULT;
    }
    return 0;
}

typedef int (*open_fn)(const char *, int, ...);
typedef ssize_t (*read_fn)(int, void *, size_t);
typedef int (*close_fn)(int);

int open(const char * path, int flags, ...) {
    static open_fn real_open;
    if (real_open == NULL) {
        real_open = (open_fn) dlsym(RTLD_NEXT, "open");
    }
    mode_t mode = 0;
    if ((flags & O_CREAT) != 0) {
        va_list args;
        va_start(args, flags);
        mode = (mode_t) va_arg(args, int);
        va_end(args);
    }
    const char * open_suffix = getenv("DSV4_FAULT_OPEN_SUFFIX");
    if ((flags & O_DIRECT) != 0 && open_suffix != NULL && *open_suffix != '\0' &&
            ends_with(path, open_suffix)) {
        errno = EINVAL;
        return -1;
    }
    int fd = (flags & O_CREAT) != 0 ? real_open(path, flags, mode) : real_open(path, flags);
    if (fd >= 0 && fd < MAX_TRACKED_FD && (flags & O_DIRECT) != 0) {
        partial_done[fd] = 0;
        const char * read_suffix = getenv("DSV4_FAULT_READ_SUFFIX");
        if (read_suffix != NULL && *read_suffix != '\0' && ends_with(path, read_suffix)) {
            selected_read_fd = fd;
        }
    }
    return fd;
}

int open64(const char * path, int flags, ...) {
    mode_t mode = 0;
    if ((flags & O_CREAT) != 0) {
        va_list args;
        va_start(args, flags);
        mode = (mode_t) va_arg(args, int);
        va_end(args);
        return open(path, flags, mode);
    }
    return open(path, flags);
}

ssize_t read(int fd, void * buffer, size_t count) {
    static read_fn real_read;
    if (real_read == NULL) {
        real_read = (read_fn) dlsym(RTLD_NEXT, "read");
    }
    if (fd < 0 || fd >= MAX_TRACKED_FD) {
        return real_read(fd, buffer, count);
    }
    if (fd != selected_read_fd) {
        return real_read(fd, buffer, count);
    }
    const char * partial_text = getenv("DSV4_FAULT_PARTIAL_BYTES");
    if (partial_done[fd] == 0 && partial_text != NULL && *partial_text != '\0') {
        char * end = NULL;
        unsigned long long requested = strtoull(partial_text, &end, 10);
        if (end != partial_text && *end == '\0' && requested > 0 && requested < count) {
            partial_done[fd] = 1;
            return real_read(fd, buffer, (size_t) requested);
        }
    }
    const int injected = selected_errno();
    if (injected != 0) {
        errno = injected;
        return -1;
    }
    return real_read(fd, buffer, count);
}

int close(int fd) {
    static close_fn real_close;
    if (real_close == NULL) {
        real_close = (close_fn) dlsym(RTLD_NEXT, "close");
    }
    if (fd >= 0 && fd < MAX_TRACKED_FD) {
        partial_done[fd] = 0;
    }
    return real_close(fd);
}
