#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

int main(void) {
    sleep(2);
    pid_t child = fork();
    if (child < 0) {
        return 112;
    }
    if (child == 0) {
        if (setsid() < 0) {
            return 113;
        }
        signal(SIGHUP, SIG_IGN);
        signal(SIGTERM, SIG_IGN);
        const char *path = getenv("MUTATION_PID_FILE");
        FILE *handle = path ? fopen(path, "w") : NULL;
        if (!handle) {
            return 114;
        }
        fprintf(handle, "%ld\n", (long)getpid());
        fclose(handle);
        struct timespec remaining = {20, 0};
        while (nanosleep(&remaining, &remaining) != 0 && errno == EINTR) {
        }
        return 0;
    }
    return 0;
}
