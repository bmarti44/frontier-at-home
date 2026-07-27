#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>

static pid_t start_candidate(const char *path) {
    pid_t pid = fork();
    if (pid == 0) {
        execl(path, "ds4-server", "30", (char *)NULL);
        _exit(127);
    }
    return pid;
}

int main(int argc, char **argv) {
    if (argc != 3 || (strcmp(argv[1], "clean") && strcmp(argv[1], "replace")))
        return 2;
    pid_t first = start_candidate(argv[2]);
    if (first <= 0) return 3;
    sleep(1);
    if (kill(first, SIGTERM)) return 4;
    /* Deliberately leave the first child as a zombie, matching the production
       arm's short stop_server-to-shell-exit window. */
    usleep(300000);
    if (!strcmp(argv[1], "replace")) {
        pid_t second = start_candidate(argv[2]);
        if (second <= 0) return 5;
        sleep(2);
        kill(second, SIGTERM);
        return 0;
    }
    usleep(300000);
    return 0;
}
