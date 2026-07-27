#include <unistd.h>

int main(void) {
    sleep(2);
    execl("/bin/sleep", "sleep", "2", (char *)0);
    return 111;
}
