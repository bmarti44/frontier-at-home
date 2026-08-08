#include "llama-mmap.h"

#include <cstdio>
#include <exception>

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: direct_io_fault_probe MODEL_FILE\n");
        return 2;
    }
    try {
        llama_file file(argv[1], "rb", true);
        unsigned char byte = 0;
        file.read_raw(&byte, sizeof(byte));
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
