#include "llama-mmap.h"

#include <cstdio>
#include <cstring>
#include <exception>

int main(int argc, char ** argv) {
    if (argc != 3 || (std::strcmp(argv[1], "optional") != 0 && std::strcmp(argv[1], "required") != 0)) {
        std::fprintf(stderr, "usage: direct_io_fault_probe optional|required MODEL_FILE\n");
        return 2;
    }
    try {
        const bool required = std::strcmp(argv[1], "required") == 0;
        llama_file file(argv[2], "rb", true, required);
        unsigned char byte = 0;
        file.read_raw(&byte, sizeof(byte));
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
