#!/usr/bin/env bash
# Host-class parameterization for the engine build scripts.
#
# Sourced by scripts/1*_build_*.sh. Each script accepts --host-class NAME
# (default cuda-spark, which keeps the historical Spark behavior and every
# fail-closed assertion byte-identical), plus --cuda-arch N for cuda-generic
# and --rocm-arch gfxNNNN for rocm.
#
# Classes: cuda-spark | cuda-generic | metal | rocm | cpu
# (configs/hardware-matrix.json records which engine pins apply per class.)
#
# The caller is expected to provide die_env() and die_build().

BUILD_HOST_CLASS=cuda-spark
BUILD_CUDA_ARCH=native
BUILD_ROCM_ARCH=gfx1151
BUILD_HOST_CLASS_ARGS=()

# Consume host-class flags, leave everything else in BUILD_HOST_CLASS_ARGS.
build_host_class_parse() {
    BUILD_HOST_CLASS_ARGS=()
    while (( $# > 0 )); do
        case $1 in
            --host-class)
                [[ -n ${2:-} ]] || die_env '--host-class requires a value'
                BUILD_HOST_CLASS=$2
                shift 2
                ;;
            --cuda-arch)
                [[ ${2:-} =~ ^(native|[0-9]+)$ ]] \
                    || die_env '--cuda-arch must be native or a numeric sm value'
                BUILD_CUDA_ARCH=$2
                shift 2
                ;;
            --rocm-arch)
                [[ ${2:-} =~ ^gfx[0-9a-f]+$ ]] \
                    || die_env '--rocm-arch must look like gfxNNNN'
                BUILD_ROCM_ARCH=$2
                shift 2
                ;;
            *)
                BUILD_HOST_CLASS_ARGS+=("$1")
                shift
                ;;
        esac
    done
    case $BUILD_HOST_CLASS in
        cuda-spark|cuda-generic|metal|rocm|cpu) ;;
        *) die_env "unknown --host-class $BUILD_HOST_CLASS (cuda-spark|cuda-generic|metal|rocm|cpu)" ;;
    esac
}

# Platform/toolchain gates. cuda-spark keeps the historical assertions
# verbatim; every other class fails closed on its own requirements.
build_host_class_require_platform() {
    case $BUILD_HOST_CLASS in
        cuda-spark)
            [[ "$(uname -m)" == aarch64 ]] \
                || die_env 'this build requires uname -m to report aarch64 (pass --host-class to build elsewhere)'
            export PATH="/usr/local/cuda/bin:$PATH"
            command -v nvcc >/dev/null 2>&1 \
                || die_env 'nvcc not found after adding /usr/local/cuda/bin to PATH'
            command -v cuobjdump >/dev/null 2>&1 \
                || die_env 'cuobjdump not found after adding /usr/local/cuda/bin to PATH'
            local nvcc_version
            nvcc_version="$(nvcc --version)" || die_env 'nvcc --version failed'
            [[ "$nvcc_version" =~ release[[:space:]]13\. ]] \
                || die_env 'nvcc release must start with 13.'
            ;;
        cuda-generic)
            export PATH="/usr/local/cuda/bin:$PATH"
            command -v nvcc >/dev/null 2>&1 || die_env 'nvcc is required for cuda-generic'
            command -v cuobjdump >/dev/null 2>&1 \
                || die_env 'cuobjdump is required for cuda-generic'
            # nvcc major is recorded in the manifest, not gated: 12.x consumer
            # toolchains are expected off-Spark.
            ;;
        metal)
            [[ "$(uname -s)" == Darwin ]] || die_env 'metal builds require macOS'
            [[ "$(uname -m)" == arm64 ]] || die_env 'metal builds require Apple Silicon'
            xcrun --find metal >/dev/null 2>&1 \
                || die_env 'Metal toolchain not found (xcrun --find metal failed)'
            ;;
        rocm)
            command -v hipcc >/dev/null 2>&1 || die_env 'hipcc is required for rocm'
            ;;
        cpu)
            ;;
    esac
}

# Accelerator cmake flags for llama.cpp-family builds.
build_host_class_cmake_flags() {
    case $BUILD_HOST_CLASS in
        cuda-spark)
            printf '%s\n' -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121
            ;;
        cuda-generic)
            printf '%s\n' -DGGML_CUDA=ON "-DCMAKE_CUDA_ARCHITECTURES=$BUILD_CUDA_ARCH"
            ;;
        metal)
            printf '%s\n' -DGGML_METAL=ON -DGGML_CUDA=OFF
            ;;
        rocm)
            printf '%s\n' -DGGML_HIP=ON "-DAMDGPU_TARGETS=$BUILD_ROCM_ARCH" -DGGML_CUDA=OFF
            ;;
        cpu)
            printf '%s\n' -DGGML_CUDA=OFF -DGGML_METAL=OFF
            ;;
    esac
}

# Post-build artifact assertion. Prints the observed arch evidence on stdout
# (recorded in the manifest); fails closed per class.
build_host_class_assert_artifacts() {
    local bin_dir=$1 observed=
    case $BUILD_HOST_CLASS in
        cuda-spark)
            local elf_head elf_status
            set +o pipefail
            elf_head="$(cuobjdump --list-elf "$bin_dir/libggml-cuda.so" 2>/dev/null | head)"
            elf_status=$?
            set -o pipefail
            (( elf_status == 0 )) \
                || die_build 'cuobjdump inspection of libggml-cuda.so failed'
            [[ "$elf_head" == *sm_121* ]] \
                || die_build 'libggml-cuda.so CUDA objects do not report sm_121'
            observed=sm_121
            ;;
        cuda-generic)
            local elf_head elf_status
            set +o pipefail
            elf_head="$(cuobjdump --list-elf "$bin_dir/libggml-cuda.so" 2>/dev/null | head)"
            elf_status=$?
            set -o pipefail
            (( elf_status == 0 )) \
                || die_build 'cuobjdump inspection of libggml-cuda.so failed'
            observed=$(printf '%s\n' "$elf_head" | grep -o 'sm_[0-9]*' | sort -u | tr '\n' ',' )
            [[ -n $observed ]] \
                || die_build 'libggml-cuda.so reports no sm_ architectures'
            ;;
        metal)
            local server=$bin_dir/llama-server
            otool -L "$server" 2>/dev/null | grep -q Metal \
                || [[ -e $bin_dir/libggml-metal.dylib || -e $bin_dir/default.metallib ]] \
                || die_build 'build shows no Metal linkage or metallib artifact'
            observed=metal
            ;;
        rocm)
            [[ -e $bin_dir/libggml-hip.so || -e $bin_dir/libggml-rocm.so ]] \
                || die_build 'build produced no ROCm ggml library'
            observed=$BUILD_ROCM_ARCH
            ;;
        cpu)
            observed=cpu
            ;;
    esac
    printf '%s\n' "$observed"
}

# Backend name for manifests/profiles.
build_host_class_backend() {
    case $BUILD_HOST_CLASS in
        cuda-spark|cuda-generic) printf 'cuda\n' ;;
        metal) printf 'apple-silicon\n' ;;
        rocm) printf 'rocm\n' ;;
        cpu) printf 'cpu\n' ;;
    esac
}
