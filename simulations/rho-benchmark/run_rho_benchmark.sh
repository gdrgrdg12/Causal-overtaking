#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
SECP_DIR="$ROOT_DIR/third_party/secp256k1"
BUILD_DIR=${BUILD_DIR:-"$SCRIPT_DIR/build"}
RESULT_DIR=${RESULT_DIR:-"$SCRIPT_DIR/results/local-run00"}
ITERATIONS=${ITERATIONS:-50000}
WARMUP=${WARMUP:-5000}
BUILD_JOBS=${BUILD_JOBS:-$(nproc)}
SECP_BUILD_DIR="$BUILD_DIR/secp256k1"
SECP_ARCHIVE="$SECP_BUILD_DIR/.libs/libsecp256k1.a"

mkdir -p "$BUILD_DIR" "$RESULT_DIR"

if [[ ! -f "$SECP_ARCHIVE" ]]; then
    rm -rf "$SECP_BUILD_DIR"
    mkdir -p "$SECP_BUILD_DIR"
    cp -a "$SECP_DIR/." "$SECP_BUILD_DIR/"
    make -C "$SECP_BUILD_DIR" distclean >/dev/null 2>&1 || true
    (
        cd "$SECP_BUILD_DIR"
        bash ./configure \
            --disable-shared \
            --enable-static \
            --disable-benchmark \
            --disable-tests \
            --disable-exhaustive-tests \
            --with-asm=auto
        make -j"$BUILD_JOBS"
    )
fi

g++ -O3 -DNDEBUG -std=c++17 -Wall -Wextra -Wpedantic \
    -I"$SECP_BUILD_DIR/include" \
    "$SCRIPT_DIR/rho_benchmark.cpp" \
    "$SECP_ARCHIVE" \
    -lcrypto -lpthread \
    -o "$BUILD_DIR/rho_benchmark"

for mode in prebuilt build signed; do
    "$BUILD_DIR/rho_benchmark" \
        --mode "$mode" \
        --iterations "$ITERATIONS" \
        --warmup "$WARMUP" \
        --output "$RESULT_DIR/raw-$mode.csv"
done

python3 "$SCRIPT_DIR/summarize_rho.py" \
    --output_dir "$RESULT_DIR" \
    "$RESULT_DIR/raw-prebuilt.csv" \
    "$RESULT_DIR/raw-build.csv" \
    "$RESULT_DIR/raw-signed.csv"

echo "rho benchmark results: $RESULT_DIR"
