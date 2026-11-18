#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
source "$SCRIPT_DIR/../runtime.sh"
SECP_DIR="$ROOT_DIR/third_party/secp256k1"
BUILD_DIR=${BUILD_DIR:-"$SCRIPT_DIR/build"}
RESULT_DIR=${RESULT_DIR:-"$SCRIPT_DIR/results/local-run00"}
ITERATIONS=${ITERATIONS:-50000}
WARMUP=${WARMUP:-5000}
BUILD_JOBS=${BUILD_JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf '1')}
SECP_BUILD_DIR="$BUILD_DIR/secp256k1"
SECP_ARCHIVE="$SECP_BUILD_DIR/.libs/libsecp256k1.a"

# Prefer an explicit prefix, then pkg-config, then existing Homebrew paths.
# Read installed libraries only; this script never installs system packages.
crypto_flags=(-lcrypto)
if [[ -n "${OPENSSL_PREFIX:-}" ]]; then
    crypto_flags=("-I$OPENSSL_PREFIX/include" "-L$OPENSSL_PREFIX/lib" -lcrypto)
elif command -v pkg-config >/dev/null 2>&1 && pkg-config --exists libcrypto; then
    read -r -a crypto_flags <<< "$(pkg-config --cflags --libs libcrypto)"
elif [[ "$(uname -s)" == Darwin ]]; then
    for prefix in /opt/homebrew/opt/openssl@3 /usr/local/opt/openssl@3; do
        if [[ -f "$prefix/include/openssl/sha.h" ]]; then
            crypto_flags=("-I$prefix/include" "-L$prefix/lib" -lcrypto)
            break
        fi
    done
    if (( ${#crypto_flags[@]} == 1 )); then
        echo "OpenSSL not found. Set OPENSSL_PREFIX to an existing OpenSSL installation." >&2
        exit 2
    fi
fi

mkdir -p "$BUILD_DIR" "$RESULT_DIR"

if [[ ! -f "$SECP_ARCHIVE" ]]; then
    rm -rf "$SECP_BUILD_DIR"
    mkdir -p "$SECP_BUILD_DIR"
    cp -a "$SECP_DIR/." "$SECP_BUILD_DIR/"
    # ZIP downloads may strip executable bits. Restore them in the build copy.
    chmod +x "$SECP_BUILD_DIR/configure" "$SECP_BUILD_DIR/build-aux/"*
    make -C "$SECP_BUILD_DIR" distclean >/dev/null 2>&1 || true
    (
        cd "$SECP_BUILD_DIR"
        bash ./configure \
            --disable-shared \
            --enable-static \
            --disable-benchmark \
            --disable-tests \
            --disable-exhaustive-tests \
            --with-bignum=no \
            --with-asm=auto
        # Generated configure/Makefile.in are vendored. Suppress timestamp-based
        # regeneration so unpacking the artifact needs no autotools installation.
        make -j"$BUILD_JOBS" AUTOCONF=: AUTOHEADER=: AUTOMAKE=: ACLOCAL=:
    )
fi

"${CXX:-c++}" -O3 -DNDEBUG -std=c++17 -Wall -Wextra -Wpedantic \
    -I"$SECP_BUILD_DIR/include" \
    "$SCRIPT_DIR/rho_benchmark.cpp" \
    "$SECP_ARCHIVE" \
    "${crypto_flags[@]}" -pthread \
    -o "$BUILD_DIR/rho_benchmark"

for mode in prebuilt build signed; do
    "$BUILD_DIR/rho_benchmark" \
        --mode "$mode" \
        --iterations "$ITERATIONS" \
        --warmup "$WARMUP" \
        --output "$RESULT_DIR/raw-$mode.csv"
done

"$PYTHON_BIN" "$SCRIPT_DIR/summarize_rho.py" \
    --output_dir "$RESULT_DIR" \
    "$RESULT_DIR/raw-prebuilt.csv" \
    "$RESULT_DIR/raw-build.csv" \
    "$RESULT_DIR/raw-signed.csv"

echo "rho benchmark results: $RESULT_DIR"
