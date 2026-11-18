# Local Processing Benchmark

Requires a C++17 compiler, `make`, and OpenSSL development libraries.

From the repository root:

```bash
ITERATIONS=50000 WARMUP=5000 bash simulations/run.sh benchmark
```

The benchmark measures three modes: releasing a prebuilt transaction,
constructing one without signing, and constructing and signing one.
Raw timings and percentile summaries are written to `runs/benchmark/`.
Set `OPENSSL_PREFIX` if OpenSSL is installed in a nonstandard location.
