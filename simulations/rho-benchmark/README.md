# Local Reaction-Time Benchmark

This benchmark measures the local component of the reaction time used by the
matched multi-entry simulator. The timer starts after one entry has completely
read a target transaction `T` and ends when the same entry releases `F` through
a pre-established connection.

The trigger contains a fresh nonce on every iteration. Three fixed bot
implementations are measured:

- `prebuilt`: inspect `T` and release an already signed transaction. This is a
  strong-attacker lower bound for an `F` that does not need to contain data from
  `T`.
- `build`: construct a nonce-dependent payload, hash it, serialize it as a
  JSON-RPC `eth_sendRawTransaction` request, and send it without a new
  signature.
- `signed`: construct and hash the nonce-dependent payload, produce a
  deterministic secp256k1 signature with the repository's optimized
  `libsecp256k1`, serialize JSON-RPC, and send it.

Both the observation feed and submission channel are persistent TCP loopback
connections with `TCP_NODELAY`. Timing uses `CLOCK_MONOTONIC_RAW`. The output
distinguishes:

- `rho_send_call_us`: application visibility of `T` to the call that releases
  `F`;
- `rho_send_done_us`: application visibility to completion of the socket
  write;
- `rho_sink_us`: application visibility to complete receipt at the local mock
  submission endpoint.

Run a full local measurement with:

```bash
ITERATIONS=50000 WARMUP=5000 \
RESULT_DIR=simulations/rho-benchmark/results/local-run00 \
bash simulations/rho-benchmark/run_rho_benchmark.sh
```

`summary.csv` reports the mean and p50/p90/p95/p99/p99.9 for every stage. Raw
samples are retained for bootstrap analysis.

## Interpretation Boundary

This is a local processing and persistent-socket benchmark. It does not
measure public pending-transaction feed latency, a remote RPC provider queue,
TLS connection establishment, cross-region observer-to-entry fanout, or WAN
submission latency. Those quantities must be measured separately. In the
simulator, public-feed observation latency replaces the victim-to-entry path;
it must not be added to `rho` a second time.
