# Causal Overtaking Experiments

## Setup

Requires Python 3.10–3.12 with `venv` and `pip`, and a C++17 compiler on Linux
or macOS.

```bash
bash setup.sh
bash simulations/run.sh smoke
```

Python dependencies are installed in `.venv`. Required network inputs are
included; see [data sources](simulations/DATA.md). The `smoke` command runs a
reduced configuration and checks Python/C++ Unspool agreement.

## Reproduce

```bash
bash simulations/run.sh all
```

Defaults: 20 deployments, 20 workloads per deployment, 300 background
transactions, and 20 Unspool permutations. The capability experiment uses
100 pair trials per deployment. Bootstrap intervals, where computed, use
10,000 resamples. Results are written to `runs/`.

| Experiment | Command |
| --- | --- |
| Committee sizes 9, 21, 101 | `bash simulations/run.sh attack` |
| Observer/report capability ablation | `bash simulations/run.sh capability` |
| Singleton/union composition | `bash simulations/run.sh composition` |
| Reaction-delay sensitivity | `bash simulations/run.sh rho` |
| Placement sensitivity | `bash simulations/run.sh placement` |
| One/five observers with increasing entries | `bash simulations/run.sh entries` |
| Reporter selection | `bash simulations/run.sh reporters` |
| Directional-delay sensitivity | `bash simulations/run.sh asymmetry` |
| ESnet measured one-way delay | `bash simulations/run.sh owd` |
| Gossip probability and batching | `bash simulations/run.sh gossip` |
| Regional Ordering | `bash simulations/run.sh regional` |
| Adapted attack policies | `bash simulations/run.sh adaptive` |

The `attack` command also includes reporter-selection and directional-delay
experiments. Generate tables and figures from completed runs with:

```bash
bash simulations/run.sh summarize
```

For a smaller run:

```bash
DEPLOYMENTS=2 TRIALS=3 WORKERS=2 RESULT_ROOT=runs/quick \
  bash simulations/run.sh regional
```

The local processing benchmark runs separately and requires `make` and OpenSSL
development libraries:

```bash
bash simulations/run.sh benchmark
```

## Code provenance and license

This artifact builds on the [Ambush artifact](https://github.com/NetSP-KAIST/the-ambush-attack)
for Park et al., *On Frontrunning Risks in Batch-Order Fair Systems for
Blockchains* (CCS 2025).

| Component | Origin and changes |
| --- | --- |
| `simulations/include/hotstuff/aequitas.h` | Reused unchanged from Ambush; implements the Themis graph, SCC, and Unspool routines. |
| `simulations/src-unspooling_algo/unspool.cpp` | Adapted from Ambush. Additions include explicit quorum, pair-order and public-order inputs, and optional interval-receipt and hard-edge inputs. |
| `simulations/src-gen_orderings/` and `simulations/experiments/` | Experiment code added for this study: causal multi-entry workloads, capability and delay controls, regional ordering, and statistical summaries. The Python Themis replay implements the upstream graph and Unspool algorithms. |
| `setup.sh`, `simulations/run.sh`, `simulations/runtime.sh`, and `simulations/rho-benchmark/` | Setup, experiment orchestration, and processing benchmarks added for this study. |
| `third_party/secp256k1/` | Vendored dependency, retaining its upstream [MIT license](third_party/secp256k1/COPYING). |

Modification notice, 2026-11-18: this derivative artifact adds the replay
interfaces and experiments listed above. Upstream copyright and license notices
are retained. The artifact is distributed under [GPL-3.0](LICENSE), subject to
the retained file-specific notices. Data sources are listed in
[simulations/DATA.md](simulations/DATA.md).
