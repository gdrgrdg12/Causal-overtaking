# Causal Overtaking Reproducibility Artifact

## Validated Environment

This artifact was validated on **Ubuntu 22.04.5 LTS (Jammy), AArch64
(ARM64)** with Linux 5.15, Python 3.10.12, G++ 11.4.0, and GNU Make 4.3.

The repository contains reproducible source code and the two network-delay
inputs required by the experiments. It intentionally contains no paper PDF,
figures, generated tables, benchmark samples, or precomputed experiment
results.

## Setup

Install the system dependencies on a clean Ubuntu 22.04 machine:

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential python3-venv libssl-dev autoconf automake libtool

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

The Python dependencies are pinned in `requirements.txt`. The local reaction
time benchmark builds the vendored `third_party/secp256k1` source and links
against OpenSSL and pthreads.

## Quick Run

Run a small end-to-end scaling job. This checks the pipeline without running a
paper-sized measurement:

```bash
RESULT_ROOT=/tmp/causal-overtaking-smoke \
STAGES=scaling \
DEPLOYMENTS=1 \
TRIALS_PER_DEPLOYMENT=1 \
PAIR_TRIALS_PER_DEPLOYMENT=1 \
PUBLIC_REPETITIONS=2 \
BOOTSTRAP_REPETITIONS=20 \
WORKERS=1 \
bash simulations/run_paper_evaluation.sh
```

## Full Reproduction

Run these commands from the repository root. Defaults match the manuscript:
20 deployments, 20 final trials per deployment, 100 pair trials per
deployment, 20 public repetitions, 10,000 bootstrap repetitions, and at most
five workers.

```bash
source .venv/bin/activate

RESULT_DIR=simulations/rho-benchmark/results/article-run00 \
ITERATIONS=50000 WARMUP=5000 BUILD_JOBS=5 \
bash simulations/rho-benchmark/run_rho_benchmark.sh

bash simulations/run_paper_evaluation.sh
bash simulations/run_paper_gossip_scaling.sh
bash simulations/run_paper_practical_rho.sh
```

The scripts write all generated outputs below
`simulations/data-causality-comparison/` and
`simulations/rho-benchmark/results/`. Both directories are ignored by Git.
Runs are deterministic for the default seed, but local reaction-time values
remain machine-dependent. Full reproduction is compute-intensive; the quick
run above is the recommended first check.

Individual stages can be selected with, for example,
`STAGES=scaling,rho bash simulations/run_paper_evaluation.sh`. Supported stages
are `scaling`, `composition`, `budget`, `placement`, `rho`, and `guard`.

## Experiment Coverage

| Manuscript experiment | Entry point |
| --- | --- |
| Section 4 committee scaling (Figure 2) | `STAGES=scaling bash simulations/run_paper_evaluation.sh` |
| Section 4 observer composition | `STAGES=composition bash simulations/run_paper_evaluation.sh` |
| Section 4 corruption budget (Figure 3) | `STAGES=budget bash simulations/run_paper_evaluation.sh` |
| Section 4 reaction time (Figure 4) | `STAGES=rho bash simulations/run_paper_evaluation.sh` and `run_paper_practical_rho.sh` |
| Section 4 delayed gossip sensitivity | `bash simulations/run_paper_gossip_scaling.sh` |
| Appendix placement sensitivity (Table 2) | `STAGES=placement bash simulations/run_paper_evaluation.sh` |
| Section 6 guard and appendix ablations (Tables 1 and 3) | `STAGES=guard bash simulations/run_paper_evaluation.sh` |

## Delay Assumption

The AWS input is an RTT sample table and the main experiments use `RTT / 2` as
an OWD proxy (`--one_way_mode half`). The WonderProxy snapshot was also derived
from hourly ping summaries by dividing RTT statistics by two. These values are
not claimed to be directly measured one-way delays.

## License

The project is licensed under GPL-3.0; see `LICENSE`. The vendored secp256k1
library retains its own MIT license in `third_party/secp256k1/COPYING`.
