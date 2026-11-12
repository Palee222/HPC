#!/bin/bash
# Role 2 - sequential reference baseline on Elliptic (1 CPU core).
# Builds graphfeat, runs it 3 times, checks that the runs are byte-identical,
# and records the environment for reproducibility.
#
# Before submitting, generate Role 1's files on the cluster (data-evaluation branch):
#   export ELLIPTIC_DATA_DIR=/path/to/elliptic_bitcoin_dataset
#   python3 data/check_checksums.py && python3 data/build_mapping.py
# Then, from the repository root:
#   sbatch slurm/role2_reference.sbatch
#
# Override inputs with:  sbatch --export=ALL,DATA_DIR=/other/dir slurm/role2_reference.sbatch
#SBATCH --job-name=role2-reference
#SBATCH --output=results/role2/slurm_%j.out
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

# Compiler: use the cluster's GCC module. Change the version to one listed by `module avail GCC`.
module load "${GCC_MODULE:-GCC/12.3.0}" 2>/dev/null || echo "warning: could not load ${GCC_MODULE:-GCC/12.3.0}, using default cc"

DATA_DIR="${DATA_DIR:-data}"
OUT="results/role2/job_${SLURM_JOB_ID:-local}"
mkdir -p "$OUT"

{
    echo "date:       $(date -Iseconds)"
    echo "host:       $(hostname)"
    echo "git_commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "git_dirty:  $(git status --porcelain 2>/dev/null | grep -qv '^??' && echo yes || echo no)"
    echo "compiler:   $(cc --version | head -1)"
    echo "cflags:     -O2 -std=c11 -Wall -Wextra -Wpedantic"
    echo "slurm:      job=${SLURM_JOB_ID:-} nodes=${SLURM_JOB_NODELIST:-} cpus_per_task=${SLURM_CPUS_PER_TASK:-}"
    echo "cpu:        $(lscpu 2>/dev/null | grep 'Model name' | sed 's/.*: *//')"
    echo "mapping:    $DATA_DIR/transaction_mapping.csv $(sha256sum "$DATA_DIR/transaction_mapping.csv" | cut -c1-64)"
    echo "edges:      $DATA_DIR/node_edges.csv $(sha256sum "$DATA_DIR/node_edges.csv" | cut -c1-64)"
} | tee "$OUT/environment.txt"

make -C src/reference clean
make -C src/reference graphfeat

for r in 1 2 3; do
    srun --ntasks=1 --cpus-per-task=1 src/reference/graphfeat \
        --mapping "$DATA_DIR/transaction_mapping.csv" \
        --edges   "$DATA_DIR/node_edges.csv" \
        --out     "$OUT/graph_features_run$r.csv" \
        --meta    "$OUT/run$r.meta.json"
done

cmp "$OUT/graph_features_run1.csv" "$OUT/graph_features_run2.csv"
cmp "$OUT/graph_features_run1.csv" "$OUT/graph_features_run3.csv"
echo "3 runs byte-identical"

sha256sum "$OUT/graph_features_run1.csv" | tee "$OUT/graph_features.sha256"
echo "expected (laptop run): $(cut -c1-64 results/role2/elliptic_graph_features.sha256)"
if [ "$(cut -c1-64 "$OUT/graph_features.sha256")" = "$(cut -c1-64 results/role2/elliptic_graph_features.sha256)" ]; then
    echo "MATCH: cluster output is identical to the recorded reference"
else
    echo "DIFFERENT: compare with python3 tests/compare.py (pagerank may differ in the last bits across compilers)"
fi
cp "$OUT/graph_features_run1.csv" results/role2/graph_features.csv
grep -A10 timings_sec "$OUT"/run*.meta.json
