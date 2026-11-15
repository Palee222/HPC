#!/bin/bash
# Reserve one node with 16 CPU cores for the complete Role 3 benchmark matrix.
# Submit from the repository root after setting DATA_DIR and, when needed,
# MPI_MODULE to the cluster's MPI module name.
#   mkdir -p results/role3
#   sbatch --export=ALL,DATA_DIR=data,MPI_MODULE=OpenMPI ROLE3/role3_bench.sbatch
# The local cluster may require a different partition, module, or time limit.
#SBATCH --job-name=role3-bench
#SBATCH --output=slurm-role3-%j.out
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

if type module >/dev/null 2>&1; then
    module load "${GCC_MODULE:-GCC/12.3.0}"
    if [[ -n "${MPI_MODULE:-}" ]]; then
        module load "$MPI_MODULE"
    fi
fi

make -C src/reference graphfeat
make -C ROLE3 all
bash ROLE3/run_benchmarks.sh
