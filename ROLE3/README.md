# Role 3: OpenMP, MPI, and CPU performance

## Build

On a Linux cluster with GCC and an MPI compiler wrapper:

```bash
module load GCC/12.3.0
module load <your-MPI-module>
make -C src/reference graphfeat
make -C ROLE3
```

The Role 3 Makefile builds:

- `ROLE3/openmp_graphfeat`: OpenMP implementation.
- `ROLE3/mpi_graphfeat`: MPI plus OpenMP implementation. Use one thread per
  rank for MPI-only runs and multiple threads per rank for hybrid runs.
- `ROLE3/speedupplots`: standard-C utility that summarizes timings and writes
  SVG plots; it has no plotting-library dependency.

On macOS with Homebrew `open-mpi` and `libomp`, use:

```bash
make -C ROLE3 \
  OMPFLAGS="-Xpreprocessor -fopenmp -isystem $(brew --prefix libomp)/include" \
  OMPLIBS="-L$(brew --prefix libomp)/lib -lomp"
```

The MPI program loads the canonical graph on rank 0 and broadcasts its CSR
arrays to the other ranks. It assigns whole time steps to ranks with a greedy
largest-estimated-work-first policy. The estimate includes degree-squared
work for two-hop/triangle kernels and per-node/per-edge PageRank work. Each
rank computes the features for its assigned steps; the ranks combine the
result arrays at rank 0. The graph is therefore replicated in memory on every
rank. MPI communication and repeated graph storage are part of the measured
costs, not hidden scaling benefits. Metadata times graph loading/broadcast and
the result reduction separately, so communication costs can be distinguished
from the per-kernel timings.

Within a rank, OpenMP handles independent node work. The triangle and two-hop
loops use dynamic scheduling for uneven hub-driven work. Two-hop has a
private stamp array per thread. Connected components use a per-rank BFS over
that rank's time steps. PageRank is computed by whole time step, so no
per-iteration MPI exchange is needed. MPI calls stay on the main thread.

## Correctness check

Generate a deterministic test graph, build a reference output, and compare
the parallel output:

```bash
python3 tests/gen_graph.py /tmp/role3-graph --nodes 2000 --edges 26000 \
  --steps 7 --seed 42

src/reference/graphfeat \
  --mapping /tmp/role3-graph/transaction_mapping.csv \
  --edges /tmp/role3-graph/node_edges.csv \
  --out /tmp/role3-graph/reference.csv

OMP_NUM_THREADS=4 ROLE3/openmp_graphfeat \
  --mapping /tmp/role3-graph/transaction_mapping.csv \
  --edges /tmp/role3-graph/node_edges.csv \
  --out /tmp/role3-graph/openmp.csv

OMP_NUM_THREADS=2 mpiexec -n 2 ROLE3/mpi_graphfeat \
  --mapping /tmp/role3-graph/transaction_mapping.csv \
  --edges /tmp/role3-graph/node_edges.csv \
  --out /tmp/role3-graph/mpi.csv

python3 tests/compare.py /tmp/role3-graph/reference.csv /tmp/role3-graph/openmp.csv
python3 tests/compare.py /tmp/role3-graph/reference.csv /tmp/role3-graph/mpi.csv
```

The checker requires identical integer features, `nbr_deg_mean`, and
`clustering`; PageRank may differ by at most `1e-9`.

## Benchmarks

Run the complete matrix inside a Slurm allocation with at least 16 CPU cores
on one node. Generate Role 1's canonical input files first; alternatively,
set `DATA_DIR` to a directory containing generated test inputs.

```bash
mkdir -p results/role3
sbatch --export=ALL,DATA_DIR=data,MPI_MODULE=<your-MPI-module> \
  ROLE3/role3_bench.sbatch
```

The script runs three repetitions by default (`REPEATS=3` can override this)
for:

- OpenMP: 1, 2, 4, 8, and 16 threads.
- MPI: 1, 2, 4, 8, and 16 ranks.
- Hybrid: 1 rank x 1 thread, then 1x2, 2x2, 4x2, and 8x2. The CPU count is
  ranks multiplied by threads per rank, so those configurations use 1, 2, 4,
  8, and 16 CPU cores respectively.

Every parallel run is compared with the reference before its timings are
accepted. The results directory contains `environment.txt`, the run plan,
raw per-run logs and metadata, one feature CSV per configuration, the
measurements, and the generated `scaling.csv`, `speedup.svg`, and
`efficiency.svg`.

Speedup is each mode's one-core median time divided by its configuration's
median time. Efficiency is speedup divided by CPU count. Do not treat
synthetic-graph timings as official Elliptic results; keep the dataset and
cluster environment recorded with the logs.

---------------------------------------------------------------------------------------------
The goal is to calculate the same graph features as Role 2, only faster by using multiple CPU cores. 

- OpenMP uses threads in one process. Threads can work on different nodes at the same time. For two-hop reach, each thread needs its own stamp array; otherwise threads could overwrite each other’s work.

- MPI uses separate processes, called ranks. This implementation assigns whole time steps to ranks, balances them using an estimate of their work, then combines their results. The graph is copied to each rank, which simplifies computing a time step without repeated communication, but uses more memory and makes graph broadcast and result collection costs part of the performance story.

- Hybrid combines the two: multiple MPI ranks, with multiple OpenMP threads per rank. CPU count is ranks × threads per rank—for example, 4 ranks with 2 threads each use 8 CPUs.

- Parity checks that the faster programs still answer the same question: exact matches for integer features, nbr_deg_mean, and clustering.

- Scaling tests repeat timings so you can compare medians. Speedup is one-CPU time / parallel time; efficiency is speedup / CPU count. Ideally, speedup grows with CPU count and efficiency stays near 1. Hubs can create uneven work, PageRank can be memory-bound, and MPI communication can outweigh saved compute time—those are the kinds of results you’ll explain.

- The benchmark script compares every run with the reference before recording it. Its default is three repetitions and it writes raw logs, metadata, a timing summary, and speedup/efficiency plots.