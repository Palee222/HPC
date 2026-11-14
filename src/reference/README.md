# Role 2 — sequential reference (`graphfeat`)

`graphfeat` is a deterministic, single-threaded C11 program. It computes the
11 structural features defined in [`docs/feature_spec.md`](../../docs/feature_spec.md)
for every Elliptic transaction. Its output is the ground truth that the
OpenMP and MPI versions (Role 3) must match.

## Build

```bash
make -C src/reference          # uses $CC, default cc; flags: -O2 -std=c11
```

There are no dependencies beyond the C standard library and `libm`. On the
cluster, load any GCC module first, for example `module load GCC`. Do **not**
add `-ffast-math` or `-march=native`, because the output must be identical
across runs and machines.

## Run on Elliptic

First generate Role 1's files (see `data/README.md` on the `data-evaluation`
branch). This creates `data/transaction_mapping.csv` and `data/node_edges.csv`.
Then run:

```bash
mkdir -p results
src/reference/graphfeat \
    --mapping data/transaction_mapping.csv \
    --edges   data/node_edges.csv \
    --out     results/graph_features.csv \
    --meta    results/graph_features.meta.json
```

Expected for the real dataset: 203,769 rows plus a header, 0 edges dropped,
and a SHA-256 equal to `results/role2/elliptic_graph_features.sha256`. The run
takes under a second, but the official baseline still goes through Slurm on
the cluster:

```bash
sbatch slurm/role2_reference.sbatch     # 1 core, 3 repetitions, records environment
```

The job writes its logs and `environment.txt` to `results/role2/job_<id>/`.
It also tells you whether the cluster output matches the recorded checksum.

Options: `--pr-iters K` (default 200) and `--damping A` (default 0.85).
Changing either one changes the contract, so do it only for experiments.

## Test

```bash
bash tests/run_tests.sh
```

The script checks that:

1. it builds without warnings;
2. the 3 golden fixtures match their hand-checked `expected.csv` (see
   `fixtures/README.md`);
3. the cleaning counts (cross-step, self-loop, duplicate) are correct;
4. three random graphs match an **independent Python oracle**
   (`tests/oracle.py`);
5. three repeated runs give **byte-identical** output;
6. bad input (duplicate IDs, ID gaps, unknown edge endpoints) is rejected.

## For Role 3 (parallel versions)

- Check parity with
  `python3 tests/compare.py reference.csv yours.csv`. Integers,
  `nbr_deg_mean` and `clustering` must match exactly, and PageRank must be
  within 1e-9.
- Make test graphs of any size with
  `python3 tests/gen_graph.py OUT_DIR --nodes N --edges M --seed S`.
- The kernels are separate functions in `features.c`, each with its own timer
  in the metadata. Two-hop and triangles carry the uneven, hub-driven work.
  PageRank is memory-bound and uses a pull layout that needs no atomics.
  Time steps are independent, so they make a natural MPI split.
- `feat_two_hop` uses one shared stamp array. A threaded version needs one
  per thread.

## Files

| File | Purpose |
|---|---|
| `graph.c/.h` | CSV loading, validation, edge cleaning, CSR build (out / in / undirected) |
| `features.c/.h` | the 6 feature kernels |
| `main.c` | command line, timers, CSV + JSON output |
| `../../tests/oracle.py` | independent Python implementation, for small graphs |
| `../../tests/compare.py` | parity checker |
| `../../tests/gen_graph.py` | seeded synthetic graph generator |
| `../../fixtures/` | hand-checked golden graphs |
