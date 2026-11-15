# Parallel Financial Fraud Detection

**IE University · High Performance Computing · Group 7**

Can parallel computing make financial fraud detection faster without making
it worse? We use the [Elliptic Bitcoin Dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set),
which has 203,769 transactions, 234,355 links and 49 time steps. On top of it
we build two things:

1. **Graph analytics in C**: structural features for every transaction
   (degrees, two-hop reach, triangles, PageRank and more). There is a
   sequential reference, plus OpenMP and MPI versions that must give the same
   numbers.
2. **A graph neural network (GNN)**: it classifies transactions as illicit or
   licit, first with the original features and then with the graph features
   added.

We measure both sides: runtime, speedup and scaling for the HPC part, and
precision, recall and F1 for detection. All heavy runs go through Slurm on the
SciTech cluster.

```
raw Elliptic CSVs ──▶ Role 1: validate, map IDs, chronological splits
                           │
                           ▼
                      Role 2: feature spec + sequential C reference ──▶ golden fixtures
                           │
                           ▼
                      Role 3: OpenMP / MPI versions (must match Role 2 exactly)
                           │
                           ▼
                      Role 5: feature → model adapter ──▶ Role 4: GNN (baseline vs + graph features)
                                                              │
                                                              ▼
                                                 Role 6: evidence checks, figures, report
```

## Team and status

| Role | Owner | What it covers | Where | Status |
|---|---|---|---|---|
| 1 · Data & evaluation | Zaid | dataset checks, ID mapping, chronological splits, scoring function | [`data/`](data/README.md) | ✅ merged |
| 2 · Sequential graph algorithms | Nikoloz | feature spec, C reference, golden fixtures, complexity | [`docs/feature_spec.md`](docs/feature_spec.md), [`src/reference/`](src/reference/README.md) | ✅ merged |
| 3 · OpenMP / MPI & CPU performance | Laura | parallel versions, scaling benchmarks, plots | [`ROLE3/`](ROLE3/README.md) | ✅ on `main`, parity checked by Role 2 |
| 4 · GNN & model experiments | Sergio | two-layer GNN, training, distributed runs | — | ⏳ in progress |
| 5 · Interfaces & pipeline | Owen | environment, validators, job templates, end-to-end run | — | ⏳ in progress |
| 6 · Evidence & report | Hari | provenance checks, figures, report, slides | — | ⏳ in progress |

## Repository layout

```
data/            Role 1: dataset validation, ID mapping, splits, evaluation
docs/            feature specification (and later the report and EuroHPC plan)
src/reference/   Role 2: sequential C reference (graphfeat)
ROLE3/           Role 3: OpenMP and MPI versions, benchmark scripts, plots
fixtures/        small hand-checked graphs with expected outputs
tests/           test suite, Python oracle, parity checker, graph generator
slurm/           Slurm job scripts
results/         run metadata, checksums and benchmark outputs
```

The dataset itself is **not** in the repository. Download it as shown below.

## Quick start

**1. Get the data.** Download the official
[Elliptic Data Set](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)
by Elliptic and unzip it outside the repo. You should have a folder
`elliptic_bitcoin_dataset/` with three CSV files.

**2. Validate it and build the canonical graph** (Role 1):

```bash
export ELLIPTIC_DATA_DIR=/path/to/elliptic_bitcoin_dataset
python3 data/check_checksums.py      # must say CHECKSUM VALIDATION PASSED
python3 data/validate_data.py
python3 data/build_mapping.py        # writes data/transaction_mapping.csv, data/node_edges.csv
python3 data/split_data.py
```

**3. Build and test the reference** (Role 2):

```bash
make -C src/reference
bash tests/run_tests.sh              # expect: passed: 11  failed: 0
```

**4. Compute the graph features:**

```bash
mkdir -p results/role2
src/reference/graphfeat \
    --mapping data/transaction_mapping.csv \
    --edges   data/node_edges.csv \
    --out     results/role2/graph_features.csv \
    --meta    results/role2/graph_features.meta.json
```

The output has 203,769 rows, and its SHA-256 must match
`results/role2/elliptic_graph_features.sha256`. On the cluster, use
`sbatch slurm/role2_reference.sbatch` instead.

**5. Parallel versions and benchmarks** (Role 3): see
[`ROLE3/README.md`](ROLE3/README.md). Any parallel output can be checked
against the reference with:

```bash
python3 tests/compare.py results/role2/graph_features.csv your_output.csv
```

## Key results so far

- **Data:** checksums verified, with 4,545 illicit, 42,019 licit and 157,205
  unknown transactions. Every edge stays inside one time step. The splits are
  train 1–34, validation 35–41 and test 42–49.
- **Sequential reference** (laptop, 1 core): all features for the full graph
  in **0.47 s** using **27 MiB**. Repeated runs give byte-identical output.
  PageRank takes most of the compute, and reading and writing the CSVs takes
  about half the total time.
- **Parallel versions:** the OpenMP and MPI outputs are byte-identical to the
  reference on every fixture, synthetic graph and the real Elliptic data.
  Official scaling numbers will come from SciTech.

Details and complexity analysis are in
[`docs/feature_spec.md`](docs/feature_spec.md).

## Rules we follow

- **No future data.** Features use only edges inside a transaction's own time
  step, and labels are never used as inputs.
- **Unknown ≠ licit.** Unlabelled transactions are left out of training and
  scoring.
- **Parity before speed.** A parallel run only counts if it matches the
  reference (`tests/compare.py`).
- **Everything through Slurm** on the cluster, with raw logs kept.
