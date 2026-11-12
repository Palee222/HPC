# Role 2 — Graph Feature Specification (feature contract v1)

Owner: Role 2 (Sequential Graph Algorithms).
Status: v1, proposed to Roles 3, 4 and 5.

This file defines what the structural graph features mean, how the graph is
stored in memory, the exact output format, and how much time and memory each
feature costs. The sequential C program in `src/reference/` is the reference
implementation of this file. Any parallel version (Role 3) must give the same
output under the parity rules in section 6.

## 1. Inputs (from Role 1)

The reference reads two files that Role 1's scripts generate in `data/`.

| File | Columns used | Meaning |
|---|---|---|
| `transaction_mapping.csv` | `node_id`, `time_step` | Canonical node ID (0..N-1) and its time step (1..49) |
| `node_edges.csv` | `source_node`, `target_node` | Directed edge, money flows from source to target |

Columns are found by header name, so extra columns (such as
`transaction_id`) are ignored. Both `\n` and `\r\n` line endings are accepted,
because Python's `csv` module writes `\r\n`.

The program checks that:

- every `node_id` from 0 to N-1 appears exactly once in the mapping;
- every time step is a whole number from 0 to 100000;
- every edge endpoint is a valid node ID.

If any check fails, the program stops with an error and a non-zero exit code.
It never guesses.

## 2. The feature graph and the temporal rule

Role 1's policy says that features must not use information from future time
steps. Role 1 also checked that all 234,355 Elliptic edges join two
transactions in the same time step.

**Rule:** the features of a node with time step `t` are computed only on the
subgraph of time step `t`. Because no edge can reach another step, a node
can never see the future. The rule is therefore leakage-free by design, and it
holds even if a later dataset version adds edges that cross time steps.

Formally, for each time step `t`:

- `V_t` is the set of nodes with `time_step == t`, and `n_t = |V_t|`.
- `G_t` is the **directed simple graph** on `V_t`. It contains the edge
  `(u, v)` if the edge list has at least one row `u → v` with
  `step(u) == step(v) == t` and `u != v`.
- `U_t` is the **undirected simple graph** on `V_t`. It contains `{u, v}` if
  `(u, v)` or `(v, u)` is in `G_t`.
- `N(v)` is the set of neighbours of `v` in `U_t`, and `d(v) = |N(v)|`.

Edge cleaning, in this order:

1. **Cross-step edges** are dropped and counted (`cross_step_dropped`). The
   current Elliptic data has 0 of them.
2. **Self-loops** `u → u` are dropped and counted (`self_loops_dropped`).
3. **Duplicate** directed edges are merged into one and counted
   (`duplicates_dropped`).

The counts are written to the metadata file (section 5).

## 3. Features

All features are computed for **every** node, including nodes with unknown
labels. Labels are never read, so this file cannot leak labels.

| # | Column | Type | Definition | When undefined |
|---|---|---|---|---|
| 1 | `in_degree` | int | number of `u` with `(u, v)` in `G_t` | — |
| 2 | `out_degree` | int | number of `w` with `(v, w)` in `G_t` | — |
| 3 | `total_degree` | int | `in_degree + out_degree` | — |
| 4 | `undirected_degree` | int | `d(v)`, distinct neighbours in `U_t` | — |
| 5 | `nbr_deg_mean` | float | `(1/d(v)) · Σ_{u ∈ N(v)} d(u)` | 0 if `d(v)=0` |
| 6 | `nbr_deg_max` | int | `max_{u ∈ N(v)} d(u)` | 0 if `d(v)=0` |
| 7 | `two_hop_reach` | int | number of distinct nodes at undirected distance 1 or 2 from `v` (not counting `v`) | — |
| 8 | `triangles` | int | number of pairs `{u, w} ⊆ N(v)` with `{u, w}` in `U_t` | — |
| 9 | `clustering` | float | local clustering coefficient `2·triangles / (d(v)·(d(v)−1))` | 0 if `d(v)<2` |
| 10 | `component_size` | int | number of nodes in the connected component of `U_t` that contains `v` (at least 1) | — |
| 11 | `pagerank` | float | PageRank on `G_t` (section 3.1), multiplied by `n_t` | — |

Notes:

- `total_degree − undirected_degree` is the number of reciprocal pairs
  (`u → v` and `v → u`). Reciprocal flows can be a mixing signal.
- `nbr_deg_mean` is computed as an exact integer sum, divided once by `d(v)`.
  `clustering` is computed from exact integers too. Both are therefore
  bit-identical in any correct implementation that uses the same formula.

### 3.0 Direction, radius, missing values, overflow, scaling

- **Direction.** Features 1–3 and 11 use edge direction (`G_t`). Features 4–10
  ignore direction (`U_t`).
- **Radius.** The largest neighbourhood used is 2 hops (`two_hop_reach`,
  `triangles`). `component_size` and `pagerank` use the whole step subgraph,
  and they are still bounded by the time step.
- **Missing values.** There are none. Every node gets a finite value in every
  column. The "When undefined" column above gives the value to use (0) when a
  formula would divide by zero. NaN and Inf never appear, and a consumer may
  reject them.
- **Overflow.** Counts are bounded by `N` and are stored as `int32`.
  `triangles` can reach `d²/2` and is stored as `int64`. Edge offsets are
  `int64`. The loader stops if `N ≥ 2³¹−1`.
- **No normalization, no labels.** The values are raw. Scaling or
  standardization is the consumer's job (Role 4/5), and it must be fitted on
  the training mask only (Role 1 policy). Labels are never read.
- **Prediction-time availability.** A node at step `t` uses only the edges of
  step `t`. Under a prospective policy, those edges exist when step `t` is
  scored. Under a transductive policy, the rule is stricter than needed. So
  the features are valid under both policies Role 1 is still choosing between.

### 3.1 PageRank, exact recipe

PageRank runs separately in each time step, on the directed graph `G_t`.

- Damping `α = 0.85`. The number of iterations is fixed at `K = 200`, with no
  early stopping. A fixed count keeps serial and parallel runs comparable.
  With 100 iterations, a hub-heavy synthetic graph of Elliptic size still had
  an L1 change of 3e-8. With 200 it is close to machine precision, and the
  cost is only about 0.1 s more.
- Start: `x_0(v) = 1 / n_t`.
- Dangling mass: `D_t = Σ_{u ∈ V_t, out_degree(u)=0} x_k(u)`, summed in
  ascending node ID.
- Update:
  `x_{k+1}(v) = (1−α)/n_t + α · ( Σ_{u:(u,v) ∈ G_t} x_k(u)/out_degree(u) + D_t/n_t )`
  where the inner sum runs over in-neighbours in ascending node ID.
- Output: `pagerank(v) = n_t · x_K(v)`.

Scaling by `n_t` makes the mean PageRank of every step equal to 1. Values can
then be compared across steps of different sizes. A value above 1 means "more
central than average in its step".

The metadata file records the final L1 change
`Σ_v |x_K(v) − x_{K−1}(v)|` as `pagerank_final_l1_delta`, so anyone can check
convergence.

## 4. Adjacency layout (CSR)

The graph is stored as three CSR (compressed sparse row) structures. Each one
is an offset array and an index array:

```
row v's neighbours = idx[ ptr[v] .. ptr[v+1]-1 ],  sorted ascending
```

| Structure | Rows | Used by |
|---|---|---|
| `out_ptr`, `out_idx` | out-neighbours in `G_t` | out-degree, PageRank (dangling test) |
| `in_ptr`, `in_idx` | in-neighbours in `G_t` | in-degree, PageRank (pull update) |
| `und_ptr`, `und_idx` | `N(v)` in `U_t` (each edge stored twice) | degrees 4–6, two-hop, triangles, components |

Types: node IDs are `int32_t`, and offsets are `int64_t` so that graphs with
more than 2³¹ edges still work. Rows are sorted. This is needed for the merge
intersection in triangle counting and makes all loops visit nodes in a fixed
order.

Build: edges are bucketed by source with a counting sort (O(N+M)). Each row is
then sorted and deduplicated. `in` is built from `out` with a second counting
sort, so its rows come out sorted for free. `und` is a per-row merge of the
`out` and `in` rows.

Why pull (in-CSR) for PageRank: each node writes only its own new value. A
parallel loop over nodes therefore needs no atomics or locks. This is a hint
for Role 3, not a requirement.

## 5. Output contract

### 5.1 Feature CSV

- Header (exact):
  ```
  node_id,time_step,in_degree,out_degree,total_degree,undirected_degree,nbr_deg_mean,nbr_deg_max,two_hop_reach,triangles,clustering,component_size,pagerank
  ```
- One row per node, exactly N rows, sorted by `node_id` from 0 to N-1. Row `i`
  (after the header) is node `i`. The file is **row-aligned** with the
  canonical node IDs and with Role 1's feature matrix.
- Integers are printed in plain decimal. Floats use `%.17g`, which keeps every
  bit of a double.
- The separator is a comma, line endings are `\n`, and the encoding is ASCII.
  There is no index column and no trailing comma.

### 5.2 Metadata JSON (optional, `--meta`)

The metadata file records `feature_contract_version`, the input row counts,
the edge-cleaning counts, the edge counts after cleaning, the number of time
steps, the PageRank settings, `pagerank_final_l1_delta`, `peak_rss_mib` (peak
resident memory) and the wall-clock time of each kernel in seconds. Timings
change from run to run, so the determinism check compares the CSV only.

### 5.3 Implementation language

The reference is plain **C11** with no dependencies. The Roles text says "C",
and the brief says "C++". The behaviour is defined by this file, not by the
language, so a C++ parallel version is fine as long as it passes section 6.
The group should confirm the language at Gate B.

## 6. Parity rules for parallel versions (Role 3)

A parallel output **passes parity** with the reference when:

1. the header and the number of rows are identical, and `node_id` is in the
   same order;
2. all integer columns, plus `nbr_deg_mean` and `clustering`, are **exactly
   equal**;
3. `pagerank` is within absolute difference `1e-9` on every row. Parallel
   reductions (for example `D_t`) may add numbers in a different order, so
   tiny floating-point differences are allowed here only.

Check it with:

```
python3 tests/compare.py reference.csv parallel.csv
```

The reference itself must give **byte-identical** output on repeated runs.
`tests/run_tests.sh` checks this.

## 7. Complexity

Notation: `N` nodes, `M` directed edges after cleaning, `M_u` undirected
edges, `K` PageRank iterations, `S = Σ_v d(v)²`.

| Step | Time | Extra memory |
|---|---|---|
| Read + validate inputs | O(N + M_raw) | O(N + M_raw) |
| Build out/in CSR (counting sort + row sort + dedup) | O(N + M_raw + Σ d·log d) | 2·(8(N+1) + 4M) bytes |
| Build undirected CSR (row merge) | O(N + M) | 8(N+1) + 8M_u bytes |
| Degrees 1–4 | O(N) | — |
| `nbr_deg_mean`, `nbr_deg_max` | O(N + M_u) | — |
| `two_hop_reach` (stamp array) | O(Σ_{v} Σ_{u∈N(v)} d(u)) = O(S) | 4N bytes |
| `triangles` (sorted merge intersection) | O(Σ_{\{u,v\}} (d(u)+d(v))) = O(S) | — |
| `clustering` | O(N) | — |
| `component_size` (BFS) | O(N + M_u) | 8N bytes |
| `pagerank` | O(K · (N + M)) | 16N bytes |

Total time is **O(N + M_raw·log + S + K·(N+M))**, and total memory is
**O(N + M)**.

For Elliptic (N = 203,769, M ≈ 234,355), every array together is only a few
megabytes, so the whole graph fits in the L2/L3 cache of one node. Two points
matter for performance:

- **`S` is driven by hubs.** One node with degree `d` costs about `d²` in
  two-hop and triangle counting. Transaction graphs have a few very
  high-degree nodes, so per-node work is very uneven. A parallel loop with
  static scheduling will therefore suffer from load imbalance. Dynamic or
  guided scheduling is the obvious fix to try (Role 3's choice).
- **PageRank is memory-bound.** Each iteration reads `x[u]` at random
  positions through `in_idx`. Its speed is limited by memory access, not
  arithmetic.

### 7.1 Measured (sequential reference, synthetic graphs)

These are early measurements on a laptop (Apple M3, `cc -O2`, one core). The
graphs come from `tests/gen_graph.py --seed 42 --steps 49`, which has
Zipf-like hubs. They are **not** Elliptic itself and **not** SciTech. The
official baseline must be rerun on the cluster through Slurm (Role 3 / 5).

| Graph | Nodes | Edges read | Max degree | load+CSR | two_hop | triangles | pagerank (200 it) | total | peak RSS |
|---|---|---|---|---|---|---|---|---|---|
| Elliptic-sized | 203,769 | 234,545 | 945 | 0.12 s | 0.03 s | 0.19 s | 0.21 s | 0.68 s | 27 MiB |
| 10× | 2,037,690 | 2,345,943 | 8,626 | 1.25 s | 2.85 s | 9.53 s | 2.19 s | 17.3 s | 260 MiB |

What this shows:

- PageRank, CSR build and memory grow about **10×** when the graph grows
  10×. That is linear, as predicted by O(K(N+M)) and O(N+M).
- Two-hop (86×) and triangles (50×) grow much faster than the graph. The
  maximum degree grew about 9× (945 → 8,626), and their cost follows
  `S = Σ d²`. A few hub nodes dominate. This is the expected load-imbalance
  hotspot for a parallel version.
- At Elliptic size the reference finishes in under a second. Parallel
  speedup there will be limited by fixed costs such as I/O and thread start-up.
  This supports the proposal's plan to use larger synthetic graphs for
  scaling.

### 7.2 Measured on the real Elliptic graph

Input: the official Kaggle "Elliptic Data Set". All three file checksums match
`data/checksums.json`, and Role 1's scripts generated the mapping and edges.
The run was on a laptop (Apple M3, one core) with 3 repetitions. The outputs
were byte-identical, with SHA-256 `464557e0…0aed`. The full value is in
`results/role2/elliptic_graph_features.sha256`.

| | Value |
|---|---|
| nodes / edges / steps | 203,769 / 234,355 / 49 |
| edges dropped (cross-step, self-loop, duplicate) | 0 / 0 / 0 (agrees with Role 1's audit) |
| reciprocal pairs | 0, so directed edges = undirected edges |
| max in / out / undirected degree | 284 / 472 / 473 |
| triangles in the whole graph | 8,623 (7,202 nodes are in at least one) |
| components | 49, exactly one per time step (1,089 to 7,880 nodes) |
| PageRank mean per step | 1 within 1e-14; final L1 change 6.8e-17 |
| total time / peak memory | 0.47 s / 27 MiB |
| time per kernel | PageRank 0.19 s, load+CSR 0.13 s, write 0.13 s, triangles 0.015 s, two-hop 0.008 s |

The real graph has far fewer hubs than the synthetic one, because
`Σd² = 4.9M`. Triangle and two-hop counting are therefore cheap here, and
**PageRank dominates the compute**. I/O (reading and writing the CSVs) takes
about half of the total. This sets realistic expectations for Role 3. On
Elliptic itself, strong scaling is limited by I/O and by the small amount of
work, as Amdahl's law predicts. The larger synthetic graphs in 7.1 are the
better test for scaling the hub-heavy kernels.

These are laptop numbers. The official baseline is produced on SciTech with
`sbatch slurm/role2_reference.sbatch`.

Because Elliptic is small, the time steps (up to about 8,000 nodes each) are
independent. MPI can split by time step without any halo exchange, which is
another hint for Role 3.

## 8. Versioning

Any change to a definition, the column order or the PageRank recipe raises
the version (`feature_contract_version` in the metadata) and must be agreed
with Roles 3, 4 and 5 first.
