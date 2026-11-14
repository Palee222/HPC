# Role 2 golden fixtures

Each folder holds a small graph in the same format as Role 1's files, plus the
reference output (`expected.csv`). Every value below was worked out by hand
first and then compared with `graphfeat`. `tests/oracle.py` (an independent
Python version) also agrees with all of them.

| Fixture | What it tests |
|---|---|
| `tiny/` | duplicate edge, self-loop, cross-step edge, reciprocal pair, isolated node, a triangle, a directed cycle, an out-star, **shuffled mapping rows**, Windows `\r\n` line endings |
| `k4_mixed/` | a 4-clique with mixed edge directions, one reciprocal pair and a pendant node |
| `no_edges/` | an edges file with only a header, and isolated nodes in two steps |

Notation: `d(v)` is the undirected degree and `N(v)` is the set of undirected
neighbours (see `docs/feature_spec.md`).

## `tiny/` — hand check

Nodes 0–4 are in step 1, nodes 5–7 in step 2, and nodes 8–10 in step 3. The
mapping rows are in random order. A correct loader must use `node_id`, not
the row number.

Raw edges (13 rows):
`0→1, 1→2, 2→0, 2→3, 3→2, 0→1 (dup), 4→4 (self), 5→6, 6→7, 7→5, 8→9, 8→10, 3→8 (cross-step 1→3)`.

Cleaning: 1 cross-step, 1 self-loop and 1 duplicate are dropped. That leaves
**10 directed** edges and **9 undirected** edges (4 in step 1, 3 in step 2,
2 in step 3).

**Step 1** (`n_t = 5`). Directed edges: `0→1, 1→2, 2→0, 2→3, 3→2`.

| v | in | out | total | N(v) | d | nbr_deg_mean | max | two-hop set | tri | clust | comp |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1 (2) | 1 (1) | 2 | {1,2} | 2 | (2+3)/2 = 2.5 | 3 | {1,2,3} → 3 | 1 | 2·1/(2·1) = 1 | 4 |
| 1 | 1 (0) | 1 (2) | 2 | {0,2} | 2 | (2+3)/2 = 2.5 | 3 | {0,2,3} → 3 | 1 | 1 | 4 |
| 2 | 2 (1,3) | 2 (0,3) | 4 | {0,1,3} | 3 | (2+2+1)/3 = 1.667 | 2 | {0,1,3} → 3 | 1 | 2·1/(3·2) = 1/3 | 4 |
| 3 | 1 (2) | 1 (2) | 2 | {2} | 1 | 3 | 3 | {2,0,1} → 3 | 0 | 0 (d<2) | 4 |
| 4 | 0 | 0 | 0 | {} | 0 | 0 | 0 | {} → 0 | 0 | 0 | 1 |

Node 2 has `total_degree 4` but `undirected_degree 3`. The difference of 1 is
the reciprocal pair 2⇄3. Node 3 keeps its values because the cross-step edge
`3→8` was dropped.

**Step 2** (`n_t = 3`) is the directed 3-cycle `5→6→7→5`. Every node has
in = out = 1, d = 2, one triangle and clustering 1. PageRank is uniform: every
node gets `x = 1/3`, so the scaled value is **exactly 1** at every iteration.

**Step 3** (`n_t = 3`) is the out-star `8→9, 8→10`. Node 8 has in 0, out 2,
d 2, neighbour degrees {1,1} (mean 1, max 1), two-hop {9,10} → 2, and
clustering 0 because 9 and 10 are not linked. Nodes 9 and 10 have in 1, out 0,
d 1, neighbour degree 2, and two-hop {8, the other leaf} → 2.

PageRank for step 3, solved by hand. Let `a = x(8)` and `b = x(9) = x(10)`.
Nodes 9 and 10 are dangling, so `D = 2b`. With α = 0.85:

```
a = 0.15/3 + 0.85 · (2b/3)
b = 0.15/3 + 0.85 · (a/2 + 2b/3)
a + 2b = 1
=> b = 57/154,  a = 20/77
scaled: pagerank(8) = 3a = 60/77   = 0.779220779…
        pagerank(9) = 3b = 171/154 = 1.110389610…
```

`expected.csv` holds `0.77922077922077926` and `1.1103896103896105`, which
match.

**Step 1 PageRank** (0.9733, 1.0080, 1.8648, 0.9733, 0.1807) is too long to
solve by hand. It is checked against the independent oracle instead. Two
sanity checks still work by hand. First, the five values add up to 5, so the
mean is 1. Second, node 4 is isolated and dangling: it only receives the
teleport and dangling shares, so it gets the lowest score. Nodes 0 and 3 have
the same in-neighbour (node 2, which has out-degree 2), so they get the same
score.

## `k4_mixed/` — hand check

Edges: `0→1, 2→0, 0→3, 1→2, 3→1, 2→3, 3→2, 3→4`. Nodes 0–3 form a 4-clique
when directions are ignored, and node 4 hangs off node 3.

| v | in | out | d | nbr_deg_mean | max | two-hop | tri | clust |
|---|---|---|---|---|---|---|---|---|
| 0 | 1 | 2 | 3 | (3+3+4)/3 = 3.333 | 4 | 4 | 3 | 1 |
| 1 | 2 | 1 | 3 | 3.333 | 4 | 4 | 3 | 1 |
| 2 | 2 | 2 | 3 | 3.333 | 4 | 4 | 3 | 1 |
| 3 | 2 | 3 | 4 | (3+3+3+1)/4 = 2.5 | 3 | 4 | 3 | 2·3/(4·3) = 0.5 |
| 4 | 1 | 0 | 1 | 4 | 4 | 4 | 0 | 0 |

Every node has `component_size 5`. The reciprocal pair 2⇄3 gives node 2
`total 4` vs `d 3`, and node 3 `total 5` vs `d 4`.

## `no_edges/` — hand check

There are no edges, so every count is 0 and every component has size 1. Every
node is dangling, so PageRank is uniform within each step and the scaled value
is exactly 1.
