#!/usr/bin/env python3
"""Independent Python oracle for the Role 2 feature contract.

Written straight from docs/feature_spec.md using sets and dicts. It shares no
code or data structures with the C reference. Use it on small graphs only,
because it is slow.

    python3 tests/oracle.py MAPPING EDGES OUT [--pr-iters 200] [--damping 0.85]
"""
import argparse
import csv
from collections import defaultdict, deque

HEADER = ["node_id", "time_step", "in_degree", "out_degree", "total_degree",
          "undirected_degree", "nbr_deg_mean", "nbr_deg_max", "two_hop_reach",
          "triangles", "clustering", "component_size", "pagerank"]


def load(mapping_path, edges_path):
    step = {}
    with open(mapping_path, newline="") as f:
        for row in csv.DictReader(f):
            step[int(row["node_id"])] = int(row["time_step"])
    n = len(step)
    assert sorted(step) == list(range(n)), "node_id must be 0..N-1"
    edges = set()
    with open(edges_path, newline="") as f:
        for row in csv.DictReader(f):
            u, v = int(row["source_node"]), int(row["target_node"])
            if step[u] != step[v] or u == v:
                continue
            edges.add((u, v))
    return n, step, edges


def compute(n, step, edges, iters, alpha):
    out_n = defaultdict(set)
    in_n = defaultdict(set)
    und = defaultdict(set)
    for u, v in edges:
        out_n[u].add(v)
        in_n[v].add(u)
        und[u].add(v)
        und[v].add(u)
    deg = {v: len(und[v]) for v in range(n)}

    rows = []
    for v in range(n):
        nb = und[v]
        d = len(nb)
        reach = set(nb)
        for u in nb:
            reach |= und[u]
        reach.discard(v)
        nbl = sorted(nb)
        tri = sum(1 for i in range(d) for j in range(i + 1, d) if nbl[j] in und[nbl[i]])
        rows.append({
            "in_degree": len(in_n[v]), "out_degree": len(out_n[v]),
            "total_degree": len(in_n[v]) + len(out_n[v]), "undirected_degree": d,
            "nbr_deg_mean": (sum(deg[u] for u in nb) / d) if d else 0.0,
            "nbr_deg_max": max((deg[u] for u in nb), default=0),
            "two_hop_reach": len(reach), "triangles": tri,
            "clustering": (2.0 * tri / (d * (d - 1))) if d >= 2 else 0.0,
        })

    # components by BFS
    comp_size = {}
    seen = set()
    for s in range(n):
        if s in seen:
            continue
        comp, q = [s], deque([s])
        seen.add(s)
        while q:
            x = q.popleft()
            for y in und[x]:
                if y not in seen:
                    seen.add(y)
                    comp.append(y)
                    q.append(y)
        for x in comp:
            comp_size[x] = len(comp)

    # PageRank per step (dict-based power iteration)
    by_step = defaultdict(list)
    for v in range(n):
        by_step[step[v]].append(v)
    pr = {}
    for t, nodes in by_step.items():
        nt = len(nodes)
        x = {v: 1.0 / nt for v in nodes}
        for _ in range(iters):
            dang = sum(x[v] for v in nodes if not out_n[v])
            x = {v: (1 - alpha) / nt
                 + alpha * (sum(x[u] / len(out_n[u]) for u in in_n[v]) + dang / nt)
                 for v in nodes}
        for v in nodes:
            pr[v] = x[v] * nt

    for v in range(n):
        rows[v].update(node_id=v, time_step=step[v],
                       component_size=comp_size[v], pagerank=pr[v])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mapping")
    ap.add_argument("edges")
    ap.add_argument("out")
    ap.add_argument("--pr-iters", type=int, default=200)
    ap.add_argument("--damping", type=float, default=0.85)
    a = ap.parse_args()
    n, step, edges = load(a.mapping, a.edges)
    rows = compute(n, step, edges, a.pr_iters, a.damping)
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(HEADER)
        for r in rows:
            w.writerow([repr(r[c]) if isinstance(r[c], float) else r[c] for c in HEADER])


if __name__ == "__main__":
    main()
