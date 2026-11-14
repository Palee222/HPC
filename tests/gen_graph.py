#!/usr/bin/env python3
"""Seeded synthetic graph in Role 1's file format.

Used for the random oracle tests, and by Role 3 for scaling runs on graphs
larger than Elliptic. Targets have a skewed (hub-heavy) popularity, like
transaction graphs. The generator also adds some duplicates, self-loops and
(optionally) cross-step edges, so the cleaning code is exercised.

    python3 tests/gen_graph.py OUT_DIR --nodes 2000 --edges 2600 --steps 49 --seed 1
writes OUT_DIR/transaction_mapping.csv and OUT_DIR/node_edges.csv
"""
import argparse
import os
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--nodes", type=int, default=2000)
    ap.add_argument("--edges", type=int, default=2600)
    ap.add_argument("--steps", type=int, default=49)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--hub-exponent", type=float, default=1.2,
                    help="larger = more skewed target popularity")
    ap.add_argument("--noise", type=float, default=0.02,
                    help="fraction of extra duplicate / self-loop / cross-step edges")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    os.makedirs(a.out_dir, exist_ok=True)

    step = [1 + (v * a.steps) // a.nodes for v in range(a.nodes)]
    members = {}
    for v, t in enumerate(step):
        members.setdefault(t, []).append(v)

    # Per-step Zipf-like weights: rank r gets weight 1/(r+1)^s.
    # Cumulative weights are precomputed once, so each draw is O(log n_t).
    cum = {}
    for t, vs in members.items():
        acc, c = 0.0, []
        for r in range(len(vs)):
            acc += 1.0 / (r + 1) ** a.hub_exponent
            c.append(acc)
        cum[t] = c
    order = {t: rng.sample(vs, len(vs)) for t, vs in members.items()}

    edges = []
    for _ in range(a.edges):
        u = rng.randrange(a.nodes)
        t = step[u]
        if len(members[t]) < 2:
            continue
        v = rng.choices(order[t], cum_weights=cum[t])[0]
        if v != u:
            edges.append((u, v))
    n_noise = int(a.noise * len(edges))
    for _ in range(n_noise):
        kind = rng.randrange(3)
        if kind == 0 and edges:
            edges.append(rng.choice(edges))          # duplicate
        elif kind == 1:
            u = rng.randrange(a.nodes)
            edges.append((u, u))                     # self-loop
        else:
            edges.append((rng.randrange(a.nodes), rng.randrange(a.nodes)))  # maybe cross-step
    rng.shuffle(edges)

    ids = list(range(a.nodes))
    rng.shuffle(ids)  # mapping rows in random order, like a real export might be
    with open(os.path.join(a.out_dir, "transaction_mapping.csv"), "w") as f:
        f.write("transaction_id,node_id,time_step\n")
        for v in ids:
            f.write(f"{100000 + v},{v},{step[v]}\n")
    with open(os.path.join(a.out_dir, "node_edges.csv"), "w") as f:
        f.write("source_node,target_node\n")
        for u, v in edges:
            f.write(f"{u},{v}\n")


if __name__ == "__main__":
    main()
