#!/usr/bin/env bash
# Role 2 test suite. Run from anywhere:  bash tests/run_tests.sh
# Needs: a C compiler, make, python3 (standard library only).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/src/reference/graphfeat"
OUT="$ROOT/tests/out"
mkdir -p "$OUT"
pass=0; fail=0
ok()  { echo "  PASS  $1"; pass=$((pass + 1)); }
bad() { echo "  FAIL  $1"; fail=$((fail + 1)); }

echo "== build"
make -s -C "$ROOT/src/reference" graphfeat

echo "== golden fixtures (hand-checked, see fixtures/README.md)"
for d in "$ROOT"/fixtures/*/; do
    name="$(basename "$d")"
    "$BIN" --mapping "$d/mapping.csv" --edges "$d/edges.csv" --out "$OUT/$name.csv" 2>/dev/null
    if python3 "$ROOT/tests/compare.py" "$d/expected.csv" "$OUT/$name.csv" >/dev/null; then
        ok "$name matches expected.csv"
    else
        bad "$name differs from expected.csv"
    fi
done

echo "== cleaning counts on tiny (1 cross-step, 1 self-loop, 1 duplicate)"
"$BIN" --mapping "$ROOT/fixtures/tiny/mapping.csv" --edges "$ROOT/fixtures/tiny/edges.csv" \
       --out "$OUT/tiny_meta.csv" --meta "$OUT/tiny.meta.json" 2>/dev/null
if grep -q '"cross_step_dropped": 1,' "$OUT/tiny.meta.json" &&
   grep -q '"self_loops_dropped": 1,' "$OUT/tiny.meta.json" &&
   grep -q '"duplicates_dropped": 1,' "$OUT/tiny.meta.json" &&
   grep -q '"directed_edges": 10,' "$OUT/tiny.meta.json" &&
   grep -q '"undirected_edges": 9,' "$OUT/tiny.meta.json"; then
    ok "metadata counts"
else
    bad "metadata counts"
fi

echo "== random graphs vs independent Python oracle"
for seed in 1 2 3; do
    g="$OUT/rand$seed"
    python3 "$ROOT/tests/gen_graph.py" "$g" --nodes 600 --edges 900 --steps 7 --seed "$seed" --noise 0.05
    "$BIN" --mapping "$g/transaction_mapping.csv" --edges "$g/node_edges.csv" --out "$g/c.csv" 2>/dev/null
    python3 "$ROOT/tests/oracle.py" "$g/transaction_mapping.csv" "$g/node_edges.csv" "$g/py.csv"
    if python3 "$ROOT/tests/compare.py" "$g/c.csv" "$g/py.csv" >/dev/null; then
        ok "seed $seed oracle parity"
    else
        bad "seed $seed oracle parity"
        python3 "$ROOT/tests/compare.py" "$g/c.csv" "$g/py.csv" || true
    fi
done

echo "== determinism (repeated runs must be byte-identical)"
g="$OUT/det"
python3 "$ROOT/tests/gen_graph.py" "$g" --nodes 20000 --edges 26000 --steps 49 --seed 7
for r in 1 2 3; do
    "$BIN" --mapping "$g/transaction_mapping.csv" --edges "$g/node_edges.csv" --out "$g/run$r.csv" 2>/dev/null
done
if cmp -s "$g/run1.csv" "$g/run2.csv" && cmp -s "$g/run1.csv" "$g/run3.csv"; then
    ok "3 runs identical (sha256 $({ sha256sum "$g/run1.csv" 2>/dev/null || shasum -a 256 "$g/run1.csv"; } | cut -c1-16)...)"
else
    bad "repeated runs differ"
fi

echo "== invalid input is rejected"
e="$OUT/bad"; mkdir -p "$e"
printf 'node_id,time_step\n0,1\n1,1\n1,1\n' > "$e/dup_map.csv"
printf 'source_node,target_node\n' > "$e/empty_edges.csv"
printf 'node_id,time_step\n0,1\n1,1\n' > "$e/map.csv"
printf 'source_node,target_node\n0,5\n' > "$e/bad_edge.csv"
printf 'node_id,time_step\n0,1\n2,1\n' > "$e/gap_map.csv"
check_reject() {
    if "$BIN" --mapping "$1" --edges "$2" --out "$e/x.csv" 2>/dev/null; then bad "$3 accepted"; else ok "$3 rejected"; fi
}
check_reject "$e/dup_map.csv" "$e/empty_edges.csv" "duplicate node_id"
check_reject "$e/gap_map.csv" "$e/empty_edges.csv" "node_id gap"
check_reject "$e/map.csv"     "$e/bad_edge.csv"    "edge to unknown node"

echo
echo "passed: $pass  failed: $fail"
[ "$fail" -eq 0 ]
