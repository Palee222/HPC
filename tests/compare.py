#!/usr/bin/env python3
"""Parity check between two feature CSVs (docs/feature_spec.md section 6).

    python3 tests/compare.py REFERENCE.csv CANDIDATE.csv [--pr-tol 1e-9]

Rules:
  * the header, row count and node_id order must be identical;
  * integer columns, nbr_deg_mean and clustering must be exactly equal;
  * pagerank may differ by at most --pr-tol (absolute).
Exit code 0 = parity passed, 1 = mismatch.
"""
import argparse
import csv
import sys

EXACT_FLOAT = {"nbr_deg_mean", "clustering"}
TOL_FLOAT = {"pagerank"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reference")
    ap.add_argument("candidate")
    ap.add_argument("--pr-tol", type=float, default=1e-9)
    ap.add_argument("--max-report", type=int, default=10)
    a = ap.parse_args()

    with open(a.reference, newline="") as f:
        ref = list(csv.reader(f))
    with open(a.candidate, newline="") as f:
        cand = list(csv.reader(f))

    if not ref or not cand:
        sys.exit("FAIL: empty file")
    if ref[0] != cand[0]:
        sys.exit(f"FAIL: header differs\n  ref:  {ref[0]}\n  cand: {cand[0]}")
    if len(ref) != len(cand):
        sys.exit(f"FAIL: row count differs: {len(ref) - 1} vs {len(cand) - 1}")

    header = ref[0]
    errors = []
    max_pr = 0.0
    for i in range(1, len(ref)):
        r, c = ref[i], cand[i]
        if len(r) != len(header) or len(c) != len(header):
            errors.append(f"row {i}: wrong number of fields")
            continue
        for col, rv, cv in zip(header, r, c):
            if col in TOL_FLOAT:
                diff = abs(float(rv) - float(cv))
                max_pr = max(max_pr, diff)
                ok = diff <= a.pr_tol
            elif col in EXACT_FLOAT:
                ok = float(rv) == float(cv)
            else:
                ok = int(rv) == int(cv)
            if not ok:
                errors.append(f"row {i} (node_id {r[0]}) {col}: ref={rv} cand={cv}")

    if errors:
        print(f"FAIL: {len(errors)} mismatches")
        for e in errors[: a.max_report]:
            print("  " + e)
        sys.exit(1)
    print(f"PASS: {len(ref) - 1} rows, max |pagerank diff| = {max_pr:.3e}")


if __name__ == "__main__":
    main()
