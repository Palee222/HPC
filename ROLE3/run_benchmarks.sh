#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DATA_DIR="${DATA_DIR:-data}"
OUT_DIR="${OUT_DIR:-results/role3/bench_$(date +%Y%m%d_%H%M%S)}"
REPEATS="${REPEATS:-3}"
ALLOW_OVERSUBSCRIBE="${ALLOW_OVERSUBSCRIBE:-0}"
REFERENCE="${REFERENCE:-src/reference/graphfeat}"
OPENMP_EXE="${OPENMP_EXE:-ROLE3/openmp_graphfeat}"
MPI_EXE="${MPI_EXE:-ROLE3/mpi_graphfeat}"
PLOT_EXE="${PLOT_EXE:-ROLE3/speedupplots}"

for input in "$DATA_DIR/transaction_mapping.csv" "$DATA_DIR/node_edges.csv"; do
    if [[ ! -f "$input" ]]; then
        echo "Missing input: $input" >&2
        exit 1
    fi
done
for executable in "$REFERENCE" "$OPENMP_EXE" "$MPI_EXE" "$PLOT_EXE"; do
    if [[ ! -x "$executable" ]]; then
        echo "Missing executable: $executable (build with make -C ROLE3 and make -C src/reference)" >&2
        exit 1
    fi
done
if ! [[ "$REPEATS" =~ ^[1-9][0-9]*$ ]]; then
    echo "REPEATS must be a positive integer" >&2
    exit 2
fi

mkdir -p "$OUT_DIR/raw" "$OUT_DIR/features"
file_hash() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    else
        shasum -a 256 "$1" | awk '{print $1}'
    fi
}
{
    echo "date: $(date -Iseconds 2>/dev/null || date)"
    echo "host: $(hostname)"
    echo "os: $(uname -a)"
    echo "compiler: $(${CC:-cc} --version 2>/dev/null | head -1 || true)"
    echo "mpi_compiler: $(${MPICC:-mpicc} --version 2>/dev/null | head -1 || true)"
    echo "mpi_runtime: $(mpiexec --version 2>/dev/null | head -1 || true)"
    echo "cflags: ${CFLAGS:-compiler default}"
    echo "openmp_flags: ${OMPFLAGS:-compiler default}"
    echo "slurm_job_id: ${SLURM_JOB_ID:-none}"
    echo "repetitions: $REPEATS"
    echo "allow_oversubscribe: $ALLOW_OVERSUBSCRIBE"
    echo "mapping_sha256: $(file_hash "$DATA_DIR/transaction_mapping.csv")"
    echo "edges_sha256: $(file_hash "$DATA_DIR/node_edges.csv")"
    echo "omp_places: cores"
    echo "omp_proc_bind: close"
} > "$OUT_DIR/environment.txt"

MEASUREMENTS="$OUT_DIR/measurements.csv"
CONFIGS="$OUT_DIR/configurations.csv"
printf 'mode,ranks,threads,total_cpus,repeat,elapsed_seconds,graph_setup_seconds,degrees_seconds,neighbor_degree_seconds,two_hop_seconds,triangles_seconds,components_seconds,pagerank_seconds,mpi_reduce_seconds,write_csv_seconds\n' > "$MEASUREMENTS"
printf 'mode,ranks,threads,total_cpus\n' > "$CONFIGS"

"$REFERENCE" \
    --mapping "$DATA_DIR/transaction_mapping.csv" \
    --edges "$DATA_DIR/node_edges.csv" \
    --out "$OUT_DIR/reference.csv" \
    --meta "$OUT_DIR/reference.meta.json" \
    > "$OUT_DIR/reference.log" 2>&1

launch_case() {
    local mode="$1" ranks="$2" threads="$3" executable="$4"
    local output="$5" meta="$6" log="$7"
    {
        echo "mode=$mode ranks=$ranks threads=$threads"
        echo "executable=$executable"
        if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v srun >/dev/null 2>&1; then
            srun --exclusive --ntasks="$ranks" --cpus-per-task="$threads" \
                --cpu-bind=cores \
                env OMP_NUM_THREADS="$threads" OMP_PROC_BIND=close OMP_PLACES=cores \
                "$executable" \
                --mapping "$DATA_DIR/transaction_mapping.csv" \
                --edges "$DATA_DIR/node_edges.csv" \
                --out "$output" --meta "$meta"
        elif [[ "$mode" == "openmp" ]]; then
            env OMP_NUM_THREADS="$threads" OMP_PROC_BIND=close OMP_PLACES=cores \
                "$executable" \
                --mapping "$DATA_DIR/transaction_mapping.csv" \
                --edges "$DATA_DIR/node_edges.csv" \
                --out "$output" --meta "$meta"
        else
            if ! command -v mpiexec >/dev/null 2>&1; then
                echo "mpiexec is required for MPI runs outside Slurm" >&2
                return 1
            fi
            local -a mpi_options=()
            if [[ "$ALLOW_OVERSUBSCRIBE" == "1" ]]; then
                mpi_options+=(--oversubscribe)
            fi
            env OMP_NUM_THREADS="$threads" OMP_PROC_BIND=close OMP_PLACES=cores \
                mpiexec "${mpi_options[@]}" -n "$ranks" "$executable" \
                --mapping "$DATA_DIR/transaction_mapping.csv" \
                --edges "$DATA_DIR/node_edges.csv" \
                --out "$output" --meta "$meta"
        fi
    } > "$log" 2>&1
}

run_configuration() {
    local mode="$1" ranks="$2" threads="$3"
    local executable="$OPENMP_EXE"
    [[ "$mode" == "openmp" ]] || executable="$MPI_EXE"
    local cpus=$((ranks * threads))
    local case_dir="$OUT_DIR/raw/${mode}_r${ranks}_t${threads}"
    local output="$OUT_DIR/features/${mode}_r${ranks}_t${threads}.csv"
    mkdir -p "$case_dir"
    printf '%s,%s,%s,%s\n' "$mode" "$ranks" "$threads" "$cpus" >> "$CONFIGS"

    for ((repeat = 1; repeat <= REPEATS; repeat++)); do
        local meta="$case_dir/run${repeat}.meta.json"
        local log="$case_dir/run${repeat}.log"
        launch_case "$mode" "$ranks" "$threads" "$executable" \
            "$output" "$meta" "$log"
        python3 tests/compare.py "$OUT_DIR/reference.csv" "$output" \
            >> "$log" 2>&1
        python3 - "$meta" "$MEASUREMENTS" "$mode" "$ranks" "$threads" "$repeat" <<'PY'
import csv
import json
import sys

meta_path, csv_path, mode, ranks, threads, repeat = sys.argv[1:]
with open(meta_path, encoding="utf-8") as f:
    metadata = json.load(f)
timings = metadata["timings_sec"]
setup = timings.get("load_and_broadcast_graph", timings.get("load_and_build_csr", 0.0))
row = [
    mode, int(ranks), int(threads), int(ranks) * int(threads), int(repeat),
    timings["total"], setup, timings["degrees"], timings["neighbor_degree"],
    timings["two_hop"], timings["triangles"], timings["components"],
    timings["pagerank"], timings.get("mpi_reduce_results", 0.0),
    timings["write_csv"],
]
with open(csv_path, "a", newline="", encoding="utf-8") as f:
    csv.writer(f).writerow(row)
PY
        echo "PASS: $mode ranks=$ranks threads=$threads repeat=$repeat"
    done
}

for cpus in 1 2 4 8 16; do
    run_configuration openmp 1 "$cpus"
    run_configuration mpi "$cpus" 1
done

# Hybrid configurations use ranks x threads per rank; CPU count is their product.
run_configuration hybrid 1 1
for ranks in 1 2 4 8; do
    run_configuration hybrid "$ranks" 2
done

"$PLOT_EXE" "$MEASUREMENTS" "$OUT_DIR/scaling.csv" \
    "$OUT_DIR/speedup.svg" "$OUT_DIR/efficiency.svg"
echo "Benchmark results: $OUT_DIR"
