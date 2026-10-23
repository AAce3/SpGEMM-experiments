#!/bin/bash
#SBATCH --job-name=spgemm
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=2
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=spgemm-%j.out
#SBATCH --error=spgemm-%j.err

# Both libraries and both layouts run sequentially in one allocation.
# Usage: sbatch --account=ACCOUNT run.slurm /path/to/matrix.mtx
set -euo pipefail
die() { echo "ERROR: $*" >&2; exit 1; }
[[ $# == 1 ]] || die "Pass one Matrix Market file."
cd "${SLURM_SUBMIT_DIR:?Submit from the repository directory}"
mtx=$(realpath "$1")
build=$(realpath "${BUILD_DIR:-build-perlmutter}")
[[ -r "$mtx" ]] || die "Cannot read $mtx"
for lib in tpetra combblas; do
    [[ -x "$build/${lib}_spgemm" ]] || die "Run bash build.sh first."
done
nodes=${SLURM_JOB_NUM_NODES:?Missing node count}
case "$nodes" in 2|8|32) ;; *) die "Use 2, 8, or 32 nodes for square CombBLAS grids." ;; esac
repeats=${REPEATS:-3}
[[ "$repeats" =~ ^[1-9][0-9]*$ ]] || die "REPEATS must be positive."
profile=${PROFILE:-0}
[[ "$profile" == 0 || "$profile" == 1 ]] || die "PROFILE must be 0 or 1."
[[ -z ${LD_PRELOAD:-} ]] || die "Unset LD_PRELOAD for clean baseline runs."
if [[ "$profile" == 1 ]]; then
    MPIP_LIB=$(realpath "${MPIP_LIB:?Set MPIP_LIB to your MPI-compatible libmpiP.so}")
    [[ -r "$MPIP_LIB" && "$MPIP_LIB" != *[[:space:]:]* ]] || die "Invalid MPIP_LIB path."
fi
export OMP_DYNAMIC=FALSE OMP_PLACES=cores OMP_PROC_BIND=close
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
export SLURM_EXPORT_ENV=ALL
unset MPIP
results="${RESULTS_DIR:-$PWD/results}/${SLURM_JOB_ID:?Missing job ID}"
mkdir -p "$results"
results=$(realpath "$results")
printf 'matrix=%s\nnodes=%s\nbuild=%s\n' "$mtx" "$nodes" "$build" > "$results/environment.txt"

run_one() {
    local lib=$1 mode=$2 rep=$3 dir
    dir="$results/p$per_node/$lib/$mode-$rep"
    mkdir -p "$dir"
    local -a command=("$build/${lib}_spgemm" "$mtx" --rank-times "$dir/rank_times.csv")
    if [[ "$mode" == profile ]]; then
        command=(env "LD_PRELOAD=$MPIP_LIB" "MPIP=-o -l -k 3 -p -y" "${command[@]}")
    fi
    echo "Running $lib: $per_node processes/node, $OMP_NUM_THREADS threads, $mode $rep"
    srun --exact --nodes="$nodes" --ntasks="$((nodes * per_node))" \
        --ntasks-per-node="$per_node" --cpus-per-task="$((2 * OMP_NUM_THREADS))" \
        --cpu-bind=cores --distribution=block:block --kill-on-bad-exit=1 \
        --export=ALL --chdir="$dir" "${command[@]}" > "$dir/stdout.log" 2> "$dir/stderr.log"
    grep -Eq "^library=$lib,.*threads=$OMP_NUM_THREADS," "$dir/stdout.log" || die "Wrong thread count: $dir"
    [[ -s "$dir/rank_times.csv" ]] || die "Missing rank timings: $dir"
    if [[ "$mode" == profile ]]; then
        compgen -G "$dir/*.mpiP" >/dev/null || die "No mpiP report: $dir"
    fi
}
for per_node in 32 128; do
    export OMP_NUM_THREADS=$((128 / per_node))
    for ((rep=1; rep<=repeats; ++rep)); do
        if ((rep % 2)); then libs=(tpetra combblas); else libs=(combblas tpetra); fi
        for lib in "${libs[@]}"; do run_one "$lib" baseline "$rep"; done
    done
    if [[ "$profile" == 1 ]]; then
        for lib in tpetra combblas; do run_one "$lib" profile 1; done
    fi
done
echo "Results: $results"
