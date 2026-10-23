#!/bin/bash
#SBATCH --job-name=spgemm-profile
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=2
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=spgemm-profile-%j.out
#SBATCH --error=spgemm-profile-%j.err

# Submit from the repository after loading your build's compiler/MPI/Trilinos
# environment. Supply your NERSC account with sbatch --account=YOUR_ACCOUNT.
# Usage: sbatch --account=... scripts/profile_perlmutter.slurm A.mtx
# Optional profiling: PROFILE=1 MPIP_LIB=/path/to/libmpiP.so
# Optional: BUILD_DIR, RESULTS_DIR, BASELINE_REPEATS (default 3).
set -euo pipefail
die() { echo "ERROR: $*" >&2; exit 1; }

[[ $# == 1 ]] || die "Pass one Matrix Market input: A.mtx."
: "${SLURM_JOB_ID:?Submit this script with sbatch}"
: "${SLURM_SUBMIT_DIR:?Missing submission directory}"
profile=${PROFILE:-0}
[[ "$profile" == 0 || "$profile" == 1 ]] || die "PROFILE must be 0 or 1."
nodes=${SLURM_JOB_NUM_NODES:?Missing node count}
ranks=${SLURM_NTASKS:?Missing MPI rank count}
per_node=${SLURM_NTASKS_PER_NODE:?Missing tasks per node}
case "$per_node" in 32) threads=4 ;; 128) threads=1 ;; *) die "Use 32 or 128 processes per node." ;; esac
cpus=$((threads * 2))
(( ranks == nodes * per_node )) || die "Use a uniform process count per node."
side=1
while (( (side + 1) * (side + 1) <= ranks )); do ((side+=1)); done
(( side * side == ranks )) || die "CombBLAS requires a square total; use 2, 8, or 32 nodes."
[[ ${SLURM_CPUS_PER_TASK:-0} == "$cpus" ]] || die "Expected $cpus logical CPUs per task."
[[ -z ${LD_PRELOAD:-} ]] || die "Unset LD_PRELOAD; mpiP is loaded only for profiling steps."
cd "$SLURM_SUBMIT_DIR"

mtx=$(realpath "$1")
build=$(realpath "${BUILD_DIR:-$SLURM_SUBMIT_DIR/build}")
mpip_library=
if [[ "$profile" == 1 ]]; then
    : "${MPIP_LIB:?Set MPIP_LIB for profiling}"
    mpip_library=$(realpath "$MPIP_LIB")
fi
[[ -r "$mtx" ]] || die "Input must exist on shared storage."
[[ "$profile" == 0 || -r "$mpip_library" ]] || die "Cannot read mpiP library: $mpip_library"
[[ "$mpip_library" != *[[:space:]:]* ]] || die "mpiP library path cannot contain whitespace or colons."
for library in tpetra combblas; do
    [[ -x "$build/${library}_spgemm" ]] || die "Missing executable: $build/${library}_spgemm"
done
repeats=${BASELINE_REPEATS:-3}
[[ "$repeats" =~ ^[1-9][0-9]*$ ]] || die "BASELINE_REPEATS must be a positive integer."

# One software thread per physical core; two logical CPUs per core.
export OMP_NUM_THREADS=$threads OMP_DYNAMIC=FALSE
export OMP_PLACES=cores OMP_PROC_BIND=close
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export SLURM_EXPORT_ENV=ALL
unset MPIP

results="${RESULTS_DIR:-$SLURM_SUBMIT_DIR/results}/profile-$SLURM_JOB_ID"
mkdir -p "$results"
results=$(realpath "$results")
{
    date -u
    echo "job=$SLURM_JOB_ID nodes=$nodes ranks=$ranks ranks_per_node=$per_node threads_per_rank=$threads tpetra_backend=OpenMP"
    echo "matrix_market=$mtx"
    echo "build=$build"
    echo "mpiP=$mpip_library"
    echo "baseline_repeats=$repeats"
    echo "MPIP=-o -l -k 3 -p -y"
    module list 2>&1 || true
    scontrol show hostnames "$SLURM_JOB_NODELIST"
    ldd "$build/tpetra_spgemm"
    ldd "$build/combblas_spgemm"
    if [[ "$profile" == 1 ]]; then ldd "$mpip_library"; fi
} > "$results/environment.txt"

run_one() {
    local mode=$1 library=$2 repetition=$3 run_dir
    run_dir="$results/$library/$mode-$repetition"
    mkdir -p "$run_dir"
    local -a command=("$build/${library}_spgemm" "$mtx" --rank-times "$run_dir/rank_times.csv")
    if [[ "$mode" == profile ]]; then
        # env runs on each rank, so libmpiP is injected into the benchmark,
        # not the batch shell or srun launcher. Reports use the step's cwd.
        command=(env "LD_PRELOAD=$mpip_library" "MPIP=-o -l -k 3 -p -y" "${command[@]}")
    fi
    echo "Starting $library $mode $repetition: $(date -u)"
    local started=$SECONDS status=0
    if srun --nodes="$nodes" --ntasks="$ranks" --ntasks-per-node="$per_node" --cpus-per-task="$cpus" \
        --cpu-bind=cores --distribution=block:block --kill-on-bad-exit=1 \
        --export=ALL --chdir="$run_dir" "${command[@]}" \
        > "$run_dir/stdout.log" 2> "$run_dir/stderr.log"; then
        status=0
    else
        status=$?
    fi
    printf 'wall_s,exit_code\n%s,%s\n' "$((SECONDS - started))" "$status" > "$run_dir/step_status.csv"
    (( status == 0 )) || die "$library $mode failed with status $status; see $run_dir."
    grep -Eq "^library=$library,.*threads=$threads," "$run_dir/stdout.log" || die "$library did not report the requested thread count."
    [[ -s "$run_dir/rank_times.csv" ]] || die "No rank timing CSV from $library."
    [[ $(wc -l < "$run_dir/rank_times.csv") -eq $((ranks + 1)) ]] || die "Incomplete rank timing CSV."
    if [[ "$mode" == profile ]]; then
        # Do not silently accept a successful run where preloading failed.
        compgen -G "$run_dir/*.mpiP" >/dev/null || die "No mpiP report; inspect $run_dir/stderr.log."
    fi
}

# Alternate ordering between baseline repetitions to reduce ordering bias.
for ((rep=1; rep<=repeats; ++rep)); do
    if ((rep % 2)); then
        run_one baseline tpetra "$rep"
        run_one baseline combblas "$rep"
    else
        run_one baseline combblas "$rep"
        run_one baseline tpetra "$rep"
    fi
done
if [[ "$profile" == 1 ]]; then
    run_one profile tpetra 1
    run_one profile combblas 1
fi
echo "Completed. Results: $results"
