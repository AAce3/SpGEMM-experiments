#!/bin/bash
#SBATCH --job-name=spgemm
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=8
#SBATCH --array=0-1%1
#SBATCH --time=12:00:00
#SBATCH --output=spgemm-%A_%a.out
#SBATCH --error=spgemm-%A_%a.err

# One independent library/layout per task; limit concurrency to one by default.
# Usage: bash submit.sh 32|128 /path/to/matrix.mtx [sbatch options]
set -euo pipefail
die() { echo "ERROR: $*" >&2; exit 1; }
[[ $# == 1 ]] || die "Pass one Matrix Market file."
cd "${SLURM_SUBMIT_DIR:?Submit from the repository directory}"
mtx=$(realpath "$1")
build=$(realpath "${BUILD_DIR:-build-perlmutter-pat}")
[[ -r "$mtx" ]] || die "Cannot read $mtx"
case "${SLURM_ARRAY_TASK_ID:?Submit as a job array}" in
    0) lib=combblas; per_node=32 ;;
    1) lib=tpetra; per_node=32 ;;
    2) lib=combblas; per_node=128 ;;
    3) lib=tpetra; per_node=128 ;;
    *) die "Array index must be 0, 1, 2, or 3." ;;
esac
[[ -x "$build/${lib}_spgemm" ]] || die "Run bash build.sh first."
nodes=${SLURM_JOB_NUM_NODES:?Missing node count}
case "$nodes" in 2|8|32) ;; *) die "Use 2, 8, or 32 nodes for square CombBLAS grids." ;; esac
profile=${PROFILE:-1}
[[ "$profile" == 0 || "$profile" == 1 ]] || die "PROFILE must be 0 or 1."
# Make the batch environment self-contained rather than relying on the login shell.
type module >/dev/null 2>&1 || die "No module command in the batch environment."
module unload darshan
case ":${LOADEDMODULES:-}:" in
    *:perftools/*|*:perftools:*|*:perftools-lite*)
        die "Unload perftools/perftools-lite; this launcher uses pat_run." ;;
esac
[[ -z ${LD_PRELOAD:-} ]] || die "Unset external LD_PRELOAD before submitting."
cpus=$((256 / per_node))
[[ ${SLURM_CPUS_PER_TASK:-} == "$cpus" ]] || die "Allocation must request --cpus-per-task=$cpus; use submit.sh."
[[ ${SLURM_NTASKS_PER_NODE:-} == "$per_node" ]] || die "Allocation must request --ntasks-per-node=$per_node; use submit.sh."
[[ ${SLURM_NTASKS:-} == "$((nodes * per_node))" ]] || die "Allocation task count does not match the layout."
# Reject old binaries linked against Darshan, or unresolved shared libraries.
linkage=$(ldd "$build/${lib}_spgemm") || die "Cannot inspect dynamic dependencies."
if grep -Eiq 'darshan|not found' <<< "$linkage"; then
    die "Darshan or unresolved library in executable dependencies; rebuild with build.sh."
fi
if [[ "$profile" == 1 ]]; then
    module load perftools-base
    command -v pat_run >/dev/null || die "Load perftools-base (pat_run must be on PATH) before submitting."
    # Ignore stale profiler experiment settings inherited from earlier runs.
    while IFS= read -r variable; do unset "$variable"; done < <(compgen -v PAT_RT_)
    export PAT_RT_CALLSTACK_MODE=frames
    export PAT_RT_SUMMARY=1
    # Disable added synchronization barriers:
    # MPI time still includes application waiting and is not network-only time.
    export PAT_RT_MPI_SYNC=0
    export PAT_RT_EXPDIR_CLEANUP=0
    profile_kind=${PROFILE_KIND:-mpi}
    case "$profile_kind" in mpi|sample) ;; *) die "PROFILE_KIND must be mpi or sample." ;; esac
    mode=profile-$profile_kind
else
    mode=baseline
fi
export OMP_NUM_THREADS=$((128 / per_node))
export OMP_DYNAMIC=FALSE OMP_PLACES=cores OMP_PROC_BIND=close
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
export SLURM_EXPORT_ENV=ALL
unset MPIP
results="${RESULTS_DIR:-${SCRATCH:?Set SCRATCH or RESULTS_DIR}/spgemm-results}/${SLURM_ARRAY_JOB_ID:?Missing array job ID}"
dir="$results/p$per_node/$lib/$mode-1"
mkdir -p "$(dirname "$dir")"
# Preserve even failed-run logs; a requeue must not silently overwrite evidence.
mkdir "$dir" || die "Results directory already exists: $dir"
dir=$(realpath "$dir")
trap 'status=$?; printf "exit_code=%s\nfinished_utc=%s\n" "$status" "$(date -u +%FT%TZ)" > "$dir/status.txt"' EXIT
printf 'matrix=%s\nnodes=%s\nbuild=%s\nlibrary=%s\nprocesses_per_node=%s\nthreads=%s\nprofile=%s\nprofile_kind=%s\n' \
    "$mtx" "$nodes" "$build" "$lib" "$per_node" "$OMP_NUM_THREADS" "$profile" "${profile_kind:-none}" > "$dir/environment.txt"
scontrol show job "$SLURM_JOB_ID" > "$dir/allocation.txt"
module -t list > "$dir/modules.txt" 2>&1
printf '%s\n' "$linkage" > "$dir/linkage.txt"
sha256sum "$build/${lib}_spgemm" > "$dir/binary.sha256"
git rev-parse HEAD > "$dir/source-revision.txt"
git diff --binary > "$dir/source-changes.patch"
for metadata in build-modules.txt source-revision.txt source-changes.patch binaries.sha256; do
    if [[ -f "$build/$metadata" ]]; then cp "$build/$metadata" "$dir/build-$metadata"; fi
done
if [[ "$profile" == 1 ]]; then pat_run -V > "$dir/profiler-version.txt" 2>&1; fi
command=("$build/${lib}_spgemm" "$mtx" --rank-times "$dir/rank_times.csv")
if [[ "$profile" == 1 ]]; then
    # Profile the whole execution; inspect the multiplication call tree separately.
    mkdir -p "$dir/measurements"
    export PAT_RT_EXPDIR_BASE="$dir/measurements"
    if [[ "$profile_kind" == mpi ]]; then
        command=(pat_run -w -g mpi -r "${command[@]}")
    else
        command=(pat_run -s cs,time -r "${command[@]}")
    fi
fi
printf '%q ' "${command[@]}" > "$dir/command.txt"
printf '\n' >> "$dir/command.txt"
echo "Running $lib: $nodes nodes, $per_node processes/node, $OMP_NUM_THREADS threads, $mode"
echo "Results: $dir"
srun --exact --nodes="$nodes" --ntasks="$((nodes * per_node))" \
    --ntasks-per-node="$per_node" --cpus-per-task="$cpus" \
    --mem=0 --cpu-bind=verbose,cores --distribution=block:block --kill-on-bad-exit=1 \
    --export=ALL --chdir="$dir" "${command[@]}" > "$dir/stdout.log" 2> "$dir/stderr.log"
grep -Eq "^library=$lib,.*threads=$OMP_NUM_THREADS, ranks=$((nodes * per_node))," "$dir/stdout.log" || die "Wrong library/thread/rank count: $dir"
[[ -s "$dir/rank_times.csv" ]] || die "Missing rank timings: $dir"
[[ $(wc -l < "$dir/rank_times.csv") -eq $((nodes * per_node + 1)) ]] || die "Incomplete rank timings: $dir"
if [[ "$profile" == 1 ]]; then
    [[ -n $(find "$dir/measurements" -type f \( -name '*.xf' -o -name '*.ap2' \) -size +0c -print -quit) ]] || die "No CrayPat data: $dir"
    echo "CrayPat report: see stdout.log and measurements/"
    echo "For further analysis: pat_report EXPERIMENT_DIRECTORY (inside measurements/)"
fi
