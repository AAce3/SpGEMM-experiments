#!/usr/bin/env bash
# One-command preparation + both-library/both-layout benchmark submission.
# Usage: bash scripts/run-permutation-experiment.sh INPUT.mtx [NEW_OUTPUT_DIR [SEED ...]]
# Defaults: seeds 42 43 44, three baselines and one MPI profile per configuration.
# Optional environment: ACCOUNT, CXX, BUILD_DIR, TIMING_RUNS, PROFILE_KIND.
set -euo pipefail
die() { echo "ERROR: $*" >&2; exit 1; }

# Internal batch-worker mode. No benchmark is launched until this job succeeds.
if [[ ${1:-} == --prepare ]]; then
    [[ $# -ge 4 ]] || die 'Invalid preparation arguments'
    shift
    input=$1; experiment=$2; shift 2
    "$experiment/permute_matrix" "$input" "$experiment/identity" identity
    for seed in "$@"; do
        "$experiment/permute_matrix" "$input" "$experiment/seed-$seed" "$seed"
    done
    exit 0
fi

[[ $# -ge 1 ]] || die 'Usage: bash scripts/run-permutation-experiment.sh INPUT.mtx [NEW_OUTPUT_DIR [SEED ...]]'
[[ -z ${SLURM_JOB_ID:-} ]] || die 'Run this launcher with bash on a login node, not sbatch.'
root=$(cd "$(dirname "$0")/.." && pwd)
input=$(realpath "$1")
[[ -f "$input" && -r "$input" ]] || die 'Input must be a readable Matrix Market file'
[[ "$input" == *.mtx ]] || die 'Input filename must end in .mtx'
name=$(basename "$input" .mtx)
experiment=${2:-${SCRATCH:?Set SCRATCH or pass NEW_OUTPUT_DIR}/permutation-$name-$(date +%Y%m%d-%H%M%S)-$$}
if [[ $# -ge 2 ]]; then shift 2; else shift; fi
[[ $# -gt 0 ]] || set -- 42 43 44
seeds=("$@")
seen=' '
for seed in "${seeds[@]}"; do
    [[ "$seed" =~ ^(0|[1-9][0-9]*)$ && ${#seed} -le 9 ]] || die 'Seeds must be canonical integers from 0 through 999999999'
    [[ "$seen" != *" $seed "* ]] || die "Duplicate seed: $seed"
    seen+="$seed "
done
export TIMING_RUNS=${TIMING_RUNS:-3} PROFILE_KIND=${PROFILE_KIND:-mpi}
[[ "$TIMING_RUNS" =~ ^[1-9][0-9]*$ ]] || die 'TIMING_RUNS must be positive'
[[ "$PROFILE_KIND" == mpi || "$PROFILE_KIND" == sample ]] || die 'PROFILE_KIND must be mpi or sample'
cd "$root"
export BUILD_DIR=$(realpath "${BUILD_DIR:-build-perlmutter-pat}")
for lib in combblas tpetra; do
    [[ -x "$BUILD_DIR/${lib}_spgemm" ]] || die "Missing $BUILD_DIR/${lib}_spgemm; run bash build.sh first"
done
command -v sbatch >/dev/null || die 'sbatch is required'
command -v "${CXX:-c++}" >/dev/null || die 'C++ compiler is required (set CXX if necessary)'
mkdir "$experiment" # Refuse an existing experiment; never overwrite results.
experiment=$(realpath "$experiment")
cp "$root/scripts/permute_matrix.cpp" "$experiment/permute_matrix.cpp"
cp "$root/scripts/run-permutation-experiment.sh" "$experiment/launcher.sh"
cp "$root/run.slurm" "$experiment/run.slurm"
"${CXX:-c++}" -O3 -std=c++11 "$experiment/permute_matrix.cpp" -o "$experiment/permute_matrix"
export RESULTS_DIR="$experiment/results"
common=(--parsable --export=ALL --chdir="$root")
[[ -z ${ACCOUNT:-} ]] || common+=(--account="$ACCOUNT")
printf 'kind\tvariant\tlayout\tjob_id\n' > "$experiment/jobs.tsv"
printf 'input=%s\nbuild=%s\ntiming_runs=%s\nprofile_kind=%s\nseeds=%s\n' \
    "$input" "$BUILD_DIR" "$TIMING_RUNS" "$PROFILE_KIND" "${seeds[*]}" > "$experiment/experiment.txt"
echo "Experiment: $experiment"
trap 'echo "Submission stopped. Previously accepted jobs remain submitted; inspect $experiment/jobs.tsv before retrying." >&2' ERR
prep=$(sbatch "${common[@]}" --job-name=matrix-permutation --constraint=cpu --qos=regular \
    --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=8G --time=12:00:00 \
    --output="$experiment/prep-%j.out" --error="$experiment/prep-%j.err" \
    "$experiment/launcher.sh" --prepare "$input" "$experiment" "${seeds[@]}")
prep=${prep%%;*}
[[ "$prep" =~ ^[0-9]+$ ]] || die "Unexpected sbatch response: $prep"
printf 'prepare\tall\t-\t%s\n' "$prep" >> "$experiment/jobs.tsv"
variants=(identity)
for seed in "${seeds[@]}"; do variants+=("seed-$seed"); done
for variant in "${variants[@]}"; do
    if [[ "$variant" == identity ]]; then tag=identity; else tag=perm-seed${variant#seed-}; fi
    matrix="$experiment/$variant/$name.$tag.mtx"
    for ppn in 32 128; do
        if [[ "$ppn" == 32 ]]; then tasks='0-1%1'; cpus=8; else tasks='2-3%1'; cpus=2; fi
        job=$(sbatch "${common[@]}" --dependency="afterok:$prep" --kill-on-invalid-dep=yes \
            --nodes=2 --array="$tasks" --ntasks-per-node="$ppn" --cpus-per-task="$cpus" \
            --output="$experiment/$variant-p$ppn-%A_%a.out" --error="$experiment/$variant-p$ppn-%A_%a.err" \
            "$experiment/run.slurm" "$matrix")
        job=${job%%;*}
        [[ "$job" =~ ^[0-9]+$ ]] || die "Unexpected sbatch response: $job"
        printf 'benchmark\t%s\tp%s\t%s\n' "$variant" "$ppn" "$job" >> "$experiment/jobs.tsv"
    done
done
trap - ERR
printf 'Submitted preparation job %s and %s benchmark arrays.\n' "$prep" "$((${#variants[@]} * 2))"
printf 'Benchmarks wait for successful preparation. Job IDs: %s/jobs.tsv\nResults: %s\n' "$experiment" "$RESULTS_DIR"
