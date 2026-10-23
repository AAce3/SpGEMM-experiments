# Tpetra vs CombBLAS SpGEMM

Compute A × A using Tpetra's 1D row distribution and CombBLAS's 2D grid.
Both use MPI and OpenMP and read the same Matrix Market file.

On Perlmutter, load your CPU compiler/MPI environment (`cc`, `CC`), with
CMake 3.27+, C++20 support, and BLAS/LAPACK available. Then:

```sh
bash build.sh
sbatch --account=YOUR_ACCOUNT run.slurm \
  /global/cfs/cdirs/m4646/julian-aaron-shared/matrices/cage15.mtx
```

The build script fetches and builds the pinned libraries, then both benchmarks.
Rerunning it reuses the dependency checkouts and incremental builds. Set JOBS
(default 8) to change build parallelism. Keep the same compiler/MPI environment.
No matrix downloads or conversion are needed; pass any existing `.mtx` path.

One job runs both libraries with **32 processes/node × 4 threads** and
**128 processes/node × 1 thread**, with three repetitions each. Default: two
CPU nodes. Override with `sbatch --nodes=8` or `--nodes=32`, and `--time` as needed.
CombBLAS requires square total process counts. Set REPEATS to change repetitions.

For communication profiling, also set:

```sh
export PROFILE=1
export MPIP_LIB=/actual/path/to/MPI-compatible/libmpiP.so
```

Then use the same sbatch command. This adds one mpiP run per library/layout.
Results go to `results/JOBID/p32/` and `p128/` (override RESULTS_DIR if desired).
Each run writes stdout.log, stderr.log and rank_times.csv; profiles add mpiP reports.
Use baseline `spgemm_max_s` for runtime. MPI time includes waiting/synchronization,
not just network transfer; compare per-rank values from the same profiled run.

Timing includes symbolic and numeric multiplication, excluding input loading and
output. Inputs must be square coordinate real/integer/pattern matrices with
general/symmetric storage. Duplicates are summed. Tpetra's reader loads through
rank zero, so input memory must fit there. Product fill-in can require much more
memory. Local numerical checks passed at 1 rank/1 thread and 4 ranks/4 threads;
Perlmutter placement and mpiP compatibility still require a cluster smoke test.
