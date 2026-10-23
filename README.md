# Tpetra 1D vs CombBLAS 2D SpGEMM

Two executables read the same Matrix Market file and compute **C = A × A**:

- `tpetra_spgemm`: Trilinos/Tpetra, distributed by rows, CPU OpenMP kernels.
- `combblas_spgemm`: CombBLAS, distributed on a square 2D process grid, OpenMP kernels.

The libraries distribute the matrix. MPI processes own pieces; threads work
within each process. No binary conversion is needed.

## Build

Requires CMake 3.27+ for the documented Trilinos build, C++20, MPI, and
Trilinos with Tpetra's OpenMP backend.
Use the same MPI and compatible compiler/OpenMP runtime for both libraries.
See [PROFILING.md](PROFILING.md) for the Trilinos build and Perlmutter setup.

Obtain the pinned CombBLAS dependency and apply its compatibility patch once:

```sh
mkdir -p deps
git clone https://github.com/PASSIONLab/CombBLAS.git deps/CombBLAS
git -C deps/CombBLAS checkout 2381a1f7a028b9f56d8531c0b7b157ae2ab27572
git -C deps/CombBLAS apply ../../patches/combblas-arm64.patch
```

Then point CMake at the Trilinos installation:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/path/to/trilinos-install -DUSE_OPENMP=ON
cmake --build build --target tpetra_spgemm combblas_spgemm -j8
```

Tpetra uses `double`, `int` local indices, `long long` global indices, and an
explicit OpenMP node type. Build those instantiations into Trilinos. A serial
or GPU-only Tpetra build is not sufficient. OpenMP must be discoverable for
CombBLAS when USE_OPENMP is ON. On macOS, an OpenMP-capable compiler such as
Homebrew LLVM may need to be selected explicitly.

For a machine without Trilinos, use `-DBUILD_TPETRA=OFF`; without CombBLAS,
use `-DBUILD_COMBBLAS=OFF`. `-DUSE_OPENMP=OFF` permits serial CombBLAS checks,
but cannot be used for the threaded batch comparison.

## Run

```sh
export OMP_NUM_THREADS=4 OMP_DYNAMIC=FALSE
mpirun -np 4 build/tpetra_spgemm matrices/tiny.mtx
mpirun -np 4 build/combblas_spgemm matrices/tiny.mtx
```

Both accept the same optional flags, intended for diagnostics and small tests:

```sh
mpirun -np 4 build/tpetra_spgemm matrices/tiny.mtx \
  --rank-times results/tpetra.csv --output results/tpetra.mtx
```

Use square, nonempty coordinate real/integer/pattern inputs with general or
symmetric storage. Symmetric entries are expanded and duplicates are summed.
Pattern entries have value 1; Tpetra's input adapter supplies these values
while streaming, including duplicate entries. Complex/skew-symmetric inputs
are unsupported. Tpetra's standard reader gathers input on rank zero before
redistribution, so large inputs must fit in that rank's memory. Loading is
outside timing; the reader is not a scalable parallel-I/O benchmark.

The default `benchmark_matrices.tsv` uses the existing files in:

```text
/global/cfs/cdirs/m4646/julian-aaron-shared/matrices
```

Available labels: `cage15`, `delaunay_n24`, `eukarya`, `HV15R`, `it-2004`,
`mouse_gene`, `nlpkkt200`, `Queen_4147`, `stokes`, and `subgraph4`.
Both executables read these files directly; no download or conversion is needed.
The default pilot is `cage15`. Select other files with `--matrices NAME ...`.

## Measurement

Each program reports ranks, threads, dimensions, nonzero counts, and
`spgemm_min_s`, `spgemm_mean_s`, `spgemm_max_s`. Tpetra also reports its backend.
**Use `spgemm_max_s` as runtime.** These are rank statistics from one fresh
multiplication, not statistics across repetitions.

Timing includes symbolic and numeric multiplication and product construction.
It excludes loading, metadata, output, and CombBLAS's required operand copy.
A barrier precedes timing; no trailing MPI barrier is included. Tpetra fences
Kokkos before timing and before stopping the timer to include completed work.
MPI profiling markers bracket the same multiplication region.

Both libraries must use matching ranks, threads, and node placement for each
comparison. CombBLAS requires square total rank counts. Performance reflects
local algorithms and storage as well as distribution, not partitioning alone.

## Validate

```sh
python3 tests/check_benchmarks.py --ranks 1 --threads 1
python3 tests/check_benchmarks.py --ranks 4 --threads 4
python3 tests/test_scaling_plan.py
python3 tests/test_profile_batch.py
```

The numerical checker compares entries against serial A², checks nonzero
counts and thread counts, and reconciles per-rank CSVs with reported times.
Fixtures cover weighted entries, empty rows, symmetric storage, duplicates,
and pattern inputs. Use `--combblas-only` without Trilinos. Batch tests mock
Slurm and submit no jobs.

For the two-node **32 processes × 4 threads versus 128 × 1** comparison and
optional mpiP communication profiling, follow [PROFILING.md](PROFILING.md).

Local validation: Trilinos 17.0.0 was built with MPI/OpenMP under
`deps/trilinos-install`; working benchmark binaries are in `build-tpetra/`.
Both libraries passed all four fixtures at 1 rank/1 thread and 4 ranks/4 threads,
including reported thread counts and timing CSV agreement. Eight planning and
mock-batch tests passed. Perlmutter placement and mpiP remain cluster checks.
