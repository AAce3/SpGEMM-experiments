# Perlmutter: Tpetra and CombBLAS

Compare both libraries on the same matrix with all 128 physical cores/node:

| Processes/node | Threads/process | Slurm logical CPUs/process |
|---:|---:|---:|
| 32 | 4 | 8 |
| 128 | 1 | 2 |

Use 2, 8, or 32 nodes so both layouts have square total process counts for
CombBLAS. Both libraries use the same binaries for one- and four-thread runs.

## Build

Load a consistent compiler/MPI environment. Use an existing MPI/OpenMP Trilinos
installation with Tpetra, or build a minimal one. This example targets Trilinos
17.0.0; adjust compiler and installation paths for your allocation environment:

```sh
git clone --depth 1 --branch trilinos-release-17-0-0 \
  https://github.com/trilinos/Trilinos.git deps/Trilinos
export CRAYPE_LINK_TYPE=dynamic
cmake -S deps/Trilinos -B build-trilinos \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=20 \
  -DCMAKE_C_COMPILER=cc -DCMAKE_CXX_COMPILER=CC \
  -DCMAKE_INSTALL_PREFIX="$PWD/deps/trilinos-install" \
  -DBUILD_SHARED_LIBS=ON -DTPL_ENABLE_MPI=ON \
  -DTrilinos_ENABLE_Fortran=OFF -DTrilinos_ENABLE_Tpetra=ON \
  -DTrilinos_ENABLE_ALL_OPTIONAL_PACKAGES=OFF \
  -DTrilinos_ENABLE_TESTS=OFF -DTrilinos_ENABLE_EXAMPLES=OFF \
  -DTrilinos_ENABLE_OpenMP=ON -DKokkos_ENABLE_OPENMP=ON \
  -DTpetra_INST_OPENMP=ON \
  -DTpetra_INST_SERIAL=OFF -DTpetra_INST_DOUBLE=ON \
  -DTpetra_INST_INT_LONG_LONG=ON
cmake --build build-trilinos -j16
cmake --install build-trilinos

cmake -S . -B build-perlmutter -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_C_COMPILER=cc -DCMAKE_CXX_COMPILER=CC \
  -DCMAKE_PREFIX_PATH="$PWD/deps/trilinos-install" \
  -DBUILD_SHARED_LIBS=ON -DUSE_OPENMP=ON
cmake --build build-perlmutter --target tpetra_spgemm combblas_spgemm -j16
export BUILD_DIR="$PWD/build-perlmutter"
export RESULTS_DIR="$SCRATCH/spgemm-results"
```

Obtain CombBLAS as described in README.md. Trilinos includes its Kokkos
integration; no separate PETSc or binary matrix conversion is used. Select
CPU OpenMP, not a CUDA/HIP/SYCL-only build. BLAS/LAPACK must also be available
for Trilinos's required dependencies. Keep all executables and dependencies on
shared storage. For mpiP, use dynamic MPI linking and a compatible shared mpiP
library built for the same MPI ABI.

## Run one matrix first

The default `benchmark_matrices.tsv` points to the ten existing `.mtx` files in
`/global/cfs/cdirs/m4646/julian-aaron-shared/matrices`. No download or conversion
is needed. Its two columns are `name<TAB>Matrix Market path`; absolute paths
are used directly, and relative paths are resolved against the manifest.
From the repository:

```sh
# Preview two jobs: cage15, two nodes, both layouts. No Slurm contact.
python3 scripts/submit_scaling.py --account YOUR_ACCOUNT

# Submit: three baseline repetitions per library and layout.
python3 scripts/submit_scaling.py --account YOUR_ACCOUNT --submit

# Add one communication profile per library and layout.
export MPIP_LIB=/absolute/path/to/Cray-MPICH-compatible/libmpiP.so
python3 scripts/submit_scaling.py --account YOUR_ACCOUNT --profile --submit
```

Use `--matrices NAME ...`, `--nodes 2 8 32`, `--processes-per-node 32 128`,
`--time HH:MM:SS`, and `--manifest PATH` to select other cases. Nothing submits
without `--submit`. Each job runs both libraries sequentially with alternating
baseline order. BASELINE_REPEATS defaults to 3. Each invocation reloads its
matrix outside timing and measures a fresh symbolic-plus-numeric product.
Large inputs/products may exhaust memory; start with a pilot before expanding.

There is one batch script, `scripts/profile_perlmutter.slurm`. Direct sbatch
usage defaults to two nodes and 32 processes/node:

```sh
sbatch --account=YOUR_ACCOUNT scripts/profile_perlmutter.slurm /path/to/A.mtx
```

For pure MPI on two nodes, add `--ntasks=256 --ntasks-per-node=128
--cpus-per-task=2` before the script path. Set PROFILE=1 to enable mpiP;
timing-only runs do not require it. The script binds threads to cores and
checks both executables' reported thread counts.

## Results

Each job writes `RESULTS_DIR/profile-JOBID/environment.txt` plus per-library
`baseline-1/`, `baseline-2/`, `baseline-3/`, and optional `profile-1/` directories.
Runs contain stdout.log, stderr.log, rank_times.csv, and step_status.csv.
Profile directories also contain mpiP reports. Metadata records placement,
threads, backend, input, and dependencies.

Use baseline `spgemm_max_s` as runtime and report variability across repetitions.
mpiP is enabled only around multiplication; loading, initial barrier, operand
copying, timing reductions, metadata, and output are excluded. A missing mpiP
report is an error. mpiP is preloaded into benchmark processes only.

MPI time includes waiting and synchronization: label it “time inside MPI,” not
pure network overhead. Compare each rank's MPI time with that rank's elapsed
time from the same profiled run. Don't subtract aggregate maxima or mix baseline
and profile measurements. Check report units. MPI payload bytes are not physical
network traffic. Tpetra and CombBLAS also differ in local algorithms and storage.

## Validate on the cluster

Run the numerical checks in README.md with one and four threads on a small
allocation (the checker uses mpirun; adapt to the site's launcher policy).
Check actual thread/CPU activity on a representative matrix; a configured
thread count alone does not establish useful parallel work. Smoke-test mpiP
and Slurm binding before expanding the sweep. Local tests cannot verify the
cluster's compiler/MPI/mpiP compatibility or scheduler placement.

References: [Tpetra](https://trilinos.github.io/tpetra.html),
[Trilinos build guide](https://docs.trilinos.org/files/TrilinosBuildReference.pdf),
[CombBLAS](https://github.com/PASSIONLab/CombBLAS),
[NERSC affinity](https://docs.nersc.gov/jobs/affinity/),
[mpiP](https://software.llnl.gov/mpiP/).
