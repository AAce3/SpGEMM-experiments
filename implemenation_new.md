> Historical plan, superseded by README.md and PROFILING.md. The active
> benchmark now compares Tpetra against CombBLAS; PETSc was removed.

# Initial SpGEMM Benchmark Plan: 1D PETSc vs 2D CombBLAS

## Goal

Build a minimal benchmark that compares:

- **1D distributed SpGEMM using PETSc**
- **2D distributed SpGEMM using CombBLAS**

Both implementations should perform the same operation:

    load sparse matrix A
    ↓
    compute C = A × A
    ↓
    measure SpGEMM runtime
    ↓
    print basic metadata/results

For the initial implementation, the goal is simply to get both versions working correctly on the same input and under the same MPI configuration.

---

# 1. Experimental Setup

Use:

    PETSc MATMPIAIJ
        -> 1D row-distributed sparse matrix

    CombBLAS SpParMat
        -> 2D distributed sparse matrix

The initial comparison is:

    same sparse matrix A
    same MPI process count
    same node allocation
    same threads per rank
    same operation: C = A²

    PETSc 1D
    vs.
    CombBLAS 2D

Initially, set:

    export OMP_NUM_THREADS=1

so the comparison is MPI-only.

This avoids mixing the 1D-vs-2D question with MPI/OpenMP configuration differences.

---

# 2. Project Structure

Create:

    spgemm-bench/
    ├── petsc/
    │   ├── benchmark.cpp
    │   └── CMakeLists.txt
    ├── combblas/
    │   ├── benchmark.cpp
    │   └── CMakeLists.txt
    ├── matrices/
    ├── results/
    └── README.md

Keep PETSc and CombBLAS as separate executables initially.

Do not try to create a shared abstraction layer yet.

---

# 3. PETSc Implementation: 1D SpGEMM

## 3.1 Install or Load PETSc

First check whether PETSc is already installed on the cluster:

    module avail petsc

If available:

    module load petsc

Also load the appropriate compiler/MPI environment, for example:

    module load gcc
    module load openmpi

or the cluster-specific equivalents.

If PETSc is not installed, clone it:

    git clone https://gitlab.com/petsc/petsc.git
    cd petsc

Then configure it using the MPI compilers available on the system.

For the initial experiment, only CPU + MPI support is needed.

No GPU support is necessary.

---

## 3.2 PETSc Matrix Representation

Use:

    MATMPIAIJ

as the distributed sparse matrix format.

Conceptually, PETSc distributes matrix rows across ranks:

    rank 0 -> first block of rows
    rank 1 -> next block of rows
    rank 2 -> next block of rows
    ...

This is the 1D distributed layout for the benchmark.

---

## 3.3 PETSc Program Structure

The PETSc benchmark should perform:

    PetscInitialize
    ↓
    load matrix A
    ↓
    ensure A is distributed as MATMPIAIJ
    ↓
    check that A is square
    ↓
    barrier
    ↓
    time MatMatMult(A, A)
    ↓
    collect rank timings
    ↓
    print metadata/results
    ↓
    destroy matrices
    ↓
    PetscFinalize

Starter structure:

    #include <petscmat.h>
    #include <mpi.h>

    #include <iostream>
    #include <string>

    int main(int argc, char** argv)
    {
        PetscInitialize(&argc, &argv, nullptr, nullptr);

        MPI_Comm comm = PETSC_COMM_WORLD;

        int rank, size;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);

        if (argc < 2)
        {
            if (rank == 0)
                std::cerr << "Usage: " << argv[0] << " matrix_file\n";

            PetscFinalize();
            return 1;
        }

        std::string filename = argv[1];

        Mat A = nullptr;
        Mat C = nullptr;

        // TODO:
        // Load A.
        // Ensure it is a distributed MATMPIAIJ matrix.

        PetscInt rows, cols;
        MatGetSize(A, &rows, &cols);

        if (rows != cols)
        {
            if (rank == 0)
                std::cerr << "Matrix must be square.\n";

            MatDestroy(&A);
            PetscFinalize();
            return 1;
        }

        MPI_Barrier(comm);

        double start = MPI_Wtime();

        MatMatMult(
            A,
            A,
            MAT_INITIAL_MATRIX,
            PETSC_DETERMINE,
            &C
        );

        double local_time = MPI_Wtime() - start;

        double min_time = 0.0;
        double max_time = 0.0;
        double sum_time = 0.0;

        MPI_Reduce(
            &local_time,
            &min_time,
            1,
            MPI_DOUBLE,
            MPI_MIN,
            0,
            comm
        );

        MPI_Reduce(
            &local_time,
            &max_time,
            1,
            MPI_DOUBLE,
            MPI_MAX,
            0,
            comm
        );

        MPI_Reduce(
            &local_time,
            &sum_time,
            1,
            MPI_DOUBLE,
            MPI_SUM,
            0,
            comm
        );

        MatInfo infoA;
        MatInfo infoC;

        MatGetInfo(A, MAT_GLOBAL_SUM, &infoA);
        MatGetInfo(C, MAT_GLOBAL_SUM, &infoC);

        if (rank == 0)
        {
            double mean_time = sum_time / size;

            std::cout
                << "library=petsc"
                << ", matrix=" << filename
                << ", ranks=" << size
                << ", rows=" << rows
                << ", cols=" << cols
                << ", nnzA=" << infoA.nz_used
                << ", nnzC=" << infoC.nz_used
                << ", spgemm_min_s=" << min_time
                << ", spgemm_mean_s=" << mean_time
                << ", spgemm_max_s=" << max_time
                << '\n';
        }

        MatDestroy(&C);
        MatDestroy(&A);

        PetscFinalize();
        return 0;
    }

The main unresolved implementation detail is the matrix-loading path.

---

# 4. PETSc Matrix Loading

There are two reasonable approaches.

## Option A: Load Matrix Market Directly

Preferred if straightforward.

The desired interface is:

    mpirun -np 4 ./petsc_spgemm matrices/example.mtx

This makes PETSc and CombBLAS consume the same source matrix.

If direct Matrix Market loading turns out to be annoying, do not spend too much time fighting it.

## Option B: Convert Matrix Market to PETSc Binary

PETSc has strong support for loading its own binary matrix format.

The workflow becomes:

    SuiteSparse .mtx
    ↓
    one-time conversion
    ↓
    PETSc binary matrix
    ↓
    PETSc benchmark

This is acceptable because matrix input is outside the timed SpGEMM region.

The important requirement is that both benchmarks represent the same original matrix.

Do not include conversion time in the SpGEMM measurement.

---

# 5. CombBLAS Implementation: 2D SpGEMM

The CombBLAS benchmark no longer needs permutation logic.

It should perform:

    MPI_Init
    ↓
    load A using ParallelReadMM
    ↓
    check that A is square
    ↓
    barrier
    ↓
    time PSpGEMM(A, A)
    ↓
    collect rank timings
    ↓
    print metadata/results
    ↓
    MPI_Finalize

Starter structure:

    #include <CombBLAS/CombBLAS.h>
    #include <mpi.h>

    #include <cstdint>
    #include <iostream>
    #include <string>

    using namespace combblas;

    using IT = int64_t;
    using NT = double;

    using DER = SpDCCols<IT, NT>;
    using ParMat = SpParMat<IT, NT, DER>;

    using SR = PlusTimesSRing<NT, NT>;

    int main(int argc, char** argv)
    {
        int provided;

        MPI_Init_thread(
            &argc,
            &argv,
            MPI_THREAD_SERIALIZED,
            &provided
        );

        int rank, nprocs;

        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

        if (argc < 2)
        {
            if (rank == 0)
                std::cerr << "Usage: " << argv[0] << " matrix.mtx\n";

            MPI_Finalize();
            return 1;
        }

        std::string filename = argv[1];

        ParMat A;

        double read_start = MPI_Wtime();

        A.ParallelReadMM(
            filename,
            true,
            maximum<NT>()
        );

        double read_time = MPI_Wtime() - read_start;

        if (A.getnrow() != A.getncol())
        {
            if (rank == 0)
                std::cerr << "Matrix must be square.\n";

            MPI_Finalize();
            return 1;
        }

        MPI_Barrier(MPI_COMM_WORLD);

        double start = MPI_Wtime();

        ParMat C = PSpGEMM<SR>(A, A);

        double local_time = MPI_Wtime() - start;

        double min_time = 0.0;
        double max_time = 0.0;
        double sum_time = 0.0;

        MPI_Reduce(
            &local_time,
            &min_time,
            1,
            MPI_DOUBLE,
            MPI_MIN,
            0,
            MPI_COMM_WORLD
        );

        MPI_Reduce(
            &local_time,
            &max_time,
            1,
            MPI_DOUBLE,
            MPI_MAX,
            0,
            MPI_COMM_WORLD
        );

        MPI_Reduce(
            &local_time,
            &sum_time,
            1,
            MPI_DOUBLE,
            MPI_SUM,
            0,
            MPI_COMM_WORLD
        );

        if (rank == 0)
        {
            double mean_time = sum_time / nprocs;

            std::cout
                << "library=combblas"
                << ", matrix=" << filename
                << ", ranks=" << nprocs
                << ", rows=" << A.getnrow()
                << ", cols=" << A.getncol()
                << ", nnzA=" << A.getnnz()
                << ", nnzC=" << C.getnnz()
                << ", read_s=" << read_time
                << ", spgemm_min_s=" << min_time
                << ", spgemm_mean_s=" << mean_time
                << ", spgemm_max_s=" << max_time
                << '\n';
        }

        MPI_Finalize();
        return 0;
    }

The exact CombBLAS template/API details may require small adjustments depending on the version being used.

---

# 6. Timing Methodology

Use the same timing structure in both programs:

    MPI_Barrier(comm);

    double start = MPI_Wtime();

    C = A * A;

    double local_time = MPI_Wtime() - start;

Do not include:

    matrix file loading
    input conversion
    program startup
    result writing

inside the SpGEMM timing.

Do not put a trailing barrier inside the timed section.

Instead, collect each rank's elapsed time afterward.

Report:

    minimum rank time
    mean rank time
    maximum rank time

The main runtime metric should be:

    maximum rank time

because the distributed operation is effectively limited by the slowest rank.

---

# 7. Matching Output Format

Both executables should emit the same fields.

PETSc:

    library=petsc, matrix=..., ranks=..., rows=..., cols=..., nnzA=..., nnzC=..., spgemm_min_s=..., spgemm_mean_s=..., spgemm_max_s=...

CombBLAS:

    library=combblas, matrix=..., ranks=..., rows=..., cols=..., nnzA=..., nnzC=..., spgemm_min_s=..., spgemm_mean_s=..., spgemm_max_s=...

This will make later experiment automation easy.

---

# 8. Correctness Test

Before running SuiteSparse matrices, test both programs using a tiny matrix.

Create:

    matrices/tiny.mtx

with:

    %%MatrixMarket matrix coordinate real general
    3 3 4
    1 2 1
    2 1 1
    2 3 1
    3 2 1

This is:

    A =
    [ 0 1 0
      1 0 1
      0 1 0 ]

and:

    A² =
    [ 1 0 1
      0 2 0
      1 0 1 ]

Run:

    mpirun -np 4 ./petsc_spgemm matrices/tiny.mtx

and:

    mpirun -np 4 ./combblas_spgemm matrices/tiny.mtx

Verify:

    same rows
    same columns
    same nnz(A)
    same nnz(C)
    same numerical result

For debugging, temporarily write or gather the result matrix if needed.

Do not keep result gathering enabled for large matrices.

---

# 9. First Real Test

After the tiny matrix works, choose one moderately sized SuiteSparse matrix.

Run:

    mpirun -np 4 ./petsc_spgemm matrix_input

and:

    mpirun -np 4 ./combblas_spgemm matrix.mtx

Then try:

    4 MPI ranks
    16 MPI ranks
    64 MPI ranks

as resources allow.

For the initial CombBLAS implementation, square process counts are convenient because of its standard 2D layout.

---

# 10. Keep Threading Fixed Initially

Set:

    export OMP_NUM_THREADS=1

for CombBLAS.

The initial comparison should keep constant:

    same matrix
    same node allocation
    same MPI rank count
    one thread per rank

Do not yet vary:

    MPI ranks per node
    OpenMP threads
    process-grid geometry

Those can become later experiments.

---

# 11. Profiling

Do profiling only after both implementations are working correctly.

Run both under the same profiler.

Possible tools include:

    HPCToolkit
    Score-P / Scalasca
    mpiP
    TAU

The profiler should eventually let us inspect:

    communication time
    local computation time
    time inside MPI
    per-rank timing distribution

Do not manually instrument PETSc or CombBLAS until we know what the profiler already exposes.

---

# 12. Important Interpretation Caveat

This experiment technically compares:

    PETSc's 1D distributed SpGEMM
    vs.
    CombBLAS's 2D distributed SpGEMM

It is not a perfectly isolated experiment where only the process layout changes.

The libraries can also differ in:

    local sparse matrix representation
    local SpGEMM implementation
    memory management
    communication implementation
    internal optimizations

Therefore, results should initially be described as:

    PETSc 1D vs CombBLAS 2D

rather than claiming that every performance difference is caused solely by 1D vs 2D partitioning.

---

# 13. Implementation Order

## Step 1: Verify PETSc

Get PETSc loaded or installed.

Compile:

    #include <petscsys.h>

    int main(int argc, char** argv)
    {
        PetscInitialize(&argc, &argv, nullptr, nullptr);

        PetscPrintf(
            PETSC_COMM_WORLD,
            "PETSc works\n"
        );

        PetscFinalize();
        return 0;
    }

Run:

    mpirun -np 4 ./petsc_test

## Step 2: Load a PETSc Sparse Matrix

Get matrix loading working.

Print:

    rows
    columns
    nnz

and verify they match the source matrix.

## Step 3: Implement PETSc SpGEMM

Run:

    MatMatMult(
        A,
        A,
        MAT_INITIAL_MATRIX,
        PETSC_DETERMINE,
        &C
    );

Use the tiny matrix to verify correctness.

## Step 4: Add PETSc Timing

Add:

    local rank time
    minimum time
    mean time
    maximum time

At this point, the 1D benchmark is usable.

## Step 5: Implement/Clean Up CombBLAS Benchmark

The CombBLAS benchmark should contain only:

    load A
    ↓
    PSpGEMM(A, A)
    ↓
    time operation
    ↓
    print results

Remove all random-permutation logic from the previous version.

## Step 6: Run the Tiny Test Through Both

Verify that both produce the same A².

## Step 7: Run One SuiteSparse Matrix

Use the same matrix and MPI process count for both implementations.

At this point, the initial 1D-vs-2D benchmark is complete.

---

# 14. Initial Completion Checklist

    [ ] PETSc loads/builds successfully

    [ ] PETSc MPI hello-world works

    [ ] PETSc distributed sparse matrix loading works

    [ ] PETSc matrix dimensions and nnz are correct

    [ ] PETSc MatMatMult(A, A) works

    [ ] PETSc timing output works

    [ ] CombBLAS builds successfully

    [ ] CombBLAS Matrix Market loading works

    [ ] CombBLAS PSpGEMM(A, A) works

    [ ] CombBLAS timing output works

    [ ] both implementations pass the tiny correctness test

    [ ] both implementations run on one SuiteSparse matrix

    [ ] both use the same MPI process count

    [ ] both use the same node allocation

    [ ] OMP_NUM_THREADS=1 for the initial comparison

Once all of these are complete, the core benchmark exists. Profiling and larger matrix/configuration sweeps can then be added without redesigning the implementation.
