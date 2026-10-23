#include <CombBLAS/CombBLAS.h>
#include <mpi.h>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#ifdef THREADED
#include <omp.h>
#endif

using IT = int64_t;
using NT = double;
using LocalMat = combblas::SpDCCols<IT, NT>;
using ParMat = combblas::SpParMat<IT, NT, LocalMat>;
using SR = combblas::PlusTimesSRing<NT, NT>;
int run(int argc, char** argv, int rank, int ranks)
{
    int threads = 1;
#ifdef THREADED
#pragma omp parallel
    {
#pragma omp single
        threads = omp_get_num_threads();
    }
#endif
    std::string output, rank_times;
    bool valid = argc >= 2;
    for (int i = 2; i < argc && valid; i += 2) {
        if (i + 1 >= argc) { valid = false; break; }
        const std::string option = argv[i];
        if (option == "--output" && output.empty()) output = argv[i + 1];
        else if (option == "--rank-times" && rank_times.empty()) rank_times = argv[i + 1];
        else valid = false;
    }
    if (!valid) {
        if (rank == 0) std::cerr << "Usage: " << argv[0]
            << " matrix.mtx [--output result.mtx] [--rank-times timings.csv]\n";
        return 1;
    }
    const int side = static_cast<int>(std::sqrt(ranks));
    if (side * side != ranks) {
        if (rank == 0) std::cerr << "MPI rank count must be a perfect square.\n";
        return 1;
    }
    int readable = 1;
    if (rank == 0) readable = static_cast<bool>(std::ifstream(argv[1]));
    MPI_Bcast(&readable, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!readable) {
        if (rank == 0) std::cerr << "Cannot open matrix: " << argv[1] << '\n';
        return 1;
    }
    ParMat A(MPI_COMM_WORLD);
    // Match Tpetra: duplicate coordinates are summed.
    A.ParallelReadMM(argv[1], true, std::plus<NT>());
    const IT rows = A.getnrow(), cols = A.getncol();
    if (rows != cols || rows <= 0) {
        if (rank == 0) std::cerr << "Matrix must be nonempty and square.\n";
        return 1;
    }
    // CombBLAS rejects aliased operands. Copy before starting the timer.
    ParMat B = A;
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Pcontrol(1); // mpiP: enable only after the initial barrier.
    const double start = MPI_Wtime();
    ParMat C = combblas::PSpGEMM<SR>(A, B);
    const double elapsed = MPI_Wtime() - start;
    MPI_Pcontrol(0); // Exclude reductions, metadata, CSV, and result output.
    double minimum = 0, maximum = 0, sum = 0;
    MPI_Reduce(&elapsed, &minimum, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&elapsed, &maximum, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&elapsed, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    // These metadata queries are collective; every rank must call them.
    const IT nnzA = A.getnnz(), nnzC = C.getnnz();
    if (rank == 0) {
        std::cout << std::setprecision(9)
                  << "library=combblas, matrix=" << std::quoted(argv[1])
                  << ", threads=" << threads
                  << ", ranks=" << ranks
                  << ", rows=" << rows << ", cols=" << cols
                  << ", nnzA=" << nnzA << ", nnzC=" << nnzC
                  << ", spgemm_min_s=" << minimum
                  << ", spgemm_mean_s=" << sum / ranks
                  << ", spgemm_max_s=" << maximum << '\n';
    }
    if (!rank_times.empty()) {
        std::vector<double> timings(rank == 0 ? ranks : 0);
        MPI_Gather(&elapsed, 1, MPI_DOUBLE, timings.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        int written = 1;
        if (rank == 0) {
            std::ofstream csv(rank_times);
            csv << "rank,elapsed_s\n" << std::setprecision(17);
            for (int r = 0; r < ranks; ++r) csv << r << ',' << timings[r] << '\n';
            csv.close();
            written = !csv.fail();
            if (!written) std::cerr << "Cannot write rank timings: " << rank_times << '\n';
        }
        MPI_Bcast(&written, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (!written) return 1;
    }
    // Debug output only, outside the timed region.
    if (!output.empty()) C.SaveGathered(output);
    return 0;
}

int main(int argc, char** argv)
{
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    if (provided < MPI_THREAD_SERIALIZED) {
        std::cerr << "MPI_THREAD_SERIALIZED is required.\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Pcontrol(0); // Also use MPIP=-o to exclude initialization itself.
    int rank = 0, ranks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int status = 1;
    try {
        // All CombBLAS objects (including communicators) die before finalization.
        status = run(argc, argv, rank, ranks);
    } catch (const std::exception& error) {
        std::cerr << "Rank " << rank << ": " << error.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return status;
}
