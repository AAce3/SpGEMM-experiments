#include <Tpetra_Core.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <TpetraExt_MatrixMatrix.hpp>
#include <MatrixMarket_Tpetra.hpp>
#include <Kokkos_Core.hpp>
#include <mpi.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include "pattern_stream.hpp"

#ifndef KOKKOS_ENABLE_OPENMP
#error "Build Trilinos/Tpetra with Kokkos OpenMP enabled."
#endif
#ifndef HAVE_TPETRA_MPI
#error "Build Trilinos/Tpetra with MPI enabled."
#endif
using Node = Tpetra::KokkosCompat::KokkosOpenMPWrapperNode;
using Matrix = Tpetra::CrsMatrix<double, int, long long, Node>;

int run(int argc, char** argv, int rank, int ranks)
{
    const int threads = Kokkos::OpenMP().concurrency();
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
    auto comm = Tpetra::getDefaultComm();
    std::ifstream input;
    std::unique_ptr<PatternStreamBuffer> buffer;
    if (rank == 0) {
        input.open(argv[1]);
        buffer = std::make_unique<PatternStreamBuffer>(input);
    }
    std::istream stream(buffer.get()); // Only rank zero reads the stream.
    auto A = Tpetra::MatrixMarket::Reader<Matrix>::readSparse(stream, comm);
    const auto rows = A->getGlobalNumRows(), cols = A->getGlobalNumCols();
    if (rows == 0 || rows != cols) throw std::runtime_error("Matrix must be nonempty and square");
    Kokkos::fence();
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Pcontrol(1);
    const double start = MPI_Wtime();
    // Include result allocation, symbolic setup, numeric work and fillComplete.
    Matrix C(A->getRowMap(), 0);
    Tpetra::MatrixMatrix::Multiply(*A, false, *A, false, C, true);
    Kokkos::fence(); // Include completion of all local work in the measurement.
    const double elapsed = MPI_Wtime() - start;
    MPI_Pcontrol(0);
    double minimum = 0, maximum = 0, sum = 0;
    MPI_Reduce(&elapsed, &minimum, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&elapsed, &maximum, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&elapsed, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    // These metadata queries are collective; every rank must call them.
    const auto nnzA = A->getGlobalNumEntries(), nnzC = C.getGlobalNumEntries();
    if (rank == 0) {
        std::cout << std::setprecision(9)
                  << "library=tpetra, backend=OpenMP, matrix=" << std::quoted(argv[1])
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
    if (!output.empty()) Tpetra::MatrixMarket::Writer<Matrix>::writeSparseFile(output, C);
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
    {
        Tpetra::ScopeGuard scope(&argc, &argv);
        try {
            status = run(argc, argv, rank, ranks);
        } catch (const std::exception& error) {
            std::cerr << "Rank " << rank << ": " << error.what() << '\n';
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    } // Matrices die before Kokkos, and Kokkos before MPI.
    MPI_Finalize();
    return status;
}
