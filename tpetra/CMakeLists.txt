find_package(Tpetra REQUIRED CONFIG)
add_executable(tpetra_spgemm benchmark.cpp)
target_compile_features(tpetra_spgemm PRIVATE cxx_std_20)
target_link_libraries(tpetra_spgemm PRIVATE Tpetra::all_libs MPI::MPI_CXX)
