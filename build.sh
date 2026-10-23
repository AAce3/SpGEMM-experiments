#!/usr/bin/env bash
# Run on Perlmutter after loading your compiler/MPI environment.
set -euo pipefail
cd "$(dirname "$0")"
root=$PWD
jobs=${JOBS:-8}
export CRAYPE_LINK_TYPE=dynamic
mkdir -p deps

if [[ ! -d deps/CombBLAS ]]; then
    git clone https://github.com/PASSIONLab/CombBLAS.git deps/CombBLAS
    git -C deps/CombBLAS checkout 2381a1f7a028b9f56d8531c0b7b157ae2ab27572
fi
# Apply the compatibility patch once; accept an already patched checkout.
if git -C deps/CombBLAS apply --check ../../patches/combblas-arm64.patch 2>/dev/null; then
    git -C deps/CombBLAS apply ../../patches/combblas-arm64.patch
elif ! git -C deps/CombBLAS apply --reverse --check ../../patches/combblas-arm64.patch 2>/dev/null; then
    echo "CombBLAS does not match the expected compatibility patch." >&2
    exit 1
fi
if [[ ! -d deps/Trilinos ]]; then
    git clone --depth 1 --branch trilinos-release-17-0-0 \
        https://github.com/trilinos/Trilinos.git deps/Trilinos
fi

cmake -S deps/Trilinos -B build-trilinos \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=20 \
    -DCMAKE_C_COMPILER="${CC:-cc}" -DCMAKE_CXX_COMPILER="${CXX:-CC}" \
    -DCMAKE_INSTALL_PREFIX="$root/deps/trilinos-install" \
    -DBUILD_SHARED_LIBS=ON -DTPL_ENABLE_MPI=ON \
    -DTrilinos_ENABLE_Fortran=OFF -DTrilinos_ENABLE_Tpetra=ON \
    -DTrilinos_ENABLE_ALL_OPTIONAL_PACKAGES=OFF \
    -DTrilinos_ENABLE_TESTS=OFF -DTrilinos_ENABLE_EXAMPLES=OFF \
    -DTrilinos_ENABLE_OpenMP=ON -DKokkos_ENABLE_OPENMP=ON \
    -DKokkos_ENABLE_ATOMICS_BYPASS=OFF \
    -DKokkosKernels_INST_EXECSPACE_OPENMP=ON \
    -DTpetra_INST_OPENMP=ON -DTpetra_INST_SERIAL=OFF \
    -DTpetra_INST_DOUBLE=ON -DTpetra_INST_INT_LONG_LONG=ON
cmake --build build-trilinos -j "$jobs"
cmake --install build-trilinos
cmake -S . -B build-perlmutter \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_C_COMPILER="${CC:-cc}" -DCMAKE_CXX_COMPILER="${CXX:-CC}" \
    -DCMAKE_PREFIX_PATH="$root/deps/trilinos-install" \
    -DBUILD_SHARED_LIBS=ON -DUSE_OPENMP=ON \
    -DBUILD_TPETRA=ON -DBUILD_COMBBLAS=ON
cmake --build build-perlmutter --target tpetra_spgemm combblas_spgemm -j "$jobs"
echo "Built build-perlmutter/{tpetra,combblas}_spgemm"
