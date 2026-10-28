#!/usr/bin/env bash
# Run on Perlmutter after loading your compiler/MPI environment.
set -euo pipefail
cd "$(dirname "$0")"
root=$PWD
jobs=${JOBS:-8}
# Separate caches force the first profiling-ready build to relink everything.
# Darshan can be injected by compiler wrappers, so unloading only at run time
# is not sufficient for binaries built in the previous environment.
type module >/dev/null 2>&1 || { echo "Run in a Perlmutter module-enabled shell." >&2; exit 1; }
module unload darshan
case ":${LOADEDMODULES:-}:" in
    *:perftools/*|*:perftools:*|*:perftools-lite*)
        echo "Unload perftools/perftools-lite before using the pat_run build." >&2; exit 1 ;;
esac
module load perftools-base perftools-preload
[[ -z ${LD_PRELOAD:-} ]] || { echo "Unset external LD_PRELOAD before building." >&2; exit 1; }
mkdir -p build-perlmutter-pat
module -t list > build-perlmutter-pat/build-modules.txt 2>&1
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

cmake -S deps/Trilinos -B build-trilinos-pat \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_STANDARD=20 \
    -DCMAKE_C_COMPILER="${CC:-cc}" -DCMAKE_CXX_COMPILER="${CXX:-CC}" \
    -DCMAKE_INSTALL_PREFIX="$root/deps/trilinos-install-pat" \
    -DBUILD_SHARED_LIBS=ON -DTPL_ENABLE_MPI=ON \
    -DTrilinos_ENABLE_Fortran=OFF -DTrilinos_ENABLE_Tpetra=ON \
    -DTrilinos_ENABLE_ALL_OPTIONAL_PACKAGES=OFF \
    -DTrilinos_ENABLE_TESTS=OFF -DTrilinos_ENABLE_EXAMPLES=OFF \
    -DTrilinos_ENABLE_OpenMP=ON -DKokkos_ENABLE_OPENMP=ON \
    -DKokkos_ENABLE_ATOMICS_BYPASS=OFF \
    -DKokkosKernels_INST_EXECSPACE_OPENMP=ON \
    -DTpetra_INST_OPENMP=ON -DTpetra_INST_SERIAL=OFF \
    -DTpetra_INST_DOUBLE=ON -DTpetra_INST_INT_LONG_LONG=ON
cmake --build build-trilinos-pat -j "$jobs"
cmake --install build-trilinos-pat
cmake -S . -B build-perlmutter-pat \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_C_COMPILER="${CC:-cc}" -DCMAKE_CXX_COMPILER="${CXX:-CC}" \
    -DCMAKE_PREFIX_PATH="$root/deps/trilinos-install-pat" \
    -DBUILD_SHARED_LIBS=ON -DUSE_OPENMP=ON \
    -DBUILD_TPETRA=ON -DBUILD_COMBBLAS=ON
cmake --build build-perlmutter-pat --target tpetra_spgemm combblas_spgemm -j "$jobs"
git rev-parse HEAD > build-perlmutter-pat/source-revision.txt
git diff --binary > build-perlmutter-pat/source-changes.patch
sha256sum build-perlmutter-pat/tpetra_spgemm build-perlmutter-pat/combblas_spgemm > build-perlmutter-pat/binaries.sha256
echo "Built build-perlmutter-pat/{tpetra,combblas}_spgemm"
