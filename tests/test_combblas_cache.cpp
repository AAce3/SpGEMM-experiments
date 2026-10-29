// mpicxx -std=c++17 -Ideps/CombBLAS/include -Ideps/CombBLAS/psort-1.0/include tests/test_combblas_cache.cpp -o /tmp/test_combblas_cache
#include <CombBLAS/MPIOp.h>
#include <cstdlib>

static void add(void* input, void* output, int* length, MPI_Datatype*) {
    for (int i = 0; i < *length; ++i)
        static_cast<int*>(output)[i] += static_cast<int*>(input)[i];
}
static void check(bool condition) {
    if (!condition) MPI_Abort(MPI_COMM_WORLD, 1);
}
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    combblas::MPIDataTypeCache types;
    combblas::MPIOpCache ops;
    for (int iteration = 0; iteration < 2; ++iteration) {
        MPI_Datatype type;
        MPI_Type_contiguous(2, MPI_INT, &type);
        MPI_Type_commit(&type);
        types.set(&typeid(int), type);
        check(types.get(&typeid(int)) == type);
        MPI_Op op;
        MPI_Op_create(add, 1, &op);
        ops.set(&typeid(int), op);
        check(ops.get(&typeid(int)) == op);
        types.clear();
        ops.clear();
        check(types.get(&typeid(int)) == MPI_DATATYPE_NULL);
        check(ops.get(&typeid(int)) == MPI_OP_NULL);
        types.clear();
        ops.clear();
    }
    MPI_Finalize();
    // Empty caches must also be safe during post-finalize destruction.
    return 0;
}
