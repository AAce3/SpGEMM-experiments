// Streaming Matrix Market relabeling: B[p(i),p(j)] = A[i,j].
// Usage: permute_matrix INPUT.mtx NEW_OUTPUT_DIRECTORY SEED|identity
#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <sys/stat.h>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
using U = std::uint64_t;
U number(const std::string& s) {
    U value = 0;
    if (s.empty()) throw std::runtime_error("Empty unsigned integer");
    for (char c : s) {
        if (c < '0' || c > '9' ||
            value > (std::numeric_limits<U>::max() - static_cast<U>(c - '0')) / 10)
            throw std::runtime_error("Invalid or overflowing unsigned integer: " + s);
        value = value * 10 + static_cast<U>(c - '0');
    }
    return value;
}
// Rejection sampling avoids modulo bias and std::shuffle library differences.
U bounded(std::mt19937_64& rng, U bound) {
    const U threshold = -bound % bound;
    U x;
    do { x = rng(); } while (x < threshold);
    return x % bound;
}
bool data_line(std::istream& in, std::string& line) {
    while (std::getline(in, line)) {
        auto pos = line.find_first_not_of(" \t\r");
        if (pos != std::string::npos && line[pos] != '%') return true;
    }
    return false;
}
int main(int argc, char** argv) {
    std::string out;
    bool created = false;
    try {
        if (argc != 4) throw std::runtime_error("Usage: permute_matrix INPUT.mtx NEW_OUTPUT_DIRECTORY SEED|identity");
        const bool identity = std::string(argv[3]) == "identity";
        const U seed = identity ? 0 : number(argv[3]);
        std::ifstream in(argv[1]);
        if (!in) throw std::runtime_error("Cannot open input");
        std::string line, magic, object, format, field, symmetry, extra;
        std::getline(in, line);
        std::istringstream header(line);
        if (!(header >> magic >> object >> format >> field >> symmetry) || header >> extra ||
            magic != "%%MatrixMarket" || object != "matrix" || format != "coordinate" ||
            (field != "real" && field != "integer" && field != "pattern") ||
            (symmetry != "general" && symmetry != "symmetric"))
            throw std::runtime_error("Require coordinate real/integer/pattern with general/symmetric storage");
        if (!data_line(in, line)) throw std::runtime_error("Missing dimensions");
        std::istringstream dims(line);
        std::string ns, ms, zs;
        if (!(dims >> ns >> ms >> zs) || dims >> extra) throw std::runtime_error("Invalid dimensions");
        U n = number(ns), m = number(ms), nz = number(zs);
        if (!n || n != m) throw std::runtime_error("Require nonempty square matrix");
        std::vector<U> permutation(n);
        std::iota(permutation.begin(), permutation.end(), U{1});
        std::mt19937_64 rng(seed);
        if (!identity) for (U i=n; i>1; --i) std::swap(permutation[i-1], permutation[bounded(rng,i)]);
        out = argv[2];
        if (::mkdir(out.c_str(), 0777) != 0)
            throw std::runtime_error("Cannot create new output directory (existing directories are refused): " +
                                     std::string(std::strerror(errno)));
        created = true;
        const std::string tag = identity ? "identity" : "perm-seed" + std::to_string(seed);
        std::string stem = argv[1];
        const auto slash = stem.find_last_of('/');
        if (slash != std::string::npos) stem.erase(0, slash + 1);
        const auto dot = stem.find_last_of('.');
        if (dot != std::string::npos && dot != 0) stem.erase(dot);
        const auto filename = stem + "." + tag + ".mtx";
        std::ofstream matrix(out + "/" + filename), map(out + "/permutation.tsv"), meta(out + "/manifest.txt");
        matrix.exceptions(std::ios::failbit | std::ios::badbit);
        map.exceptions(std::ios::failbit | std::ios::badbit);
        meta.exceptions(std::ios::failbit | std::ios::badbit);
        matrix << "%%MatrixMarket matrix coordinate " << field << ' ' << symmetry << '\n'
               << "% B[p(i),p(j)] = A[i,j]; " << tag << "; mt19937_64 Fisher-Yates v1\n"
               << n << ' ' << n << ' ' << nz << '\n';
        map << "old_index\tnew_index\n";
        for (U i=0; i<n; ++i) map << i+1 << '\t' << permutation[i] << '\n';
        map.close();
        U count = 0;
        while (data_line(in, line)) {
            if (++count > nz) throw std::runtime_error("More entries than declared");
            std::istringstream entry(line);
            std::string is, js, value;
            if (!(entry >> is >> js)) throw std::runtime_error("Invalid coordinate");
            U i=number(is), j=number(js);
            if (i<1 || i>n || j<1 || j>n) throw std::runtime_error("Coordinate out of bounds");
            if (field != "pattern" && !(entry >> value)) throw std::runtime_error("Missing numeric value");
            if (entry >> extra) throw std::runtime_error("Unexpected entry fields");
            // Preserve numeric tokens exactly; no floating point round-trip.
            i=permutation[i-1]; j=permutation[j-1];
            // Retain conventional lower-triangle storage for symmetric inputs.
            if (symmetry == "symmetric" && i<j) std::swap(i,j);
            matrix << i << ' ' << j;
            if (field != "pattern") matrix << ' ' << value;
            matrix << '\n';
        }
        if (in.bad() || count != nz) throw std::runtime_error("Read failure or truncated entry count");
        matrix.close();
        char* absolute_input = ::realpath(argv[1], nullptr);
        if (!absolute_input) throw std::runtime_error("Cannot resolve input path");
        const std::string input_path(absolute_input);
        std::free(absolute_input);
        meta << "input=" << input_path << '\n'
             << "output=" << filename << "\nmode=" << tag << "\nrows=" << n << "\nstored_entries=" << nz
             << "\nalgorithm=mt19937_64; rejection-sampled Fisher-Yates v1\n"
             << "mapping=1-based old_index to new_index; same mapping on rows and columns\n"
             << "entry_order=preserved; symmetric coordinates normalized to lower triangle\n";
        meta.close();
        std::ofstream complete(out + "/COMPLETE");
        complete.exceptions(std::ios::failbit | std::ios::badbit);
        complete << "ok\n"; complete.close();
        std::cout << (out + "/" + filename) << '\n';
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        if (created) std::cerr << "Incomplete output retained in " << out << "; do not benchmark without COMPLETE.\n";
        return 1;
    }
}
