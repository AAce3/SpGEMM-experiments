#pragma once
#include <algorithm>
#include <cctype>
#include <istream>
#include <sstream>
#include <stdexcept>
#include <streambuf>
#include <string>

// Tpetra's matrix reader accepts numerical entries only. Translate pattern
// entries to value 1 as they are read, preserving duplicates for additive merge.
// Buffer one line, not a second copy of the entire input; no conversion file.
class PatternStreamBuffer : public std::streambuf {
    std::istream& input;
    std::string line;
    bool pattern = false, dimensions = true;

    void expose() {
        line += '\n';
        setg(line.data(), line.data(), line.data() + line.size());
    }
    int_type underflow() override {
        if (gptr() < egptr()) return traits_type::to_int_type(*gptr());
        if (!std::getline(input, line)) return traits_type::eof();
        const auto start = line.find_first_not_of(" \t\r");
        if (start != std::string::npos && line[start] != '%') {
            if (dimensions) dimensions = false;
            else if (pattern) line += " 1";
        }
        expose();
        return traits_type::to_int_type(*gptr());
    }
public:
    explicit PatternStreamBuffer(std::istream& source) : input(source) {
        if (!std::getline(input, line)) throw std::runtime_error("Cannot read Matrix Market header");
        std::transform(line.begin(), line.end(), line.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        std::istringstream header(line);
        std::string banner, object, format, field, symmetry, extra;
        header >> banner >> object >> format >> field >> symmetry;
        if (banner != "%%matrixmarket" || object != "matrix" || format != "coordinate" ||
            (field != "real" && field != "integer" && field != "pattern") ||
            (symmetry != "general" && symmetry != "symmetric") || (header >> extra))
            throw std::runtime_error("Expected coordinate real/integer/pattern, general/symmetric input");
        pattern = field == "pattern";
        line = "%%MatrixMarket matrix coordinate " +
               (pattern ? std::string("real") : field) + " " + symmetry;
        expose();
    }
};
