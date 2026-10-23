"""Small, dependency-free Matrix Market reader for correctness fixtures."""
import math


def read_mm(path):
    with open(path) as f:
        header = f.readline().lower().split()
        if (len(header) != 5 or header[:3] !=
                ["%%matrixmarket", "matrix", "coordinate"] or
                header[3] not in ("real", "integer", "pattern") or
                header[4] not in ("general", "symmetric")):
            raise ValueError("Expected coordinate real/integer/pattern, general/symmetric")
        lines = (s.split() for s in f if s.strip() and not s.lstrip().startswith("%"))
        rows, cols, count = map(int, next(lines))
        if rows <= 0 or cols != rows or count < 0:
            raise ValueError("Matrix must be nonempty and square")
        entries = {}
        actual = 0
        for fields in lines:
            if len(fields) != (2 if header[3] == "pattern" else 3):
                raise ValueError("Invalid coordinate record")
            i, j = int(fields[0]) - 1, int(fields[1]) - 1
            value = 1.0 if header[3] == "pattern" else float(fields[2])
            if not (0 <= i < rows and 0 <= j < cols and math.isfinite(value)):
                raise ValueError("Invalid index or nonfinite value")
            entries[i, j] = entries.get((i, j), 0.0) + value
            if header[4] == "symmetric" and i != j:
                entries[j, i] = entries.get((j, i), 0.0) + value
            actual += 1
        if actual != count:
            raise ValueError("Coordinate count does not match header")
        return rows, cols, entries
