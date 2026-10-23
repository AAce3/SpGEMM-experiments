#!/usr/bin/env python3
"""Numerically check both executables against a serial A*A on small fixtures."""
import argparse
import csv
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from matrix_io import read_mm

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--build", type=Path, default=Path("build"))
p.add_argument("--ranks", type=int, default=4)
p.add_argument("--combblas-only", action="store_true")
p.add_argument("--threads", type=int, default=1)
args = p.parse_args()
root = Path(__file__).resolve().parents[1]
env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), OMP_DYNAMIC="FALSE")
with tempfile.TemporaryDirectory() as temp:
    temp = Path(temp)
    symmetric = temp / "symmetric.mtx"
    symmetric.write_text("%%MatrixMarket matrix coordinate real symmetric\n"
                         "3 3 4\n1 1 2\n1 2 1\n1 2 3\n2 3 -1\n")
    pattern = temp / "pattern.mtx"
    pattern.write_text("%%MatrixMarket matrix coordinate pattern symmetric\n"
                       "% duplicates must sum\n3 3 4\n1 1\n1 2\n1 2\n2 3\n")
    for fixture in (root / "matrices/tiny.mtx", root / "tests/weighted.mtx", symmetric, pattern):
        rows, cols, a = read_mm(fixture)
        expected = {}
        for (i, k), x in a.items():
            for (l, j), y in a.items():
                if k == l:
                    expected[i, j] = expected.get((i, j), 0) + x * y
        libraries = ["combblas"] if args.combblas_only else ["combblas", "tpetra"]
        for library in libraries:
            output = temp / "result.mtx"
            timings = temp / (library + "_times.csv")
            command = ["mpirun", "-np", str(args.ranks),
                       str(args.build.resolve() / (library + "_spgemm")),
                       str(fixture),
                       "--output", str(output),
                       "--rank-times", str(timings)]
            result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=60)
            if result.returncode:
                raise RuntimeError(result.stdout + result.stderr)
            line = next(s for s in result.stdout.splitlines() if s.startswith("library="))
            fields = dict(re.findall(r"(\w+)=([^,]+)", line))
            assert int(fields["threads"]) == args.threads
            assert int(fields["rows"]) == rows and int(fields["cols"]) == cols
            assert int(fields["nnzA"]) == len(a)
            assert int(fields["nnzC"]) == len(expected)
            times = [float(fields["spgemm_" + name + "_s"]) for name in ("min", "mean", "max")]
            assert 0 <= times[0] <= times[1] + 1e-8 <= times[2] + 2e-8
            with timings.open() as f:
                rank_rows = list(csv.DictReader(f))
            assert [int(row["rank"]) for row in rank_rows] == list(range(args.ranks))
            elapsed = [float(row["elapsed_s"]) for row in rank_rows]
            assert all(math.isfinite(t) and t >= 0 for t in elapsed)
            for measured, reported in zip((min(elapsed), sum(elapsed) / len(elapsed), max(elapsed)), times):
                assert math.isclose(measured, reported, rel_tol=1e-8, abs_tol=1e-12)
            r, c, actual = read_mm(output)
            assert (r, c) == (rows, cols)
            for key in expected.keys() | actual.keys():
                assert math.isclose(expected.get(key, 0), actual.get(key, 0),
                                    abs_tol=1e-12, rel_tol=1e-12), (fixture, library, key)
            print(f"PASS {library}: {fixture.name}, {args.ranks} ranks")
