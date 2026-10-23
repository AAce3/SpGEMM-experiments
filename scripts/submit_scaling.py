#!/usr/bin/env python3
"""Compare 32 x 4 and 128 x 1 on Perlmutter; default: cage15, two nodes."""
import argparse
import csv
import math
from pathlib import Path
import shlex
import subprocess

NODES = (2, 8, 32)


def layout(nodes, per_node):
    ranks = nodes * per_node
    if math.isqrt(ranks) ** 2 != ranks:
        raise ValueError("CombBLAS needs a square total rank count")
    return ranks, per_node, 2 * (128 // per_node)


def read_manifest(path):
    pairs = []
    with path.open() as stream:
        for line_number, row in enumerate(csv.reader(stream, delimiter="\t"), 1):
            if not row or row[0].lstrip().startswith("#"):
                continue
            if len(row) != 2:
                raise ValueError(f"{path}:{line_number}: expected name<TAB>mtx")
            name, mtx = row
            if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
                raise ValueError(f"Invalid matrix label: {name}")
            # Relative paths are relative to the manifest, not the calling shell.
            resolve = lambda p: (path.parent / p).resolve()
            pairs.append((name, resolve(mtx)))
    if not pairs or len({row[0] for row in pairs}) != len(pairs):
        raise ValueError("Manifest must contain unique matrix names and at least one row")
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("benchmark_matrices.tsv"))
    parser.add_argument("--account", required=True)
    parser.add_argument("--time", default="01:00:00", help="Walltime limit per matrix/node job")
    parser.add_argument("--nodes", nargs="+", type=int, choices=NODES, default=[2])
    parser.add_argument("--matrices", nargs="+", default=["cage15"], help="Restrict to selected manifest labels")
    parser.add_argument("--submit", action="store_true", help="Submit; default only prints the plan")
    parser.add_argument("--profile", action="store_true", help="Add one mpiP run per library")
    parser.add_argument("--processes-per-node", nargs="+", type=int, choices=(32, 128), default=[32, 128])
    args = parser.parse_args()
    try:
        pairs = read_manifest(args.manifest.resolve())
    except (ValueError, OSError) as error:
        parser.error(str(error))
    if args.matrices:
        unknown = set(args.matrices) - {row[0] for row in pairs}
        if unknown:
            parser.error(f"Unknown matrix labels: {sorted(unknown)}")
        pairs = [row for row in pairs if row[0] in args.matrices]
    script = Path(__file__).resolve().with_name("profile_perlmutter.slurm")
    # Fail before any submission if a matrix is unavailable.
    if args.submit:
        for _, mtx in pairs:
            if not mtx.is_file():
                parser.error(f"Input missing: {mtx}; check the shared path before submission")
        import os
        mpip = os.environ.get("MPIP_LIB", "")
        if args.profile and (not mpip or not Path(mpip).is_file()):
            parser.error("Set MPIP_LIB to a shared mpiP library built for Perlmutter MPI")
        build = Path(os.environ.get("BUILD_DIR", str(Path.cwd() / "build")))
        for library in ("tpetra", "combblas"):
            binary = build / (library + "_spgemm")
            if not os.access(binary, os.X_OK):
                parser.error(f"Executable missing: {binary}")
    for name, mtx in pairs:
        for nodes in dict.fromkeys(args.nodes):
            for per_node in dict.fromkeys(args.processes_per_node):
                ranks, per_node, cpus = layout(nodes, per_node)
                command = ["sbatch", "--parsable", f"--account={args.account}",
                           f"--job-name=spgemm-{name}-n{nodes}-p{per_node}", f"--nodes={nodes}",
                           f"--ntasks={ranks}", f"--ntasks-per-node={per_node}",
                           f"--cpus-per-task={cpus}", f"--time={args.time}",
                           f"--export=ALL,PROFILE={int(args.profile)}",
                           str(script), str(mtx)]
                print(shlex.join(command), flush=True)
                if args.submit:
                    result = subprocess.run(command, check=True, capture_output=True, text=True)
                    print(f"SUBMITTED matrix={name} nodes={nodes} ranks={ranks} job={result.stdout.strip()}", flush=True)


if __name__ == "__main__":
    main()
