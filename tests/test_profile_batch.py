#!/usr/bin/env python3
"""Exercise the batch orchestration with fake Slurm tools; submits no jobs."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/profile_perlmutter.slurm"


class BatchTests(unittest.TestCase):
    def run_batch(self, missing_report=False, nodes=2, per_node=32, profile=True, wrong_threads=False):
        ranks = nodes * per_node
        threads = 128 // per_node
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tools = root / "tools"
            tools.mkdir()
            build = root / "build"
            build.mkdir()
            for library in ("tpetra", "combblas"):
                binary = build / (library + "_spgemm")
                binary.write_text("#!/bin/sh\nexit 0\n")
                binary.chmod(0o755)
            for filename in ("input.mtx", "libmpiP.so"):
                (root / filename).touch()
            for name in ("module", "scontrol", "ldd"):
                path = tools / name
                path.write_text("#!/bin/sh\necho mock\n")
                path.chmod(0o755)
            srun = tools / "srun"
            srun.write_text("""#!/usr/bin/env python3
import json, os, pathlib, sys
args = sys.argv[1:]
with open(os.environ["CALLS"], "a") as f:
    f.write(json.dumps(args) + "\\n")
for flag in ("--nodes=" + os.environ["SLURM_JOB_NUM_NODES"],
             "--ntasks=" + os.environ["SLURM_NTASKS"], "--ntasks-per-node=" + os.environ["SLURM_NTASKS_PER_NODE"],
             "--cpus-per-task=" + os.environ["SLURM_CPUS_PER_TASK"], "--cpu-bind=cores", "--export=ALL"):
    assert flag in args, flag
assert int(os.environ["OMP_NUM_THREADS"]) * int(os.environ["SLURM_NTASKS_PER_NODE"]) == 128
assert os.environ["OMP_PLACES"] == "cores"
assert "LD_PRELOAD" not in os.environ
exe_index = next(i for i, a in enumerate(args) if a.endswith("_spgemm"))
assert pathlib.Path(args[exe_index + 1]).resolve() == pathlib.Path(os.environ["INPUT_MATRIX"]).resolve()
reported_threads = "1" if os.environ.get("WRONG_THREADS") == "1" else os.environ["OMP_NUM_THREADS"]
directory = pathlib.Path(next(a.split("=",1)[1] for a in args if a.startswith("--chdir=")))
if any(a.endswith("tpetra_spgemm") for a in args):
    print("library=tpetra, backend=OpenMP, threads=" + reported_threads + ", matrix=mock")
else:
    print("library=combblas, threads=" + reported_threads + ", matrix=mock")
profile = "env" in args
if profile:
    assert "MPIP=-o -l -k 3 -p -y" in args
    assert any(a.startswith("LD_PRELOAD=") for a in args)
option = "--rank-times"
csv = pathlib.Path(args[args.index(option)+1])
csv.write_text("rank,elapsed_s\\n" + "".join(f"{r},0.1\\n" for r in range(int(os.environ["SLURM_NTASKS"]))))
if profile and os.environ.get("MISSING_REPORT") != "1":
    (directory / "fake.4096.mpiP").write_text("mock report")
""")
            srun.chmod(0o755)
            env = dict(os.environ, PATH=str(tools) + os.pathsep + os.environ["PATH"],
                       SLURM_JOB_ID="123", SLURM_JOB_NUM_NODES=str(nodes),
                       SLURM_NTASKS=str(ranks), SLURM_CPUS_PER_TASK=str(threads * 2),
                       SLURM_NTASKS_PER_NODE=str(per_node), PROFILE=str(int(profile)),
                       SLURM_SUBMIT_DIR=str(root), SLURM_JOB_NODELIST="mock",
                       MPIP_LIB=str(root / "libmpiP.so"), BUILD_DIR=str(build),
                       RESULTS_DIR=str(root / "results"), BASELINE_REPEATS="3",
                       CALLS=str(root / "calls.jsonl"), INPUT_MATRIX=str(root / "input.mtx"),
                       WRONG_THREADS=str(int(wrong_threads)),
                       MISSING_REPORT="1" if missing_report else "0")
            env.pop("LD_PRELOAD", None)
            if not profile:
                env.pop("MPIP_LIB", None)
            result = subprocess.run(["bash", str(SCRIPT), str(root / "input.mtx")],
                                    env=env, text=True, capture_output=True, timeout=30)
            calls = [json.loads(line) for line in (root / "calls.jsonl").read_text().splitlines()]
            return result, calls

    def test_baselines_and_profiles(self):
        result, calls = self.run_batch()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(calls), 8)
        self.assertEqual(sum("env" in call for call in calls), 2)
        libraries = [next(Path(a).name for a in call if a.endswith("_spgemm")) for call in calls]
        self.assertEqual(libraries, ["tpetra_spgemm", "combblas_spgemm",
                                    "combblas_spgemm", "tpetra_spgemm",
                                    "tpetra_spgemm", "combblas_spgemm",
                                    "tpetra_spgemm", "combblas_spgemm"])

    def test_all_scaling_layouts(self):
        for nodes in (2, 8, 32):
            for per_node in (32, 128):
                with self.subTest(nodes=nodes, per_node=per_node):
                    result, calls = self.run_batch(nodes=nodes, per_node=per_node, profile=False)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertEqual(len(calls), 6)
                    self.assertFalse(any("env" in call for call in calls))

    def test_wrong_thread_count_is_an_error(self):
        result, calls = self.run_batch(wrong_threads=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requested thread count", result.stderr)
        self.assertEqual(len(calls), 1)

    def test_missing_profile_is_an_error(self):
        result, calls = self.run_batch(missing_report=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("No mpiP report", result.stderr)
        self.assertEqual(len(calls), 7)


if __name__ == "__main__":
    unittest.main()
