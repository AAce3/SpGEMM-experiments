#!/usr/bin/env python3
import importlib.util
import math
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("submit_scaling", ROOT / "scripts/submit_scaling.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ScalingPlanTests(unittest.TestCase):
    def test_layouts(self):
        for nodes in module.NODES:
            for per_node in (32, 128):
                ranks, cap, cpus = module.layout(nodes, per_node)
                self.assertEqual(math.isqrt(ranks) ** 2, ranks)
                self.assertEqual(ranks, nodes * cap)
                self.assertEqual(cap * cpus, 256)

    def test_default_manifest(self):
        rows = module.read_manifest(ROOT / "benchmark_matrices.tsv")
        self.assertEqual(len(rows), 10)
        folder = Path("/global/cfs/cdirs/m4646/julian-aaron-shared/matrices")
        self.assertEqual({name for name, _ in rows}, {
            "cage15", "delaunay_n24", "eukarya", "HV15R", "it-2004",
            "mouse_gene", "nlpkkt200", "Queen_4147", "stokes", "subgraph4"})
        self.assertTrue(all(mtx == folder / (name + ".mtx") for name, mtx in rows))

    def test_plan_and_pilot(self):
        script = str(ROOT / "scripts/submit_scaling.py")
        base = ["python3", script, "--account", "test", "--manifest", str(ROOT / "benchmark_matrices.tsv")]
        result = subprocess.run(base, text=True, capture_output=True, check=True)
        self.assertEqual(len(result.stdout.splitlines()), 2)
        self.assertIn("--nodes=2 --ntasks=64", result.stdout)
        self.assertIn("/global/cfs/cdirs/m4646/julian-aaron-shared/matrices/cage15.mtx", result.stdout)
        result = subprocess.run(base + ["--nodes", "8", "--matrices", "cage15", "--processes-per-node", "128"],
                                text=True, capture_output=True, check=True)
        self.assertEqual(len(result.stdout.splitlines()), 1)
        self.assertIn("--ntasks=1024", result.stdout)

    def test_duplicate_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "list.tsv"
            path.write_text("same\ta.mtx\nsame\tb.mtx\n")
            with self.assertRaises(ValueError):
                module.read_manifest(path)


if __name__ == "__main__":
    unittest.main()
