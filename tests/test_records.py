import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mc_search import MonteCarloSearch

import numpy as np

from ddg import ALPHABET, DDGLookup
from provenance import git_identity, sha256_file


class ArtifactTests(unittest.TestCase):
    def test_legacy_and_versioned_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "table.npz"
            fields = dict(ddg=np.zeros((1, 1, 20)), sequence="A", alphabet=ALPHABET)
            np.savez(path, **fields)
            old = DDGLookup(path)
            self.assertIsNone(old.artifact["provenance"])
            self.assertEqual(old.artifact["sha256"], sha256_file(path))
            provenance = {"schema_version": 1, "checkpoint_sha256": "example"}
            np.savez(path, **fields, provenance_json=json.dumps(provenance))
            new = DDGLookup(path)
            self.assertNotEqual(old.artifact["sha256"], new.artifact["sha256"])
            self.assertEqual(new.artifact["provenance"], provenance)
            self.assertEqual(json.loads(json.dumps(new.artifact)), new.artifact)
            np.testing.assert_equal(old.table, new.table)

    def test_unknown_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(git_identity(directory), {"commit": None, "dirty": None})


class FakeEvaluator:
    predictor = None
    ref_coords = None
    w_plddt = 1.0
    w_rmsd = 1.0

    def __init__(self):
        self.calls = 0

    def evaluate(self, seq):
        self.calls += 1
        return dict(seq=seq, fitness=float(self.calls), plddt=0.5, rmsd=1.0,
                    extra_metric=float(self.calls * 10), perm=(0,), pdb_str="dummy")

    def format_metrics(self, record):
        return "dummy"


class InitialRecordTests(unittest.TestCase):
    def test_initial_survives_new_best_and_final(self):
        mc = MonteCarloSearch(initial_seq="A", evaluator=FakeEvaluator())
        with patch("mc_search.mutate_sequence", return_value="C"):
            result = mc.run(n_steps=1)
        self.assertEqual(result["initial_metrics"]["extra_metric"], 10.0)
        self.assertEqual(result["best_metrics"]["extra_metric"], 20.0)
        self.assertEqual(result["final_metrics"]["extra_metric"], 20.0)
        self.assertEqual([r["step"] for r in result["history"]], [1])
        self.assertNotIn("pdb_str", result["initial_metrics"])
        self.assertNotIn("perm", result["initial_metrics"])
        self.assertEqual(json.loads(json.dumps(result))["initial_metrics"],
                         result["initial_metrics"])

    def test_zero_steps(self):
        mc = MonteCarloSearch(initial_seq="A", evaluator=FakeEvaluator())
        result = mc.run(n_steps=0)
        self.assertEqual(result["history"], [])
        self.assertEqual(result["initial_metrics"], result["best_metrics"])
        self.assertEqual(result["initial_metrics"], result["final_metrics"])


if __name__ == "__main__":
    unittest.main()
