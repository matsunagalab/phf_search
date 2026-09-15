import json
from pathlib import Path
import tempfile
import unittest

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


if __name__ == "__main__":
    unittest.main()
