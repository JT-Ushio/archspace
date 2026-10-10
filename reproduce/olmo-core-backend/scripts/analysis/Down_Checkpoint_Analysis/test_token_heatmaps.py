"""CPU-only snapshot tests, independent of torch/olmo-core installations."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from token_heatmaps import FeatureRecorder, POINTS


class SnapshotTests(unittest.TestCase):
    def test_alignment_full_metrics_and_bounded_channels(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = FeatureRecorder(Path(tmp) / "features", sequences=1, tokens=2, channels=2)
            recorder.begin(0, np.array([1, 3, 7]), np.arange(10) + 100, {"input_sha256": "test"})
            x = np.array([[3., -4., 0., 0.], [1., 1., 1., 1.], [0., 0., 0., 0.]], dtype=np.float32)
            for point in POINTS:
                recorder.add(1, point, x)
            recorder.finish(1)
            with np.load(Path(tmp) / "features/sequence_000.npz", allow_pickle=False) as result:
                np.testing.assert_array_equal(result["positions"], [1, 7])
                np.testing.assert_array_equal(result["token_ids"], [101, 107])
                self.assertEqual(result["L01_a_values"].shape, (2, 4))
                self.assertEqual(result["L01_h_values"].shape, (2, 2))
                np.testing.assert_allclose(result["L01_h_metrics"], [[2.5, .25, .64], [0, 0, 0]])
            recorder.begin(1, np.array([1]), np.arange(10), {})
            recorder.add(1, "a", np.full((1, 4), np.nan))  # Unselected sequences are ignored.
            recorder.finish(1)
            self.assertEqual(len(list((Path(tmp) / "features").glob("*.npz"))), 1)

    def test_missing_duplicate_nonfinite_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = FeatureRecorder(Path(tmp) / "features")
            recorder.begin(0, np.array([0]), np.array([10]), {})
            with self.assertRaises(ValueError):
                recorder.add(1, "a", np.array([[np.nan]]))
            recorder.add(1, "a", np.ones((1, 4)))
            with self.assertRaises(ValueError):
                recorder.add(1, "a", np.ones((1, 4)))
            with self.assertRaises(ValueError):
                recorder.finish(1)

    def test_cross_model_token_mismatch_rejected(self):
        from plot_token_heatmaps import load_snapshots
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "manifest.json").write_text(json.dumps({"status": "complete"}))
            for run in ("S", "I", "R", "SR"):
                recorder = FeatureRecorder(root / run / "token_features")
                ids = np.array([10 if run == "S" else 11])
                recorder.begin(0, np.array([0]), ids, dict(input_sha256="test", dataset_index=0, source="test"))
                for point in POINTS:
                    recorder.add(1, point, np.ones((1, 4)))
                recorder.finish(1)
            with self.assertRaisesRegex(ValueError, "token alignment"):
                load_snapshots(root, 0)


if __name__ == "__main__":
    unittest.main()
