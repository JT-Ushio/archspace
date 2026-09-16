import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[1] / "scripts/analysis/QK_Combination_Analysis"
sys.path.insert(0, str(ROOT))
from control_trends import build, load_history
from final_compare import audit_resume, interaction


class ControlTrendsTests(unittest.TestCase):
    def test_resume_audit_and_interaction_sign(self):
        rows = {i: {"_timestamp": 20 if i > 1 else 5} for i in range(4)}
        self.assertEqual(audit_resume(rows, 1, 10)["continuing_rows"], 2)
        rows[2]["_timestamp"] = 9
        with self.assertRaisesRegex(ValueError, "another attempt"):
            audit_resume(rows, 1, 10)
        del rows[2]
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            audit_resume(rows, 1, 10)
        self.assertAlmostEqual(interaction({"L": 3, "Q": 2.9, "K": 2.8, "QK": 2.6}), .1)

    def fixture(self, root):
        config = {"data_loader": {"global_batch_size": 8}, "model": {"block": {
            "sequence_mixer": {"q_nonlinearity": "identity"}}}}
        for label in ("L", "Q"):
            cfg = copy.deepcopy(config)
            if label == "Q":
                cfg["model"]["block"]["sequence_mixer"]["q_nonlinearity"] = "silu"
            (root / f"{label}_metadata.json").write_text(json.dumps(dict(
                state="finished", config=cfg, summary={"_step": 501},
                url=f"https://example.invalid/{label}", exported_at="test")))
            with (root / f"{label}_history.jsonl").open("w") as stream:
                for step in range(502):
                    row = {"_step": step, "throughput/total tokens": step * 8,
                           "train/CE loss": 2 if label == "L" else 1.5}
                    if step == 500:
                        row["eval/lm/toy/CE loss"] = 3 if label == "L" else 2.8
                    stream.write(json.dumps(row) + "\n")

    def test_aligned_metrics_and_svg(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            result = build(root, ["L", "Q"])
            self.assertEqual(result[1]["validation_step"], 500)
            self.assertEqual(result[1]["train_step"], 501)
            self.assertAlmostEqual(result[1]["validation_gain_vs_L"], .2)
            self.assertAlmostEqual(result[1]["train_ce_last_500"], 1.5)
            for svg in (root / "trends").glob("*.svg"):
                ElementTree.parse(svg)
            with self.assertRaises(FileExistsError):
                build(root, ["L", "Q"])

    def test_reject_token_disagreement_and_duplicate_steps(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "history.jsonl"
            path.write_text(json.dumps({"_step": 2, "throughput/total tokens": 8}) + "\n")
            with self.assertRaisesRegex(ValueError, "Step/token mismatch"):
                load_history(path, 8)
            row = json.dumps({"_step": 1, "throughput/total tokens": 8}) + "\n"
            path.write_text(row * 2)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_history(path, 8)

    def test_reject_changed_seed_and_unfinished_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            path = root / "Q_metadata.json"
            meta = json.loads(path.read_text())
            meta["config"]["init_seed"] = 123
            path.write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, "configuration difference"):
                build(root, ["L", "Q"])
            del meta["config"]["init_seed"]
            meta["state"] = "running"
            path.write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, "not finished"):
                build(root, ["L", "Q"])


if __name__ == "__main__":
    unittest.main()
