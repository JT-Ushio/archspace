import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/analysis/Initialization_Analysis"))
from compare_training import build


class InitializationComparisonTests(unittest.TestCase):
    def fixture(self, root):
        config = {"data_loader": {"global_batch_size": 8}, "init_seed": 34521,
                  "model": {"factor_init": "independent", "factor_init_artifacts": "same"}}
        for label in ("A", "C"):
            cfg = copy.deepcopy(config)
            if label == "C":
                cfg["model"]["factor_init"] = "svd_scale_matched"
            (root / f"{label}_metadata.json").write_text(json.dumps(dict(
                state="finished", config=cfg, summary={"_step": 1001})))
            rows = []
            for step in range(1002):
                row = {"_step": step, "throughput/total tokens": step * 8,
                       "train/CE loss": 2 if label == "A" else 1.9}
                if step in (500, 1000):
                    row["eval/lm/toy/CE loss"] = 3 if label == "A" else 2.8
                rows.append(row)
            self.write_rows(root, label, rows)

    def write_rows(self, root, label, rows):
        (root / f"{label}_history.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    def compare(self, root):
        return build(root, budget=8008, batch_tokens=8)

    def test_report_and_preserve_outputs(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.fixture(root)
            result = self.compare(root)
            self.assertEqual(result[1]["validation_step"], 1000)
            self.assertAlmostEqual(result[1]["validation_gain_vs_A"], .2)
            self.assertAlmostEqual(result[1]["train_ce_last_1000"], 1.9)
            for path in (root / "training_comparison").glob("*.svg"):
                ElementTree.parse(path)
            with self.assertRaises(FileExistsError):
                self.compare(root)

    def test_config_and_terminal_gates(self):
        for mutation, message in (
            (lambda m: m.update(state="running"), "not finished"),
            (lambda m: m["config"].update(init_seed=1), "configuration difference"),
            (lambda m: m["config"]["model"].update(factor_init="svd_raw"), "initialization mode"),
            (lambda m: m["summary"].update(_step=1000), "incomplete endpoint"),
        ):
            with self.subTest(message=message), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                self.fixture(root)
                path = root / "C_metadata.json"
                meta = json.loads(path.read_text())
                mutation(meta)
                path.write_text(json.dumps(meta))
                with self.assertRaisesRegex(ValueError, message):
                    self.compare(root)
                self.assertFalse((root / "training_comparison").exists())

    def test_missing_extra_nonfinite_and_duplicate_records(self):
        mutations = (
            (lambda rows: rows[500].pop("eval/lm/toy/CE loss"), "validation"),
            (lambda rows: rows[500].update({"eval/lm/extra/CE loss": 3}), "validation"),
            (lambda rows: rows[501].update({"train/CE loss": float("nan")}), "train CE"),
            (lambda rows: rows.append(rows[-1]), "duplicate"),
            (lambda rows: rows[5].update({"throughput/total tokens": 0}), "Step/token mismatch"),
        )
        for mutation, message in mutations:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                self.fixture(root)
                rows = [json.loads(line) for line in (root / "C_history.jsonl").read_text().splitlines()]
                mutation(rows)
                self.write_rows(root, "C", rows)
                with self.assertRaisesRegex(ValueError, message):
                    self.compare(root)


if __name__ == "__main__":
    unittest.main()
