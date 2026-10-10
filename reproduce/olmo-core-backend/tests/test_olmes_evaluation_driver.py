"""CPU checks for scheduling and result preservation; never load model weights."""

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1] / "scripts/eval/OLMES_Easy"
spec = importlib.util.spec_from_file_location("olmes_easy_driver", HERE / "run_easy_dp_sharded.py")
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


def tasks(*aliases):
    return [{"task_name": name, "metadata": {"alias": name}} for name in aliases]


class EvaluationDriverTests(unittest.TestCase):
    def test_historical_balance_preserves_every_task(self):
        costs = driver.load_timings(HERE / "historical_timings.json")
        configs = tasks(*costs)
        groups, loads, _ = driver.assign_tasks(configs, 8, costs)
        _, previous, _ = driver.assign_tasks(configs, 8, costs, "round-robin")
        self.assertEqual(sorted(i for group in groups for i, _ in group), list(range(192)))
        self.assertLess(max(loads), 1200)
        self.assertLess(max(loads), max(previous) * 0.5)
        self.assertEqual(sum(len(group) for group in groups), len(configs))

    def test_allocated_device_mask_is_preserved(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "3,5,7,9"}):
            self.assertEqual(driver.allocated_devices(4), ["3", "5", "7", "9"])
            env = driver.worker_environment("7", 2)
            self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "7")
        self.assertEqual(driver.allocated_devices(2, "GPU-aaa,GPU-bbb"), ["GPU-aaa", "GPU-bbb"])
        for invalid in ["", "0,0", "-1", "0"]:
            with self.assertRaises(ValueError):
                driver.allocated_devices(2, invalid)

    def test_worker_runs_one_gpu_without_tensor_parallel(self):
        command = driver.worker_command(
            SimpleNamespace(model=Path("model with spaces")),
            Path("spec.jsonl"),
            Path("output with spaces"),
            64,
        )
        self.assertEqual(command[command.index("--gpus") + 1], "1")
        self.assertEqual(command[command.index("--num-workers") + 1], "1")
        self.assertEqual(command[command.index("--batch-size") + 1], "64")
        self.assertNotIn("use_cache", json.loads(command[command.index("--model-args") + 1]))
        self.assertNotIn("--limit", command)

    def test_scoring_config_preserves_source_and_weight_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "original"
            model.mkdir()
            source_config = {"model_type": "olmo_mose", "use_cache": True}
            (model / "config.json").write_text(json.dumps(source_config))
            (model / "model.safetensors").write_bytes(b"unchanged weights")
            target = driver.prepare_scoring_model(model, root)
            self.assertEqual(json.loads((model / "config.json").read_text()), source_config)
            self.assertFalse(json.loads((target / "config.json").read_text())["use_cache"])
            self.assertTrue((target / "model.safetensors").is_symlink())
            self.assertEqual((target / "model.safetensors").read_bytes(), b"unchanged weights")

    def make_results(self, root):
        driver.write_specs(tasks("a", "b", "c"), root, 2, {"a": 1, "b": 10, "c": 4})
        manifest = json.loads((root / "shard_manifest.json").read_text())
        for shard_idx in range(2):
            folder = root / "shards" / f"shard-{shard_idx:02d}"
            folder.mkdir(parents=True)
            rows = sorted(
                (r for r in manifest["tasks"] if r["shard_idx"] == shard_idx),
                key=lambda r: r["local_idx"],
            )
            summary = {
                "tasks": [{"alias": r["alias"], "score": r["global_idx"]} for r in rows],
                "all_primary_scores": [r["alias"] for r in rows],
                "model_config": {},
            }
            (folder / "metrics.json").write_text(json.dumps(summary))
            (folder / "metrics-all.jsonl").write_text(
                "".join(
                    json.dumps({"task_idx": r["local_idx"], "score": r["global_idx"]}) + "\n"
                    for r in rows
                )
            )
        return manifest

    def test_results_return_to_original_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_results(root)
            driver.merge_results(root, 2, 3, 32)
            summary = json.loads((root / "metrics.json").read_text())
            self.assertEqual([r["alias"] for r in summary["tasks"]], ["a", "b", "c"])
            self.assertEqual([r["score"] for r in summary["tasks"]], [0, 1, 2])
            self.assertEqual(summary["all_primary_scores"], ["a", "b", "c"])

    def test_missing_or_duplicate_task_cannot_be_marked_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.make_results(root)
            group = max(range(2), key=lambda i: sum(r["shard_idx"] == i for r in manifest["tasks"]))
            p = root / "shards" / f"shard-{group:02d}" / "metrics-all.jsonl"
            rows = p.read_text().splitlines()
            p.write_text("\n".join([rows[0]] * len(rows)) + "\n")
            with self.assertRaises(RuntimeError):
                driver.merge_results(root, 2, 3, 32)
            self.assertFalse((root / "evaluation_complete.json").exists())

    def test_only_oom_worker_retries_at_smaller_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            configs = driver.write_specs(tasks("a", "b"), root, 2)
            args = SimpleNamespace(
                output_dir=root,
                num_shards=2,
                devices="GPU-a,GPU-b",
                batch_size=64,
                min_batch_size=16,
                oom_retry=True,
                gpu_sample_seconds=0,
                olmes_repo=root,
            )
            program = (
                "import sys; from pathlib import Path; "
                "p=Path(sys.argv[1]); p.mkdir(parents=True,exist_ok=True); "
                "bs=int(sys.argv[2]); failed=p.name=='shard-01' and bs==64; "
                "print('CUDA out of memory' if failed else 'ok',flush=True); "
                "sys.exit(1 if failed else 0)"
            )
            with patch.object(
                driver,
                "worker_command",
                side_effect=lambda a, s, o, b: [sys.executable, "-c", program, str(o), str(b)],
            ):
                workers = driver.launch_shards(args, configs, 2)
            self.assertEqual([w["batch_size"] for w in workers], [64, 32])
            self.assertEqual([len(w["attempts"]) for w in workers], [1, 2])

    def test_wall_time_profile_includes_inference_interval(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            driver.write_specs(tasks("one"), root, 1)
            (root / "logs").mkdir()
            (root / "logs/shard-00.log").write_text(
                "2026-08-31:02:04:00,000 INFO **** Processing task: one ****\n"
                "2026-08-31:02:05:00,000 INFO Task one processing time: 0.12 seconds\n"
                "2026-08-31:02:05:01,000 INFO Saving final metrics in output\n"
            )
            self.assertEqual(driver.collect_task_timings(root)["task_wall_seconds"], {"one": 61.0})


if __name__ == "__main__":
    unittest.main()
