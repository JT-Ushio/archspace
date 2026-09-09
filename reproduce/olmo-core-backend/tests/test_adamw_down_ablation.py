"""Run directly with Python; config/meta-model/CPU checks need no GPU or pytest."""
import copy
import runpy
import shlex
import sys
import unittest
from argparse import Namespace
from pathlib import Path

import torch

from olmo_core.optim import AdamWConfig, MuonConfig
from olmo_core.script_utils import ExperimentConfig

BACKEND_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = BACKEND_DIR / "cfps/BS256_Seq8K_30B_WSD_FP8/AdamW_Down_Ablations"
sys.path.insert(0, str(CONFIG_DIR))
from _adamw_common import build_down_activation_config  # noqa: E402


def build(label, overrides=None):
    opts = Namespace(sequence_length=None, data_root="/tmp/data", work_dir="/tmp/work",
                     save_folder="/tmp/checkpoints", name="adamw-ablation-test")
    entry = runpy.run_path(str(CONFIG_DIR / f"down_{label}_adamw_fp8.py"))
    return entry["build_config"](opts, overrides or [])


class AdamWDownTests(unittest.TestCase):
    def test_config_and_launcher_consistency(self):
        normalized = []
        for label, activation in (("silu", "silu"), ("rms", "rms_norm")):
            config = build(label)
            self.assertIsInstance(config.train_module.optim, AdamWConfig)
            self.assertEqual(config.train_module.optim.lr, .001)
            self.assertEqual(config.train_module.optim.betas, (.9, .95))
            self.assertEqual(config.train_module.optim.weight_decay, .033)
            self.assertEqual(config.train_module.optim.eps, 1e-8)
            self.assertEqual(str(config.model.block.feed_forward.down_nonlinearity), activation)
            launcher = BACKEND_DIR / "scripts/train/AdamW_Down_Ablations" / f"run_down_{label}_adamw_fp8.sh"
            flags = []
            for token in shlex.split(launcher.read_text().replace("\\\n", " ")):
                flag = token.removeprefix("--")
                if flag.startswith(("model.", "train_module.", "trainer.", "data_loader.")) and "=" in flag:
                    if not flag.startswith("trainer.work_dir="):
                        flags.append(flag)
            self.assertEqual(config.as_config_dict(), build(label, flags).as_config_dict())
            restored = ExperimentConfig.from_dict(config.as_config_dict())
            self.assertIsInstance(restored.train_module.optim, AdamWConfig)
            self.assertEqual(restored.train_module.optim.as_config_dict(),
                             config.train_module.optim.as_config_dict())
            # Changing the optimizer must leave the existing model/data/schedule intact.
            opts = Namespace(sequence_length=None, data_root="/tmp/data", work_dir="/tmp/work",
                             save_folder="/tmp/checkpoints", name="adamw-ablation-test")
            baseline = build_down_activation_config(
                opts, [], variant=config.trainer.callbacks["wandb"].group,
                down_nonlinearity=config.model.block.feed_forward.down_nonlinearity,
            )
            comparison = copy.deepcopy(config)
            comparison.train_module.optim = baseline.train_module.optim
            self.assertEqual(comparison.as_config_dict(), baseline.as_config_dict())
            config.model.block.feed_forward.down_nonlinearity = "silu"
            config.trainer.callbacks["wandb"].group = "paired-test"
            normalized.append(config.as_config_dict())
        self.assertEqual(normalized[0], normalized[1])
        self.assertEqual(build("silu", ["train_module.optim.lr=0.002"]).train_module.optim.lr, .002)

    def test_real_model_weight_decay_groups_match_muon(self):
        for label in ("silu", "rms"):
            config = build(label)
            model = config.model.build(init_device="meta")
            groups = config.train_module.optim.build_groups(model)
            names_by_id = {id(p): n for n, p in model.named_parameters() if p.requires_grad}
            seen = []
            actual_no_decay = set()
            for group in groups:
                self.assertNotIn("algorithm", group)
                for parameter in group["params"]:
                    seen.append(id(parameter))
                    if group.get("weight_decay", config.train_module.optim.weight_decay) == 0:
                        actual_no_decay.add(names_by_id[id(parameter)])
            expected = MuonConfig().categorize_parameters(model)
            self.assertEqual(actual_no_decay, set(expected["embed"] + expected["vector"]))
            self.assertEqual(set(seen), set(names_by_id))
            self.assertEqual(len(seen), len(set(seen)))

    def test_real_adamw_cpu_update(self):
        # Small stand-in with the same no-decay naming conventions.
        model = torch.nn.Module()
        model.embeddings = torch.nn.Embedding(8, 4)
        model.norm = torch.nn.LayerNorm(4, bias=False)
        model.projection = torch.nn.Linear(4, 4, bias=False)
        optimizer = build("silu").train_module.optim.build(model)
        self.assertIs(type(optimizer), torch.optim.AdamW)
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        sum(p.square().sum() for p in model.parameters()).backward()
        optimizer.step()
        for name, parameter in model.named_parameters():
            self.assertFalse(torch.equal(parameter, before[name]))
            self.assertEqual(optimizer.state[parameter]["step"].item(), 1)
        for group in optimizer.param_groups:
            self.assertEqual(group["lr"], .001)
            self.assertNotIn("algorithm", group)


if __name__ == "__main__":
    unittest.main()
