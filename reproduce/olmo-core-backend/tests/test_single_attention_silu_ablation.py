"""Each new experiment differs from all-linear at exactly one projection."""
import copy
import runpy
import shlex
import unittest
from argparse import Namespace
from pathlib import Path

import torch
import torch.nn.functional as F
from olmo_mose import LowRankAttentionConfig, MoSENonlinearity

BACKEND = Path(__file__).resolve().parents[1]
FOLDER = "All_Linear_Attention_SiLU_Ablations"


class SingleAttentionSiLUTests(unittest.TestCase):
    def test_four_configs_launchers_and_invariants(self):
        opts = Namespace(sequence_length=None, data_root="/tmp/data", work_dir="/tmp/work",
                         save_folder="/tmp/checkpoints", name="test-single-silu")
        base = runpy.run_path(str(BACKEND / "cfps/BS256_Seq8K_30B_WSD_FP8/All_Linear_Ablation/all_linear_fp8.py"))["build_config"](opts, [])
        names, ports = set(), set()
        for selected in ("q", "k", "v", "o"):
            with self.subTest(selected=selected):
                path = BACKEND / "cfps/BS256_Seq8K_30B_WSD_FP8" / FOLDER / f"{selected}_only_silu_fp8.py"
                build = runpy.run_path(str(path))["build_config"]
                config = build(opts, [])
                expected = copy.deepcopy(base)
                setattr(expected.model.block.sequence_mixer, f"{selected}_nonlinearity", MoSENonlinearity.silu)
                expected.trainer.callbacks["wandb"].group = config.trainer.callbacks["wandb"].group
                self.assertEqual(expected.as_config_dict(), config.as_config_dict())
                shell = (BACKEND / "scripts/train" / FOLDER / f"run_{selected}_only_silu_fp8.sh").read_text()
                flags = []
                for token in shlex.split(shell.replace("\\\n", " ")):
                    flag = token.removeprefix("--")
                    if flag.startswith(("model.", "train_module.", "trainer.", "data_loader.")) and "=" in flag:
                        if not flag.startswith("trainer.work_dir="):
                            flags.append(flag)
                self.assertEqual(build(opts, flags).as_config_dict(), config.as_config_dict())
                self.assertIn(f"{FOLDER}/{selected}_only_silu_fp8.py", shell)
                self.assertIn("export WANDB_RESUME=allow", shell)
                names.add(next(line for line in shell.splitlines() if line.startswith("RUN_NAME=")))
                ports.add(next(line for line in shell.splitlines() if line.startswith("MASTER_PORT=")))
                self.assertEqual(config.model.block.sequence_mixer.rank, 512)
                self.assertEqual(str(config.model.block.sequence_mixer.share_scope), "none")
                self.assertEqual(config.model.block.feed_forward.r2, 512)
                self.assertEqual(config.model.block.feed_forward.down_r2, 512)
                self.assertFalse(config.model.block.feed_forward.share_gate_up_subspace)
                for name in ("gate", "up", "down"):
                    self.assertEqual(str(getattr(config.model.block.feed_forward, f"{name}_nonlinearity")), "identity")
                self.assertEqual(config.data_loader.global_batch_size, 256 * 8192)
                self.assertEqual(config.train_module.rank_microbatch_size, 32768)
                self.assertEqual(config.data_loader.global_batch_size // (16 * 32768), 4)
                self.assertEqual(config.train_module.optim.lr, 0.005)
                self.assertEqual(config.train_module.scheduler.warmup, 2000)
                self.assertEqual(config.train_module.scheduler.decay_fraction, 0.2)
                self.assertTrue(config.train_module.float8_config.enabled)
                self.assertEqual(config.trainer.max_duration.value, 30_000_000_000)
                self.assertEqual(str(config.trainer.max_duration.unit), "tokens")
                self.assertEqual(config.trainer.callbacks["checkpointer"].save_interval, 1788)
                self.assertIsNone(config.trainer.callbacks["checkpointer"].max_checkpoints)
                self.assertIsNot(config.trainer.load_optim_state, False)
                self.assertIsNot(config.trainer.load_trainer_state, False)
        self.assertEqual(len(names), 4)
        self.assertEqual(len(ports), 4)

    def test_projection_initialization_shapes_and_gradients(self):
        def build(selected=None):
            torch.manual_seed(42)
            return LowRankAttentionConfig(
                n_heads=2, rank=8, bias=False, share_scope="none",
                **{f"{p}_nonlinearity": "silu" if selected == p else "identity" for p in ("q", "k", "v", "o")},
            ).build(8, layer_idx=0, n_layers=1)
        base = build()
        for selected in ("q", "k", "v", "o"):
            model = build(selected)
            self.assertEqual(list(model.state_dict()), list(base.state_dict()))
            self.assertEqual(sum(p.numel() for p in model.parameters()), sum(p.numel() for p in base.parameters()))
            for name, tensor in model.state_dict().items():
                torch.testing.assert_close(tensor, base.state_dict()[name], atol=0, rtol=0)
            for p, attr in zip(("q", "k", "v", "o"), ("w_q", "w_k", "w_v", "w_out")):
                projection = getattr(model, attr)
                reference = copy.deepcopy(projection)
                x = torch.randn(2, 3, 8, requires_grad=True)
                xr = x.detach().clone().requires_grad_(True)
                latent = reference.v(xr)
                expected = reference.u(F.silu(latent) if p == selected else latent)
                actual = projection(x)
                torch.testing.assert_close(actual, expected)
                actual.square().sum().backward()
                expected.square().sum().backward()
                torch.testing.assert_close(x.grad, xr.grad)
                for param, ref in zip(projection.parameters(), reference.parameters()):
                    torch.testing.assert_close(param.grad, ref.grad)


if __name__ == "__main__":
    unittest.main()
