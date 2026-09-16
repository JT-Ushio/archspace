"""The QK combination completes the existing matched 2x2, without other changes."""
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
FOLDER = "All_Linear_QK_Combination"


class QKCombinationTests(unittest.TestCase):
    def test_complete_config_and_launcher_match_control(self):
        opts = Namespace(sequence_length=None, data_root="/tmp/data", work_dir="/tmp/work",
                         save_folder="/tmp/checkpoints", name="test-qk")
        root = BACKEND / "cfps/BS256_Seq8K_30B_WSD_FP8"
        base = runpy.run_path(str(root / "All_Linear_Ablation/all_linear_fp8.py"))["build_config"](opts, [])
        build = runpy.run_path(str(root / FOLDER / "qk_silu_fp8.py"))["build_config"]
        config = build(opts, [])
        expected = copy.deepcopy(base)
        expected.model.block.sequence_mixer.q_nonlinearity = MoSENonlinearity.silu
        expected.model.block.sequence_mixer.k_nonlinearity = MoSENonlinearity.silu
        expected.trainer.callbacks["wandb"].group = "all-linear-qk-combination-fp8-30b"
        self.assertEqual(config.as_config_dict(), expected.as_config_dict())
        shell = (BACKEND / "scripts/train" / FOLDER / "run_qk_silu_fp8.sh").read_text()
        flags = []
        for token in shlex.split(shell.replace("\\\n", " ")):
            flag = token.removeprefix("--")
            if flag.startswith(("model.", "train_module.", "trainer.", "data_loader.")) and "=" in flag:
                if not flag.startswith("trainer.work_dir="):
                    flags.append(flag)
        self.assertEqual(build(opts, flags).as_config_dict(), config.as_config_dict())
        self.assertIn(f"{FOLDER}/qk_silu_fp8.py", shell)
        self.assertIn("olmo3-1b-all-linear-qk-silu-fp8-30b", shell)
        self.assertEqual(config.data_loader.global_batch_size // (16 * config.train_module.rank_microbatch_size), 4)
        # CLI overrides still have last precedence.
        overridden = build(opts, ["model.block.sequence_mixer.k_nonlinearity=identity"])
        self.assertEqual(str(overridden.model.block.sequence_mixer.k_nonlinearity), "identity")

    def test_same_parameters_init_and_correct_gradients(self):
        def build(nonlinear):
            torch.manual_seed(42)
            return LowRankAttentionConfig(
                n_heads=2, rank=8, bias=False, share_scope="none",
                q_nonlinearity="silu" if nonlinear else "identity",
                k_nonlinearity="silu" if nonlinear else "identity",
                v_nonlinearity="identity", o_nonlinearity="identity",
            ).build(8, layer_idx=0, n_layers=1)
        base, model = build(False), build(True)
        self.assertEqual(list(base.state_dict()), list(model.state_dict()))
        for name, value in base.state_dict().items():
            torch.testing.assert_close(value, model.state_dict()[name], atol=0, rtol=0)
        for attr in ("w_q", "w_k", "w_v", "w_out"):
            projection = getattr(model, attr)
            ref = copy.deepcopy(projection)
            x = torch.randn(2, 3, 8, requires_grad=True)
            xr = x.detach().clone().requires_grad_(True)
            a = ref.v(xr)
            expected = ref.u(F.silu(a) if attr in ("w_q", "w_k") else a)
            actual = projection(x)
            torch.testing.assert_close(actual, expected)
            actual.square().sum().backward()
            expected.square().sum().backward()
            torch.testing.assert_close(x.grad, xr.grad)
            for p, pr in zip(projection.parameters(), ref.parameters()):
                torch.testing.assert_close(p.grad, pr.grad)


if __name__ == "__main__":
    unittest.main()
