import importlib.util
import unittest
from pathlib import Path

from olmo_core.data import TokenizerConfig


class ReallocationTests(unittest.TestCase):
    def test_actual_counts_and_fixed_attention(self):
        path = Path(__file__).resolve().parents[1] / "cfps/BS256_Seq8K_30B_WSD_FP8/FFN_Rank_Reallocation/_common.py"
        spec = importlib.util.spec_from_file_location("ffn_reallocation_test_config", path)
        common = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(common)
        attention = None
        for target in common.VARIANTS:
            cfg = common.build_model(TokenizerConfig.dolma2(), target)
            self.assertEqual(cfg.num_params, 797_050_880)
            module = cfg.block.feed_forward.build(cfg.d_model, init_device="meta")
            self.assertEqual(sum(p.numel() for p in module.parameters()), 15_728_640)
            if attention is None:
                attention = cfg.block.sequence_mixer.as_dict()
            self.assertEqual(cfg.block.sequence_mixer.as_dict(), attention)
            common.validate_model(cfg, target)
            cfg.block.feed_forward.r2 += 16
            with self.assertRaises(ValueError):
                common.validate_model(cfg, target)


if __name__ == "__main__":
    unittest.main()
