import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
from olmo_core.nn.attention import AttentionConfig, AttentionBackendName
from olmo_core.nn.transformer.init import InitMethod
from olmo_mose.attention import NonlinearLowRankProjection
from olmo_mose.dense_single_attention import DenseSingleSiLUAttentionConfig, matched_rank


def check_projection(target):
    kwargs = dict(n_heads=2, head_dim=4, bias=False, backend=AttentionBackendName.torch)
    dense = AttentionConfig(**kwargs)
    config = DenseSingleSiLUAttentionConfig(**kwargs, target=target)
    model = config.build(8, layer_idx=0, n_layers=1)
    assert config.num_params(8) == dense.num_params(8)
    assert sum(p.numel() for p in model.parameters()) == dense.num_params(8)
    assert matched_rank(2048, 2048) == 1024
    model.init_weights(init_method=InitMethod.normal, d_model=8, block_idx=0, num_blocks=1)
    for name in ("q", "k", "v", "out"):
        projection = getattr(model, f"w_{name}")
        assert isinstance(projection, NonlinearLowRankProjection if name == target else nn.Linear)
    x = torch.randn(2, 5, 8, requires_grad=True)
    p = getattr(model, f"w_{target}")
    torch.testing.assert_close(p(x), F.linear(F.silu(F.linear(x, p.v.weight)), p.u.weight))
    y = model(x)
    assert y.shape == x.shape
    y.square().sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


class DenseSingleAttentionTests(unittest.TestCase):
    def test_q(self):
        check_projection("q")

    def test_k(self):
        check_projection("k")

    def test_full_model_config_keeps_dense_ffn_and_budget(self):
        import importlib.util
        from pathlib import Path
        from olmo_core.data import TokenizerConfig
        from olmo_core.nn.transformer import TransformerConfig
        path = Path(__file__).resolve().parents[1] / "cfps/BS256_Seq8K_30B_WSD_FP8/Dense_Single_SiLU/_common.py"
        spec = importlib.util.spec_from_file_location("dense_single_test_common", path)
        common = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(common)
        tok = TokenizerConfig.dolma2()
        dense = TransformerConfig.olmo3_1B(vocab_size=tok.padded_vocab_size(), attn_backend=AttentionBackendName.flash_3)
        for target in ("q", "k"):
            model = common.build_model(tok, target)
            self.assertEqual(model.num_params, dense.num_params)
            self.assertEqual(model.block.feed_forward.as_dict(), dense.block.feed_forward.as_dict())
            actual = model.block.sequence_mixer.as_dict()
            self.assertEqual(actual.pop("target"), target)
            self.assertEqual(actual, dense.block.sequence_mixer.as_dict())
            print(target, "full-model parameter count:", model.num_params)


if __name__ == "__main__":
    unittest.main()
