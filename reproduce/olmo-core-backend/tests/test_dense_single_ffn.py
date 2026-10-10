import unittest
import torch
import torch.nn.functional as F
from olmo_core.nn.feed_forward import FeedForwardConfig
from olmo_core.nn.transformer.init import InitMethod
from olmo_mose.dense_single_ffn import DenseSingleSiLUFFNConfig

class SingleFFNTests(unittest.TestCase):
    def test_formula_budget_and_backward(self):
        for target in ('up','gate'):
            cfg=DenseSingleSiLUFFNConfig(hidden_size=32,bias=False,target=target,rank=6)
            m=cfg.build(8)
            InitMethod.normal.init_feed_forward(m,d_model=8,block_idx=0,num_blocks=1)
            self.assertEqual(sum(p.numel() for p in m.parameters()),cfg.num_params(8))
            x=torch.randn(2,3,8,requires_grad=True)
            selected=m.w3 if target=='up' else m.w1
            z=F.linear(F.silu(F.linear(x,selected.v.weight)),selected.u.weight)
            expected=m.w2(F.silu(m.w1(x))*z) if target=='up' else m.w2(F.silu(z)*m.w3(x))
            y=m(x);torch.testing.assert_close(y,expected)
            y.square().sum().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))
    def test_dense_initialization_unchanged(self):
        m=FeedForwardConfig(hidden_size=32,bias=False).build(8)
        InitMethod.normal.init_feed_forward(m,d_model=8,block_idx=0,num_blocks=1)
        self.assertTrue(torch.isfinite(m(torch.randn(2,8))).all())
    def test_full_config(self):
        import importlib.util
        from pathlib import Path
        from olmo_core.data import TokenizerConfig
        from olmo_core.nn.transformer import TransformerConfig
        from olmo_core.nn.attention import AttentionBackendName
        p=Path(__file__).resolve().parents[1]/'cfps/BS256_Seq8K_30B_WSD_FP8/Dense_Single_SiLU/_ffn_common.py'
        spec=importlib.util.spec_from_file_location('ffn_common_test',p)
        common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
        tok=TokenizerConfig.dolma2()
        dense=TransformerConfig.olmo3_1B(vocab_size=tok.padded_vocab_size(),attn_backend=AttentionBackendName.flash_3)
        for target in ('up','gate'):
            cfg=common.build_model(tok,target)
            self.assertEqual(cfg.block.sequence_mixer.as_dict(),dense.block.sequence_mixer.as_dict())
            self.assertEqual(cfg.num_params,dense.num_params-1048576)
            meta=cfg.block.feed_forward.build(cfg.d_model,init_device='meta')
            self.assertEqual(sum(p.numel() for p in meta.parameters()),cfg.block.feed_forward.num_params(cfg.d_model))
            print(target,cfg.num_params,'dense',dense.num_params)
if __name__=='__main__':unittest.main()
