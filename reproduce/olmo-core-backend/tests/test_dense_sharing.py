import unittest,importlib.util
from pathlib import Path
import torch
import torch.nn.functional as F
from olmo_core.nn.attention import AttentionConfig,AttentionBackendName
from olmo_core.nn.transformer.init import InitMethod
from olmo_mose.dense_shared_attention import DenseSharedKVConfig
from olmo_mose import MoSESwiGLUConfig,MoSENonlinearity,SwiGLUChannelControl,SwiGLUChannelControlScope

class Tests(unittest.TestCase):
 def test_kv(self):
  cfg=DenseSharedKVConfig(n_heads=2,head_dim=4,bias=False,rank=4,backend=AttentionBackendName.torch)
  m=cfg.build(8,layer_idx=0,n_layers=1);m.init_weights(init_method=InitMethod.normal,d_model=8,block_idx=0,num_blocks=1)
  dense=AttentionConfig(n_heads=2,head_dim=4,bias=False,backend=AttentionBackendName.torch).build(8,layer_idx=0,n_layers=1)
  with torch.no_grad():
   dense.w_q.weight.copy_(m.w_q.weight);dense.w_out.weight.copy_(m.w_out.weight)
   dense.w_k.weight.copy_(m.w_k.u.weight@m.w_k.v.weight);dense.w_v.weight.copy_(m.w_v.u.weight@m.w_k.v.weight)
  x=torch.randn(2,5,8,requires_grad=True)
  torch.testing.assert_close(m(x),dense(x),atol=1e-6,rtol=1e-5)
  self.assertIs(m.w_k.v,m.w_v.v)
  self.assertEqual(sum(p.numel() for p in m.parameters()),cfg.num_params(8))
  m(x).square().sum().backward()
  self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))
 def test_ug(self):
  cfg=MoSESwiGLUConfig(hidden_size=32,bias=False,r1=0,r2=6,down_r1=0,down_r2=0,control=SwiGLUChannelControl.standard,control_scope=SwiGLUChannelControlScope.none,gate_nonlinearity=MoSENonlinearity.identity,up_nonlinearity=MoSENonlinearity.identity,share_gate_up_subspace=True)
  m=cfg.build(8);InitMethod.normal.init_feed_forward(m,d_model=8,block_idx=0,num_blocks=1)
  x=torch.randn(2,5,8,requires_grad=True);z=m.nonlinear_u(x)
  torch.testing.assert_close(m(x),m.w_down(F.silu(m.gate_nonlinear_v(z))*m.up_nonlinear_v(z)))
  self.assertEqual(sum(p.numel() for p in m.parameters()),cfg.num_params(8))
  m(x).square().sum().backward();self.assertTrue(all(p.grad is not None for p in m.parameters()))
 def test_configs(self):
  from olmo_core.data import TokenizerConfig
  from olmo_core.nn.transformer import TransformerConfig
  p=Path(__file__).resolve().parents[1]/'cfps/BS256_Seq8K_30B_WSD_FP8/Dense_Sharing/_common.py'
  spec=importlib.util.spec_from_file_location('sharing_test',p);c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
  tok=TokenizerConfig.dolma2();dense=TransformerConfig.olmo3_1B(vocab_size=tok.padded_vocab_size(),attn_backend=AttentionBackendName.flash_3)
  for name,count in [('s1',1481246720),('s2',1484392448),('s3',1480722432)]:
   cfg=c.build_model(tok,name);self.assertEqual(cfg.num_params,count)
   ff=cfg.block.feed_forward.build(cfg.d_model,init_device='meta')
   att=cfg.block.sequence_mixer.build(cfg.d_model,layer_idx=0,n_layers=16,init_device='meta')
   self.assertEqual(sum(p.numel() for p in ff.parameters()),cfg.block.feed_forward.num_params(cfg.d_model))
   self.assertEqual(sum(p.numel() for p in att.parameters()),cfg.block.sequence_mixer.num_params(cfg.d_model))
   if name=='s1':self.assertEqual(cfg.block.sequence_mixer.as_dict(),dense.block.sequence_mixer.as_dict())
   if name=='s2':self.assertEqual(cfg.block.feed_forward.as_dict(),dense.block.feed_forward.as_dict())
   print(name,count)
if __name__=='__main__':unittest.main()
