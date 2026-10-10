"""Dense Q/O with one shared linear K/V bottleneck."""
from dataclasses import dataclass
import math
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention import Attention, AttentionConfig, AttentionType
from olmo_core.nn.transformer.init import InitMethod, init_linear
from .attention import LowRankAttention, NonlinearLowRankProjection
from .feed_forward import MoSENonlinearity

@dataclass
class DenseSharedKVConfig(AttentionConfig):
    rank: int = 1360

    def __post_init__(self, registered_name=None):
        if self.name != AttentionType.default or self.rank <= 0:
            raise OLMoConfigurationError('requires default attention and positive shared rank')

    def num_params(self, d_model):
        kv = (self.n_kv_heads or self.n_heads)*(self.head_dim or d_model//self.n_heads)
        return super().num_params(d_model)-2*d_model*kv+self.rank*(d_model+2*kv)

    def build(self, d_model, *, layer_idx, n_layers, init_device='cpu', cache=None):
        kwargs=self.as_dict(exclude_none=True,recurse=False)
        kwargs.pop('name')
        sliding=kwargs.pop('sliding_window',None)
        if sliding is not None and sliding.should_use_swa(layer_idx,n_layers):
            kwargs['window_size']=sliding.get_window_size(layer_idx,n_layers)
        elif kwargs.get('rope') is not None and kwargs['rope'].no_global_rope:
            kwargs['rope']=None
        kwargs.update(dtype=kwargs.pop('dtype').as_pt(),d_model=d_model,init_device=init_device,cache=cache)
        return DenseSharedKVAttention(**kwargs)

class DenseSharedKVAttention(Attention):
    forward=LowRankAttention.forward

    def __init__(self, *, rank, **kwargs):
        super().__init__(**kwargs)
        self.rank=rank
        for name in ('w_k','w_v'):
            p=getattr(self,name)
            setattr(self,name,NonlinearLowRankProjection(p.in_features,p.out_features,rank,
                nonlinearity=MoSENonlinearity.identity,bias=p.bias is not None,
                dtype=p.weight.dtype,init_device=str(p.weight.device)))
        self.w_v.v=self.w_k.v

    def _project_qkv(self,x):
        z=self.w_k.v(x)
        return self.w_q(x),self.w_k.u(z),self.w_v.u(z)

    def init_weights(self, *, init_method,d_model,block_idx,num_blocks,std=0.02,generator=None):
        if init_method != InitMethod.normal:
            raise OLMoConfigurationError('dense shared KV requires normal initialization')
        for p in (self.w_q,self.w_k.v,self.w_out):
            init_linear(p,std=std,generator=generator)
        for p in (self.w_k.u,self.w_v.u):
            init_linear(p,std=1/math.sqrt(self.rank),generator=generator)
        if self.w_g is not None:init_linear(self.w_g,std=std,generator=generator)
