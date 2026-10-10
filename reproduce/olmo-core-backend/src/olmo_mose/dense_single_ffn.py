"""One nonlinear FFN factorization; every other dense projection is retained."""
from dataclasses import dataclass
import functools
import math
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.feed_forward import FeedForward, FeedForwardConfig, FeedForwardType
from olmo_core.nn.transformer.init import InitMethod, init_linear
from .attention import NonlinearLowRankProjection

@dataclass
class DenseSingleSiLUFFNConfig(FeedForwardConfig):
    target: str = 'up'
    rank: int = 1632

    def __post_init__(self, registered_name=None):
        if self.target not in ('up', 'gate') or self.name != FeedForwardType.default:
            raise OLMoConfigurationError('requires default FFN with target up or gate')
        if self.rank <= 0:
            raise OLMoConfigurationError('rank must be positive')

    def num_params(self, d_model):
        return super().num_params(d_model) - d_model*self.hidden_size + self.rank*(d_model+self.hidden_size)

    def build(self, d_model, *, dtype=None, init_device='cpu'):
        kwargs = self.as_dict(exclude_none=True)
        kwargs.pop('name')
        kwargs.update(d_model=d_model, init_device=init_device)
        if self.dtype is not None:
            kwargs['dtype'] = self.dtype.as_pt()
        elif dtype is not None:
            kwargs['dtype'] = dtype
        return DenseSingleSiLUFFN(**kwargs)

class DenseSingleSiLUFFN(FeedForward):
    def __init__(self, *, target, rank, **kwargs):
        super().__init__(**kwargs)
        self.target, self.rank = target, rank
        name = 'w3' if target == 'up' else 'w1'
        dense = getattr(self, name)
        setattr(self, name, NonlinearLowRankProjection(
            dense.in_features, dense.out_features, rank,
            bias=dense.bias is not None, dtype=dense.weight.dtype,
            init_device=str(dense.weight.device)))


def install_ffn_init_hook():
    current = InitMethod.init_feed_forward
    if getattr(current, '_dense_single_ffn_hook', False):
        return
    @functools.wraps(current)
    def initialize(self, module, *, d_model, block_idx, num_blocks, std=0.02, generator=None):
        if not isinstance(module, DenseSingleSiLUFFN):
            return current(self, module, d_model=d_model, block_idx=block_idx,
                           num_blocks=num_blocks, std=std, generator=generator)
        if self != InitMethod.normal:
            raise OLMoConfigurationError('single FFN SiLU requires normal initialization')
        for name in ('w1', 'w3', 'w2'):
            p = getattr(module, name)
            if isinstance(p, NonlinearLowRankProjection):
                init_linear(p.v, std=std, generator=generator)
                init_linear(p.u, std=1/math.sqrt(p.rank), generator=generator)
            else:
                init_linear(p, std=std, generator=generator)
    initialize._dense_single_ffn_hook = True
    InitMethod.init_feed_forward = initialize

install_ffn_init_hook()
