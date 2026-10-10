"""Replace just Q or K with an exactly dense-parameter-matched SiLU projection."""

from dataclasses import dataclass
import math

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention import Attention, AttentionConfig, AttentionType
from olmo_core.nn.transformer.init import InitMethod, init_linear

from .attention import NonlinearLowRankProjection


def matched_rank(in_features: int, out_features: int) -> int:
    rank, remainder = divmod(in_features * out_features, in_features + out_features)
    if remainder or rank <= 0:
        raise OLMoConfigurationError("selected projection has no integer dense-matched rank")
    return rank


@dataclass
class DenseSingleSiLUAttentionConfig(AttentionConfig):
    target: str = "q"

    def __post_init__(self, registered_name=None):
        del registered_name
        if self.target not in ("q", "k") or self.name != AttentionType.default:
            raise OLMoConfigurationError("requires default attention and target q or k")

    def num_params(self, d_model: int) -> int:
        heads = self.n_heads if self.target == "q" else (self.n_kv_heads or self.n_heads)
        matched_rank(d_model, heads * (self.head_dim or d_model // self.n_heads))
        return super().num_params(d_model)

    def build(self, d_model, *, layer_idx, n_layers, init_device="cpu", cache=None):
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        kwargs.pop("name")
        sliding = kwargs.pop("sliding_window", None)
        if sliding is not None and sliding.should_use_swa(layer_idx, n_layers):
            kwargs["window_size"] = sliding.get_window_size(layer_idx, n_layers)
        elif kwargs.get("rope") is not None and kwargs["rope"].no_global_rope:
            kwargs["rope"] = None
        kwargs.update(dtype=kwargs.pop("dtype").as_pt(), d_model=d_model,
                      init_device=init_device, cache=cache)
        return DenseSingleSiLUAttention(**kwargs)


class DenseSingleSiLUAttention(Attention):
    """Keep the base attention forward, norms, RoPE, and other dense projections."""

    def __init__(self, *, target, **kwargs):
        if target not in ("q", "k"):
            raise OLMoConfigurationError("target must be q or k")
        super().__init__(**kwargs)
        self.target = target
        name = f"w_{target}"
        dense = getattr(self, name)
        rank = matched_rank(dense.in_features, dense.out_features)
        replacement = NonlinearLowRankProjection(
            dense.in_features, dense.out_features, rank,
            bias=dense.bias is not None, dtype=dense.weight.dtype,
            init_device=str(dense.weight.device),
        )
        assert sum(p.numel() for p in dense.parameters()) == sum(
            p.numel() for p in replacement.parameters())
        setattr(self, name, replacement)

    def init_weights(self, *, init_method, d_model, block_idx, num_blocks,
                     std=0.02, generator=None):
        if init_method != InitMethod.normal:
            raise OLMoConfigurationError("dense single-SiLU experiments require normal initialization")
        # Unchanged dense projections retain the baseline's normal initialization.
        # The changed projection follows the existing MoSE A/B initialization.
        for name in ("w_q", "w_k", "w_v"):
            projection = getattr(self, name)
            if isinstance(projection, NonlinearLowRankProjection):
                init_linear(projection.v, std=std, generator=generator)
                init_linear(projection.u, std=1 / math.sqrt(projection.rank), generator=generator)
            else:
                init_linear(projection, std=std, generator=generator)
        if self.w_g is not None:
            init_linear(self.w_g, std=std, generator=generator)
        init_linear(self.w_out, std=std, generator=generator)
