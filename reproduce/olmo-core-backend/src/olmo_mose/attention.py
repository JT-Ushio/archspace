import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed.tensor.placement_types import Placement

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention import (
    Attention,
    AttentionConfig,
    AttentionType,
    SlidingWindowAttentionConfig,
)
from olmo_core.nn.attention.kv_cache import KVCacheManager
from olmo_core.nn.buffer_cache import BufferCache
from olmo_core.nn.layer_norm import RMSNorm
from olmo_core.nn.rope import RoPEConfig
from olmo_core.nn.transformer import TransformerBlockConfig, TransformerConfig
from olmo_core.nn.transformer.init import InitMethod, init_linear

from .feed_forward import MoSENonlinearity, _apply_mose_nonlinearity


def _validate_rank(rank: int) -> int:
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise OLMoConfigurationError("attention rank must be a positive integer")
    return rank


class NonlinearLowRankProjection(nn.Module):
    """A projection ``U sigma(Vx)`` with no dense bypass."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        *,
        nonlinearity: MoSENonlinearity = MoSENonlinearity.silu,
        rms_norm_learnable_weight: bool = False,
        bias: bool = False,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = _validate_rank(rank)
        self.nonlinearity = MoSENonlinearity(nonlinearity)
        if not isinstance(rms_norm_learnable_weight, bool):
            raise OLMoConfigurationError("rms_norm_learnable_weight must be a boolean")
        self.rms_norm_learnable_weight = rms_norm_learnable_weight

        # Mathematical convention: V projects into the rank-r bottleneck and
        # U projects back to the output space, i.e. U sigma(Vx).
        self.v = nn.Linear(
            in_features,
            rank,
            bias=False,
            dtype=dtype,
            device=init_device,
        )
        self.u = nn.Linear(
            rank,
            out_features,
            bias=bias,
            dtype=dtype,
            device=init_device,
        )
        if self.nonlinearity == MoSENonlinearity.rms_norm:
            self.nonlinear_norm: Optional[RMSNorm] = RMSNorm(
                size=rank,
                eps=1e-5,
                elementwise_affine=rms_norm_learnable_weight,
                bias=False,
                dtype=dtype,
                init_device=init_device,
            )
        else:
            self.nonlinear_norm = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = _apply_mose_nonlinearity(
            self.v(x),
            self.nonlinearity,
            rms_norm=self.nonlinear_norm,
        )
        return self.u(latent)


@dataclass
class LowRankAttentionConfig(AttentionConfig):
    """Attention whose Q, K, V, and output matrices are independently factorized."""

    rank: int = 512
    q_nonlinearity: MoSENonlinearity = MoSENonlinearity.silu
    k_nonlinearity: MoSENonlinearity = MoSENonlinearity.silu
    v_nonlinearity: MoSENonlinearity = MoSENonlinearity.silu
    o_nonlinearity: MoSENonlinearity = MoSENonlinearity.silu
    rms_norm_learnable_weight: bool = False

    def __post_init__(self, registered_name: Optional[str] = None) -> None:
        # ``SequenceMixerConfig`` inherits a registry ``InitVar`` which the
        # generated dataclass initializer passes positionally to this hook.
        del registered_name
        self.name = AttentionType(self.name)
        self.rank = _validate_rank(self.rank)
        self.q_nonlinearity = MoSENonlinearity(self.q_nonlinearity)
        self.k_nonlinearity = MoSENonlinearity(self.k_nonlinearity)
        self.v_nonlinearity = MoSENonlinearity(self.v_nonlinearity)
        self.o_nonlinearity = MoSENonlinearity(self.o_nonlinearity)
        if not isinstance(self.rms_norm_learnable_weight, bool):
            raise OLMoConfigurationError("rms_norm_learnable_weight must be a boolean")
        if self.name != AttentionType.default:
            raise OLMoConfigurationError(
                "low-rank attention requires the default attention implementation"
            )

    def num_params(self, d_model: int) -> int:
        n_heads = self.n_heads
        n_kv_heads = self.n_kv_heads or n_heads
        head_dim = self.head_dim or d_model // n_heads
        q_size = n_heads * head_dim
        kv_size = n_kv_heads * head_dim

        dense_projection_params = (
            d_model * q_size
            + 2 * d_model * kv_size
            + q_size * d_model
        )
        low_rank_projection_params = (
            self.rank * (d_model + q_size)
            + 2 * self.rank * (d_model + kv_size)
            + self.rank * (q_size + d_model)
        )
        if self.rms_norm_learnable_weight:
            nonlinearities = (
                self.q_nonlinearity,
                self.k_nonlinearity,
                self.v_nonlinearity,
                self.o_nonlinearity,
            )
            low_rank_projection_params += self.rank * sum(
                nonlinearity == MoSENonlinearity.rms_norm
                for nonlinearity in nonlinearities
            )
        return (
            super().num_params(d_model)
            - dense_projection_params
            + low_rank_projection_params
        )

    def build(
        self,
        d_model: int,
        *,
        layer_idx: int,
        n_layers: int,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ) -> "LowRankAttention":
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        kwargs.pop("name")
        rank = kwargs.pop("rank")

        sliding_window_config: Optional[SlidingWindowAttentionConfig] = kwargs.pop(
            "sliding_window", None
        )
        if sliding_window_config is not None and sliding_window_config.should_use_swa(
            layer_idx, n_layers
        ):
            kwargs["window_size"] = sliding_window_config.get_window_size(layer_idx, n_layers)
        else:
            rope_config: Optional[RoPEConfig] = kwargs.get("rope")
            if rope_config is not None and rope_config.no_global_rope:
                kwargs["rope"] = None

        kwargs.update(
            rank=rank,
            dtype=kwargs.pop("dtype").as_pt(),
            d_model=d_model,
            init_device=init_device,
            cache=cache,
        )
        try:
            return LowRankAttention(**kwargs)
        except TypeError as e:
            raise OLMoConfigurationError(
                f"invalid options for {self.__class__.__name__}, {e}"
            ) from e


class LowRankAttention(Attention):
    """Default OLMo attention with independent ``U sigma(Vx)`` Q/K/V/O."""

    def __init__(
        self,
        *,
        rank: int = 512,
        q_nonlinearity: MoSENonlinearity = MoSENonlinearity.silu,
        k_nonlinearity: MoSENonlinearity = MoSENonlinearity.silu,
        v_nonlinearity: MoSENonlinearity = MoSENonlinearity.silu,
        o_nonlinearity: MoSENonlinearity = MoSENonlinearity.silu,
        rms_norm_learnable_weight: bool = False,
        **kwargs,
    ) -> None:
        rank = _validate_rank(rank)
        super().__init__(**kwargs)
        self.rank = rank

        def factorize(
            projection: nn.Linear,
            nonlinearity: MoSENonlinearity,
        ) -> NonlinearLowRankProjection:
            return NonlinearLowRankProjection(
                projection.in_features,
                projection.out_features,
                rank,
                nonlinearity=nonlinearity,
                rms_norm_learnable_weight=rms_norm_learnable_weight,
                bias=projection.bias is not None,
                dtype=projection.weight.dtype,
                init_device=str(projection.weight.device),
            )

        self.w_q = factorize(self.w_q, q_nonlinearity)
        self.w_k = factorize(self.w_k, k_nonlinearity)
        self.w_v = factorize(self.w_v, v_nonlinearity)
        self.w_out = factorize(self.w_out, o_nonlinearity)

    def projection_modules(self) -> tuple[nn.Linear, ...]:
        projections = (self.w_q, self.w_k, self.w_v, self.w_out)
        return tuple(
            module
            for projection in projections
            for module in (projection.v, projection.u)
        )

    def init_weights(
        self,
        *,
        init_method: InitMethod,
        d_model: int,
        block_idx: int,
        num_blocks: int,
        std: float = 0.02,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        del block_idx, num_blocks
        if init_method == InitMethod.normalized:
            std = d_model**-0.5
        elif init_method == InitMethod.fan_in:
            std = d_model**-0.5

        v_std = 1.0 / math.sqrt(self.rank)
        for projection in (self.w_q, self.w_k, self.w_v, self.w_out):
            init_linear(projection.v, std=std, generator=generator)
            init_linear(projection.u, std=v_std, generator=generator)
            if projection.nonlinear_norm is not None:
                projection.nonlinear_norm.reset_parameters()

        if self.w_g is not None:
            gate_std = self.w_g.in_features**-0.5 if init_method == InitMethod.fan_in else std
            init_linear(self.w_g, std=gate_std, generator=generator)

    def init_kv_cache_manager(self, batch_size: int, max_seq_len: int) -> None:
        self.backend.assert_supports_kv_cache()
        self.kv_cache_manager = KVCacheManager(
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            num_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim,
            device=self.w_k.v.weight.device,
        )

    def apply_tp(
        self,
        tp_mesh: DeviceMesh,
        input_layout: Optional[Placement] = None,
        output_layout: Optional[Placement] = None,
        use_local_output: bool = True,
        float8_enabled: bool = False,
    ) -> None:
        del tp_mesh, input_layout, output_layout, use_local_output, float8_enabled
        raise NotImplementedError("tensor parallelism is not implemented for low-rank attention")


def patch_low_rank_attention(
    config: TransformerConfig,
    *,
    enabled: bool = False,
    rank: int = 512,
    q_nonlinearity: MoSENonlinearity = MoSENonlinearity.silu,
    k_nonlinearity: MoSENonlinearity = MoSENonlinearity.silu,
    v_nonlinearity: MoSENonlinearity = MoSENonlinearity.silu,
    o_nonlinearity: MoSENonlinearity = MoSENonlinearity.silu,
    rms_norm_learnable_weight: bool = False,
) -> TransformerConfig:
    """Optionally replace every default attention Q/K/V/O with rank-``rank`` factors."""
    if not isinstance(enabled, bool):
        raise OLMoConfigurationError("enabled must be a boolean")
    patched = config.copy()
    if not enabled:
        return patched
    rank = _validate_rank(rank)
    nonlinearities = {
        "q_nonlinearity": MoSENonlinearity(q_nonlinearity),
        "k_nonlinearity": MoSENonlinearity(k_nonlinearity),
        "v_nonlinearity": MoSENonlinearity(v_nonlinearity),
        "o_nonlinearity": MoSENonlinearity(o_nonlinearity),
    }
    if not isinstance(rms_norm_learnable_weight, bool):
        raise OLMoConfigurationError("rms_norm_learnable_weight must be a boolean")

    def patch_block(block: TransformerBlockConfig) -> None:
        sequence_mixer = block.sequence_mixer
        if type(sequence_mixer) not in (AttentionConfig, LowRankAttentionConfig):
            raise OLMoConfigurationError(
                "patch_low_rank_attention requires the default AttentionConfig; "
                f"got {type(sequence_mixer).__name__}"
            )
        if sequence_mixer.name != AttentionType.default:
            raise OLMoConfigurationError(
                "patch_low_rank_attention requires the default attention implementation"
            )
        kwargs = sequence_mixer.as_dict(
            exclude_none=False,
            exclude=(
                "rank",
                "q_nonlinearity",
                "k_nonlinearity",
                "v_nonlinearity",
                "o_nonlinearity",
                "rms_norm_learnable_weight",
            ),
            recurse=False,
        )
        block.sequence_mixer = LowRankAttentionConfig(
            **kwargs,
            rank=rank,
            rms_norm_learnable_weight=rms_norm_learnable_weight,
            **nonlinearities,
        )

    if isinstance(patched.block, dict):
        for block in patched.block.values():
            patch_block(block)
    else:
        patch_block(patched.block)
    if patched.block_overrides is not None:
        for block in patched.block_overrides.values():
            patch_block(block)

    return patched
