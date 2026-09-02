import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed.tensor.placement_types import Placement

from olmo_core.config import StrEnum
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention import (
    Attention,
    AttentionConfig,
    AttentionType,
    GateGranularity,
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


class LowRankAttentionSharingScope(StrEnum):
    """Low-dimensional input projections shared by low-rank Q/K/V."""

    none = "none"
    qk = "qk"
    kv = "kv"
    qkv = "qkv"


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
    share_scope: LowRankAttentionSharingScope = LowRankAttentionSharingScope.none

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
        self.share_scope = LowRankAttentionSharingScope(self.share_scope)
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
        input_projection_count = {
            LowRankAttentionSharingScope.none: 4,
            LowRankAttentionSharingScope.qk: 3,
            LowRankAttentionSharingScope.kv: 3,
            LowRankAttentionSharingScope.qkv: 2,
        }[self.share_scope]
        low_rank_projection_params = self.rank * (
            input_projection_count * d_model + 2 * q_size + 2 * kv_size
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
        share_scope: LowRankAttentionSharingScope = LowRankAttentionSharingScope.none,
        **kwargs,
    ) -> None:
        rank = _validate_rank(rank)
        super().__init__(**kwargs)
        self.rank = rank
        self.rms_norm_learnable_weight = rms_norm_learnable_weight
        self.share_scope = LowRankAttentionSharingScope(share_scope)

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

        if self.share_scope == LowRankAttentionSharingScope.qk:
            self.w_k.v = self.w_q.v
        elif self.share_scope == LowRankAttentionSharingScope.kv:
            self.w_v.v = self.w_k.v
        elif self.share_scope == LowRankAttentionSharingScope.qkv:
            self.w_k.v = self.w_q.v
            self.w_v.v = self.w_q.v

    def _can_share_activated_latent(
        self,
        projections: Tuple[NonlinearLowRankProjection, ...],
    ) -> bool:
        """Whether a shared input latent can also be shared after activation."""
        nonlinearities = {projection.nonlinearity for projection in projections}
        if len(nonlinearities) != 1:
            return False

        # With independent learnable RMSNorm weights, each branch has a different
        # activation and therefore cannot reuse the post-normalization latent.
        nonlinearity = projections[0].nonlinearity
        if nonlinearity == MoSENonlinearity.rms_norm and self.rms_norm_learnable_weight:
            first_norm = projections[0].nonlinear_norm
            return all(projection.nonlinear_norm is first_norm for projection in projections)
        return True

    def _project_shared_group(
        self,
        x: torch.Tensor,
        projections: Tuple[NonlinearLowRankProjection, ...],
    ) -> Tuple[torch.Tensor, ...]:
        """Project a Q/K/V sharing group, evaluating its shared latent once."""
        first = projections[0]
        raw_latent = first.v(x)
        if self._can_share_activated_latent(projections):
            activated_latent = _apply_mose_nonlinearity(
                raw_latent,
                first.nonlinearity,
                rms_norm=first.nonlinear_norm,
            )
            return tuple(projection.u(activated_latent) for projection in projections)

        # Sharing still saves the input projection when branch nonlinearities
        # differ; each branch must then apply its own activation separately.
        return tuple(
            projection.u(
                _apply_mose_nonlinearity(
                    raw_latent,
                    projection.nonlinearity,
                    rms_norm=projection.nonlinear_norm,
                )
            )
            for projection in projections
        )

    def _project_qkv(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Project Q/K/V while reusing shared nonlinear latents when possible."""
        if self.share_scope == LowRankAttentionSharingScope.none:
            return self.w_q(x), self.w_k(x), self.w_v(x)
        if self.share_scope == LowRankAttentionSharingScope.qk:
            q, k = self._project_shared_group(x, (self.w_q, self.w_k))
            return q, k, self.w_v(x)
        if self.share_scope == LowRankAttentionSharingScope.kv:
            k, v = self._project_shared_group(x, (self.w_k, self.w_v))
            return self.w_q(x), k, v
        if self.share_scope == LowRankAttentionSharingScope.qkv:
            return self._project_shared_group(x, (self.w_q, self.w_k, self.w_v))
        raise OLMoConfigurationError(f"unsupported attention sharing scope: {self.share_scope}")

    def forward(
        self,
        x: torch.Tensor,
        cu_doc_lens: Optional[torch.Tensor] = None,
        cu_doc_lens_q: Optional[torch.Tensor] = None,
        cu_doc_lens_k: Optional[torch.Tensor] = None,
        max_doc_len: Optional[int] = None,
        max_doc_len_q: Optional[int] = None,
        max_doc_len_k: Optional[int] = None,
        local_k_slice: Optional[slice] = None,
        pos_sin: Optional[torch.Tensor] = None,
        pos_cos: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
        cache_leftpad: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply attention with shared Q/K/V low-rank computation when enabled."""
        B, T, _ = x.shape

        # Unlike the base Attention.forward(), this computes a shared Q/K/V
        # bottleneck and activation once for the selected sharing scope. There
        # is intentionally no linear/r1 branch here: all factors are nonlinear
        # low-rank projections.
        q, k, v = self._project_qkv(x)

        if self.clip_qkv is not None:
            q.clamp_(min=-self.clip_qkv, max=self.clip_qkv)
            k.clamp_(min=-self.clip_qkv, max=self.clip_qkv)
            v.clamp_(min=-self.clip_qkv, max=self.clip_qkv)

        if not self.use_head_qk_norm:
            if self.q_norm is not None:
                q = self.q_norm(q)
            if self.k_norm is not None:
                k = self.k_norm(k)

        q = q.view(B, T, -1, self.head_dim)
        k = k.view(B, T, -1, self.head_dim)
        v = v.view(B, T, -1, self.head_dim)

        if self.use_head_qk_norm:
            if self.q_norm is not None:
                q = self.q_norm(q)
            if self.k_norm is not None:
                k = self.k_norm(k)

        if self.rope is not None:
            if self.cp_enabled and pos_sin is None and pos_cos is None and freqs_cis is None:
                raise RuntimeError(
                    "RoPE buffers must be passed through to attention after being properly "
                    "sharded by the context parallel load balancer"
                )

            start_pos = self.kv_cache_manager.current_position() if self.kv_cache_manager else None
            q, k = self._apply_rope(q, k, start_pos, pos_sin, pos_cos, freqs_cis, cu_doc_lens)

        att = self.sdpa(
            q,
            k,
            v,
            cu_doc_lens=cu_doc_lens,
            cu_doc_lens_q=cu_doc_lens_q,
            cu_doc_lens_k=cu_doc_lens_k,
            max_doc_len=max_doc_len,
            max_doc_len_q=max_doc_len_q,
            max_doc_len_k=max_doc_len_k,
            local_k_slice=local_k_slice,
            cache_leftpad=cache_leftpad,
        )

        if self.gate is not None:
            assert self.w_g is not None
            g = self.w_g(x)
            if self.gate.full_precision:
                g = g.float()
            gate_values = torch.sigmoid(g).to(att.dtype)
            if self.gate.granularity == GateGranularity.headwise:
                att = att * gate_values.unsqueeze(-1)
            elif self.gate.granularity == GateGranularity.elementwise:
                att = att.view(B, T, -1) * gate_values

        att = att.view(B, T, -1)
        return self.w_out(att)

    def projection_modules(self) -> tuple[nn.Linear, ...]:
        modules = []
        seen = set()
        for projection in (self.w_q, self.w_k, self.w_v, self.w_out):
            for module in (projection.v, projection.u):
                if id(module) not in seen:
                    modules.append(module)
                    seen.add(id(module))
        return tuple(modules)

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
        initialized = set()
        for projection in (self.w_q, self.w_k, self.w_v, self.w_out):
            if id(projection.v) not in initialized:
                init_linear(projection.v, std=std, generator=generator)
                initialized.add(id(projection.v))
            if id(projection.u) not in initialized:
                init_linear(projection.u, std=v_std, generator=generator)
                initialized.add(id(projection.u))
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
    share_scope: LowRankAttentionSharingScope = LowRankAttentionSharingScope.none,
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
                "share_scope",
            ),
            recurse=False,
        )
        block.sequence_mixer = LowRankAttentionConfig(
            **kwargs,
            rank=rank,
            rms_norm_learnable_weight=rms_norm_learnable_weight,
            share_scope=LowRankAttentionSharingScope(share_scope),
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
