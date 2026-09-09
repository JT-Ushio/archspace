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

from .feed_forward import MoSENonlinearity, _apply_mose_nonlinearity, _uses_rms_norm


def _validate_rank(rank: int) -> int:
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise OLMoConfigurationError("attention rank must be a positive integer")
    return rank


def _resolve_attention_ranks(
    rank: int,
    q_rank: Optional[int],
    k_rank: Optional[int],
    v_rank: Optional[int],
    o_rank: Optional[int],
) -> Tuple[int, int, int, int]:
    """Resolve per-projection ranks, falling back to legacy ``rank``."""
    fallback = _validate_rank(rank)
    resolved_q = fallback if q_rank is None else _validate_rank(q_rank)
    resolved_k = fallback if k_rank is None else _validate_rank(k_rank)
    resolved_v = fallback if v_rank is None else _validate_rank(v_rank)
    resolved_o = fallback if o_rank is None else _validate_rank(o_rank)
    return resolved_q, resolved_k, resolved_v, resolved_o


def _validate_sharing_ranks(
    share_scope: "LowRankAttentionSharingScope",
    q_rank: int,
    k_rank: int,
    v_rank: int,
) -> None:
    """Ensure projections that share an input bottleneck have equal rank."""
    if share_scope == LowRankAttentionSharingScope.qk and q_rank != k_rank:
        raise OLMoConfigurationError("qk attention sharing requires q_rank == k_rank")
    if share_scope == LowRankAttentionSharingScope.kv and k_rank != v_rank:
        raise OLMoConfigurationError("kv attention sharing requires k_rank == v_rank")
    if share_scope == LowRankAttentionSharingScope.qkv and (
        q_rank != k_rank or k_rank != v_rank
    ):
        raise OLMoConfigurationError(
            "qkv attention sharing requires q_rank == k_rank == v_rank"
        )


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
        if _uses_rms_norm(self.nonlinearity):
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

    # ``rank`` is kept as the backwards-compatible default. The per-projection
    # ranks below may be supplied independently; an omitted rank falls back to
    # ``rank``. Sharing scopes validate equality only where modules are shared.
    rank: int = 512
    q_rank: Optional[int] = None
    k_rank: Optional[int] = None
    v_rank: Optional[int] = None
    o_rank: Optional[int] = None
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
        if self.q_rank is not None:
            self.q_rank = _validate_rank(self.q_rank)
        if self.k_rank is not None:
            self.k_rank = _validate_rank(self.k_rank)
        if self.v_rank is not None:
            self.v_rank = _validate_rank(self.v_rank)
        if self.o_rank is not None:
            self.o_rank = _validate_rank(self.o_rank)
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
        q_rank, k_rank, v_rank, _ = _resolve_attention_ranks(
            self.rank,
            self.q_rank,
            self.k_rank,
            self.v_rank,
            self.o_rank,
        )
        _validate_sharing_ranks(
            self.share_scope,
            q_rank,
            k_rank,
            v_rank,
        )

    def num_params(self, d_model: int) -> int:
        n_heads = self.n_heads
        n_kv_heads = self.n_kv_heads or n_heads
        head_dim = self.head_dim or d_model // n_heads
        q_size = n_heads * head_dim
        kv_size = n_kv_heads * head_dim
        q_rank, k_rank, v_rank, o_rank = _resolve_attention_ranks(
            self.rank,
            self.q_rank,
            self.k_rank,
            self.v_rank,
            self.o_rank,
        )

        dense_projection_params = (
            d_model * q_size
            + 2 * d_model * kv_size
            + q_size * d_model
        )
        input_projection_params = {
            LowRankAttentionSharingScope.none: d_model
            * (q_rank + k_rank + v_rank + o_rank),
            LowRankAttentionSharingScope.qk: d_model
            * (q_rank + v_rank + o_rank),
            LowRankAttentionSharingScope.kv: d_model
            * (q_rank + k_rank + o_rank),
            LowRankAttentionSharingScope.qkv: d_model * (q_rank + o_rank),
        }[self.share_scope]
        low_rank_projection_params = input_projection_params + (
            q_rank * q_size
            + k_rank * kv_size
            + v_rank * kv_size
            + o_rank * d_model
        )
        if self.rms_norm_learnable_weight:
            low_rank_projection_params += sum(
                projection_rank
                for projection_rank, nonlinearity in (
                    (q_rank, self.q_nonlinearity),
                    (k_rank, self.k_nonlinearity),
                    (v_rank, self.v_nonlinearity),
                    (o_rank, self.o_nonlinearity),
                )
                if _uses_rms_norm(nonlinearity)
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
        q_rank = kwargs.pop("q_rank", None)
        k_rank = kwargs.pop("k_rank", None)
        v_rank = kwargs.pop("v_rank", None)
        o_rank = kwargs.pop("o_rank", None)

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
            q_rank=q_rank,
            k_rank=k_rank,
            v_rank=v_rank,
            o_rank=o_rank,
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
        q_rank: Optional[int] = None,
        k_rank: Optional[int] = None,
        v_rank: Optional[int] = None,
        o_rank: Optional[int] = None,
        q_nonlinearity: MoSENonlinearity = MoSENonlinearity.silu,
        k_nonlinearity: MoSENonlinearity = MoSENonlinearity.silu,
        v_nonlinearity: MoSENonlinearity = MoSENonlinearity.silu,
        o_nonlinearity: MoSENonlinearity = MoSENonlinearity.silu,
        rms_norm_learnable_weight: bool = False,
        share_scope: LowRankAttentionSharingScope = LowRankAttentionSharingScope.none,
        **kwargs,
    ) -> None:
        rank = _validate_rank(rank)
        q_rank, k_rank, v_rank, o_rank = _resolve_attention_ranks(
            rank,
            q_rank,
            k_rank,
            v_rank,
            o_rank,
        )
        super().__init__(**kwargs)
        self.rank = rank
        self.q_rank = q_rank
        self.k_rank = k_rank
        self.v_rank = v_rank
        self.o_rank = o_rank
        self.rms_norm_learnable_weight = rms_norm_learnable_weight
        self.share_scope = LowRankAttentionSharingScope(share_scope)
        _validate_sharing_ranks(self.share_scope, q_rank, k_rank, v_rank)

        def factorize(
            projection: nn.Linear,
            nonlinearity: MoSENonlinearity,
            projection_rank: int,
        ) -> NonlinearLowRankProjection:
            return NonlinearLowRankProjection(
                projection.in_features,
                projection.out_features,
                projection_rank,
                nonlinearity=nonlinearity,
                rms_norm_learnable_weight=rms_norm_learnable_weight,
                bias=projection.bias is not None,
                dtype=projection.weight.dtype,
                init_device=str(projection.weight.device),
            )

        self.w_q = factorize(self.w_q, q_nonlinearity, q_rank)
        self.w_k = factorize(self.w_k, k_nonlinearity, k_rank)
        self.w_v = factorize(self.w_v, v_nonlinearity, v_rank)
        self.w_out = factorize(self.w_out, o_nonlinearity, o_rank)

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
        if _uses_rms_norm(nonlinearity) and self.rms_norm_learnable_weight:
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

        initialized = set()
        for projection in (self.w_q, self.w_k, self.w_v, self.w_out):
            if id(projection.v) not in initialized:
                init_linear(projection.v, std=std, generator=generator)
                initialized.add(id(projection.v))
            if id(projection.u) not in initialized:
                init_linear(
                    projection.u,
                    std=1.0 / math.sqrt(projection.rank),
                    generator=generator,
                )
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
    q_rank: Optional[int] = None,
    k_rank: Optional[int] = None,
    v_rank: Optional[int] = None,
    o_rank: Optional[int] = None,
    q_nonlinearity: MoSENonlinearity = MoSENonlinearity.silu,
    k_nonlinearity: MoSENonlinearity = MoSENonlinearity.silu,
    v_nonlinearity: MoSENonlinearity = MoSENonlinearity.silu,
    o_nonlinearity: MoSENonlinearity = MoSENonlinearity.silu,
    rms_norm_learnable_weight: bool = False,
    share_scope: LowRankAttentionSharingScope = LowRankAttentionSharingScope.none,
) -> TransformerConfig:
    """Optionally replace default attention projections with low-rank factors.

    ``rank`` remains the fallback for backwards compatibility.  Supplying
    ``q_rank``, ``k_rank``, ``v_rank``, or ``o_rank`` overrides the legacy
    fallback for that projection independently. Sharing scopes validate only
    the ranks of projections that actually share an input bottleneck.
    """
    if not isinstance(enabled, bool):
        raise OLMoConfigurationError("enabled must be a boolean")
    patched = config.copy()
    if not enabled:
        return patched
    rank = _validate_rank(rank)
    if q_rank is not None:
        q_rank = _validate_rank(q_rank)
    if k_rank is not None:
        k_rank = _validate_rank(k_rank)
    if v_rank is not None:
        v_rank = _validate_rank(v_rank)
    if o_rank is not None:
        o_rank = _validate_rank(o_rank)
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
                "q_rank",
                "k_rank",
                "v_rank",
                "o_rank",
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
            q_rank=q_rank,
            k_rank=k_rank,
            v_rank=v_rank,
            o_rank=o_rank,
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
