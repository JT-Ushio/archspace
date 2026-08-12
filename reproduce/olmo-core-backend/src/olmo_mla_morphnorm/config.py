from dataclasses import dataclass
from typing import Optional, Tuple

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention import AttentionBackendName, AttentionConfig, AttentionType
from olmo_core.nn.attention.base import SequenceMixerConfig
from olmo_core.nn.buffer_cache import BufferCache
from olmo_core.nn.rope import RoPEConfig

from .norm_types import MLANormType


@SequenceMixerConfig.register("mla")
@dataclass
class MLAAttentionConfig(AttentionConfig):
    """Configuration for multi-head latent attention with optional MorphNorm."""

    name: AttentionType = AttentionType.default
    q_lora_rank: Optional[int] = 1024
    kv_lora_rank: int = 512
    rope_dim: int = 64
    value_head_dim: Optional[int] = None
    norm_type: MLANormType = MLANormType.morphnorm
    use_q_a_layernorm: bool = True
    use_kv_a_layernorm: Optional[bool] = None
    use_ckv_layer_residual: bool = False
    morphnorm_eps: float = 1e-6
    morphnorm_update_stats: bool = True
    softmax_scale: Optional[float] = None
    return_max_logits: bool = False
    log_max_logits_per_head: bool = True

    def _use_kv_a_layernorm(self) -> bool:
        if self.use_kv_a_layernorm is not None:
            return self.use_kv_a_layernorm
        # This matches the effective forward in the reference MorphNorm patch.
        return self.norm_type != MLANormType.morphnorm

    def _resolved_dims(self, d_model: int) -> Tuple[int, int, int, int]:
        qk_head_dim = self.head_dim if self.head_dim is not None else d_model // self.n_heads
        value_head_dim = self.value_head_dim if self.value_head_dim is not None else qk_head_dim
        n_kv_heads = self.n_kv_heads if self.n_kv_heads is not None else self.n_heads
        nope_dim = qk_head_dim - self.rope_dim
        return qk_head_dim, value_head_dim, n_kv_heads, nope_dim

    def _validate(self, d_model: int) -> None:
        qk_head_dim, value_head_dim, n_kv_heads, nope_dim = self._resolved_dims(d_model)
        if self.n_heads <= 0 or n_kv_heads <= 0 or self.n_heads % n_kv_heads != 0:
            raise OLMoConfigurationError(
                "MLA requires n_heads to be divisible by the resolved n_kv_heads"
            )
        if qk_head_dim <= 0 or value_head_dim <= 0:
            raise OLMoConfigurationError("MLA head dimensions must be positive")
        if value_head_dim > qk_head_dim:
            raise OLMoConfigurationError("value_head_dim cannot exceed qk_head_dim")
        if self.rope_dim < 0 or nope_dim < 0:
            raise OLMoConfigurationError(
                f"rope_dim must be in [0, {qk_head_dim}], got {self.rope_dim}"
            )
        if self.rope_dim % 2 != 0:
            raise OLMoConfigurationError("rope_dim must be even")
        if self.kv_lora_rank <= 0:
            raise OLMoConfigurationError("kv_lora_rank must be positive")
        if self.q_lora_rank is not None and self.q_lora_rank <= 0:
            raise OLMoConfigurationError("q_lora_rank must be positive or None")
        if self.bias:
            raise OLMoConfigurationError("Biased projections are not supported by absorbed MLA")
        if self.gate is not None:
            raise OLMoConfigurationError("Attention gates are not implemented for MLA")
        if self.clip_qkv is not None:
            raise OLMoConfigurationError("clip_qkv is not implemented for MLA")
        if self.return_max_logits and self.backend != AttentionBackendName.flash_3:
            raise OLMoConfigurationError(
                "return_max_logits requires backend='flash_3' from flash-attention-max-logits"
            )
        needs_norm = (
            (self.q_lora_rank is not None and self.use_q_a_layernorm)
            or self._use_kv_a_layernorm()
            or self.norm_type != MLANormType.baseline
        )
        if needs_norm and self.qk_norm is None:
            raise OLMoConfigurationError("qk_norm must be provided for the selected MLA norms")

    def num_params(self, d_model: int) -> int:
        """Return the number of parameters in the built MLA module."""
        self._validate(d_model)
        qk_dim, value_dim, n_kv_heads, nope_dim = self._resolved_dims(d_model)
        norm = self.qk_norm
        params = 0

        if self.q_lora_rank is None:
            params += d_model * self.n_heads * qk_dim
        else:
            params += d_model * self.q_lora_rank
            params += self.q_lora_rank * self.n_heads * qk_dim
            if self.use_q_a_layernorm:
                assert norm is not None
                params += norm.num_params(self.q_lora_rank)

        params += d_model * (self.kv_lora_rank + self.rope_dim)
        if self._use_kv_a_layernorm():
            assert norm is not None
            params += norm.num_params(self.kv_lora_rank)
        params += self.kv_lora_rank * n_kv_heads * nope_dim
        params += self.kv_lora_rank * n_kv_heads * value_dim
        params += self.n_heads * value_dim * d_model

        if self.norm_type in (
            MLANormType.materialize_norm,
            MLANormType.standard_qk_norm,
            MLANormType.morphnorm,
        ):
            assert norm is not None
            params += norm.num_params(qk_dim)
        if self.norm_type == MLANormType.standard_qk_norm:
            assert norm is not None
            params += norm.num_params(qk_dim)
        elif self.norm_type in (MLANormType.materialize_norm, MLANormType.morphnorm):
            if self.rope_dim > 0:
                assert norm is not None
                params += norm.num_params(self.rope_dim)
        if self.norm_type == MLANormType.morphnorm:
            params += nope_dim
        return params

    def build(
        self,
        d_model: int,
        *,
        layer_idx: int,
        n_layers: int,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ):
        """Build an MLA sequence mixer for a transformer layer."""
        from .module import MLAAttention

        self._validate(d_model)
        window_size: Optional[int] = None
        rope: Optional[RoPEConfig] = None if self.rope is None else self.rope.copy()
        if self.sliding_window is not None and self.sliding_window.should_use_swa(
            layer_idx, n_layers
        ):
            window_size = self.sliding_window.get_window_size(layer_idx, n_layers)
        elif rope is not None and rope.no_global_rope:
            rope = None

        if rope is not None:
            # rope_dim is already the exact rotary width; do not apply a second partial factor.
            rope.partial_rotary_factor = 1.0

        return MLAAttention(
            d_model=d_model,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            qk_head_dim=self.head_dim,
            value_head_dim=self.value_head_dim,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            rope_dim=self.rope_dim,
            norm_type=self.norm_type,
            use_q_a_layernorm=self.use_q_a_layernorm,
            use_kv_a_layernorm=self._use_kv_a_layernorm(),
            use_ckv_layer_residual=self.use_ckv_layer_residual,
            norm_config=self.qk_norm,
            rope=rope,
            dropout=self.dropout or 0.0,
            softmax_scale=self.softmax_scale,
            use_flash=self.use_flash,
            backend=self.backend,
            window_size=window_size,
            dtype=self.dtype.as_pt(),
            init_device=init_device,
            cache=cache,
            morphnorm_eps=self.morphnorm_eps,
            morphnorm_update_stats=self.morphnorm_update_stats,
            return_max_logits=self.return_max_logits,
            log_max_logits_per_head=self.log_max_logits_per_head,
        )
