import warnings
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed.tensor import Placement

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention import Attention, AttentionBackendName
from olmo_core.nn.attention.backend import AttentionBackend
from olmo_core.nn.attention.ring import RingContextParallelStyle, UlyssesContextParallelStyle
from olmo_core.nn.buffer_cache import BufferCache
from olmo_core.nn.layer_norm import LayerNorm, LayerNormConfig
from olmo_core.nn.rope import (
    ComplexRotaryEmbedding,
    FusedRotaryEmbedding,
    RoPEConfig,
    RotaryEmbedding,
)

from .attention_paths import materialized_attention, mqa_attention
from .cache import MLACacheManager
from .morphnorm import commit_morphnorm_stats, materialize_kv, prepare_cache_latents, rms
from .norm_types import MLANormType

if TYPE_CHECKING:
    from olmo_core.nn.transformer.init import InitMethod


class MLAAttention(Attention):
    """
    Multi-head latent attention with materialized GQA prefill and absorbed MQA decode.

    The training path keeps the configured number of OLMo KV groups. During cached decoding,
    head-specific key/value up-projections are absorbed into the query and output paths, while
    the cache stores only one shared latent and one RoPE key per token.
    """

    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        n_kv_heads: Optional[int],
        qk_head_dim: Optional[int],
        value_head_dim: Optional[int],
        q_lora_rank: Optional[int],
        kv_lora_rank: int,
        rope_dim: int,
        norm_type: MLANormType,
        use_q_a_layernorm: bool,
        use_kv_a_layernorm: bool,
        norm_config: Optional[LayerNormConfig],
        rope: Optional[RoPEConfig],
        dropout: float,
        softmax_scale: Optional[float],
        use_flash: Optional[bool],
        backend: Optional[AttentionBackendName],
        window_size: Optional[int],
        dtype: torch.dtype,
        init_device: str,
        cache: Optional[BufferCache],
        morphnorm_eps: float,
        morphnorm_update_stats: bool,
        return_max_logits: bool,
        log_max_logits_per_head: bool,
    ) -> None:
        nn.Module.__init__(self)
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.head_dim = qk_head_dim if qk_head_dim is not None else d_model // n_heads
        self.value_head_dim = value_head_dim if value_head_dim is not None else self.head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.rope_dim = rope_dim
        self.nope_dim = self.head_dim - rope_dim
        self.norm_type = MLANormType(norm_type)
        self.use_q_a_layernorm = use_q_a_layernorm
        self.use_kv_a_layernorm = use_kv_a_layernorm
        self.morphnorm_eps = morphnorm_eps
        self.morphnorm_update_stats = morphnorm_update_stats
        self.dropout = dropout
        self.softmax_scale = softmax_scale if softmax_scale is not None else self.head_dim**-0.5
        self.window_size = window_size

        if self.n_heads % self.n_kv_heads != 0:
            raise OLMoConfigurationError("n_heads must be divisible by n_kv_heads")
        if self.rope_dim < 0 or self.rope_dim > self.head_dim or self.rope_dim % 2:
            raise OLMoConfigurationError("rope_dim must be even and no larger than qk_head_dim")
        if self.value_head_dim > self.head_dim:
            raise OLMoConfigurationError(
                "value_head_dim cannot exceed qk_head_dim with the configured attention backends"
            )

        self.w_q: Optional[nn.Linear] = None
        self.w_q_a: Optional[nn.Linear] = None
        self.w_q_b: Optional[nn.Linear] = None
        if q_lora_rank is None:
            self.w_q = nn.Linear(
                d_model,
                n_heads * self.head_dim,
                bias=False,
                dtype=dtype,
                device=init_device,
            )
        else:
            self.w_q_a = nn.Linear(
                d_model, q_lora_rank, bias=False, dtype=dtype, device=init_device
            )
            self.w_q_b = nn.Linear(
                q_lora_rank,
                n_heads * self.head_dim,
                bias=False,
                dtype=dtype,
                device=init_device,
            )

        self.w_kv_a = nn.Linear(
            d_model,
            kv_lora_rank + rope_dim,
            bias=False,
            dtype=dtype,
            device=init_device,
        )
        self.w_k_b: Optional[nn.Linear] = None
        if self.nope_dim > 0:
            self.w_k_b = nn.Linear(
                kv_lora_rank,
                self.n_kv_heads * self.nope_dim,
                bias=False,
                dtype=dtype,
                device=init_device,
            )
        self.w_v_b = nn.Linear(
            kv_lora_rank,
            self.n_kv_heads * self.value_head_dim,
            bias=False,
            dtype=dtype,
            device=init_device,
        )
        self.w_out = nn.Linear(
            n_heads * self.value_head_dim,
            d_model,
            bias=False,
            dtype=dtype,
            device=init_device,
        )

        self.q_a_layernorm: Optional[LayerNorm] = None
        self.kv_a_layernorm: Optional[LayerNorm] = None
        self.q_norm: Optional[LayerNorm] = None
        self.k_norm: Optional[LayerNorm] = None
        self.k_rope_norm: Optional[LayerNorm] = None
        if q_lora_rank is not None and use_q_a_layernorm:
            assert norm_config is not None
            self.q_a_layernorm = norm_config.build(q_lora_rank, init_device=init_device)
        if use_kv_a_layernorm:
            assert norm_config is not None
            self.kv_a_layernorm = norm_config.build(kv_lora_rank, init_device=init_device)
        if self.norm_type != MLANormType.baseline:
            assert norm_config is not None
            self.q_norm = norm_config.build(self.head_dim, init_device=init_device)
        if self.norm_type == MLANormType.standard_qk_norm:
            assert norm_config is not None
            self.k_norm = norm_config.build(self.head_dim, init_device=init_device)
        elif self.norm_type in (MLANormType.materialize_norm, MLANormType.morphnorm):
            if rope_dim > 0:
                assert norm_config is not None
                self.k_rope_norm = norm_config.build(rope_dim, init_device=init_device)

        self.gamma_k: Optional[nn.Parameter] = None
        if self.norm_type == MLANormType.morphnorm and self.nope_dim > 0:
            self.gamma_k = nn.Parameter(torch.ones(self.nope_dim, dtype=dtype, device=init_device))
        self.register_buffer(
            "morphnorm_scale",
            torch.ones(self.n_kv_heads, dtype=dtype, device=init_device),
            persistent=True,
        )
        self.register_buffer(
            "morphnorm_pending_sum",
            torch.zeros(self.n_kv_heads, dtype=torch.float32, device=init_device),
            persistent=False,
        )
        self.register_buffer(
            "morphnorm_pending_count",
            torch.zeros((), dtype=torch.float32, device=init_device),
            persistent=False,
        )

        self.rope: Optional[RotaryEmbedding | ComplexRotaryEmbedding] = None
        if rope_dim > 0 and rope is not None:
            rope_module = rope.build(rope_dim, cache=cache)
            if isinstance(rope_module, FusedRotaryEmbedding):
                raise OLMoConfigurationError("Fused RoPE is not supported by MLAAttention")
            if not isinstance(rope_module, (RotaryEmbedding, ComplexRotaryEmbedding)):
                raise OLMoConfigurationError(
                    f"Unsupported RoPE implementation: {type(rope_module).__name__}"
                )
            self.rope = rope_module

        backend_name = AttentionBackendName(backend or AttentionBackendName.torch)
        if use_flash:
            if backend is not None and backend_name != AttentionBackendName.flash_2:
                raise OLMoConfigurationError(
                    f"use_flash is only compatible with flash_2, got {backend_name}"
                )
            backend_name = AttentionBackendName.flash_2
        if not torch.cuda.is_available() and backend_name != AttentionBackendName.torch:
            warnings.warn(
                f"Backend is set to {backend_name}, but GPUs are not available. "
                "Defaulting to torch."
            )
            backend_name = AttentionBackendName.torch

        window_size_tuple = (-1, -1) if window_size is None else (window_size - 1, 0)
        if return_max_logits and backend_name == AttentionBackendName.flash_3:
            from .max_logits import MaxLogitsFlashAttention3Backend

            self.backend: AttentionBackend = MaxLogitsFlashAttention3Backend(
                head_dim=self.head_dim,
                n_heads=self.n_heads,
                n_kv_heads=self.n_kv_heads,
                scale=self.softmax_scale,
                dropout_p=dropout,
                window_size=window_size_tuple,
                cache=cache,
                log_per_head=log_max_logits_per_head,
                device=init_device,
            )
        else:
            self.backend = backend_name.build(
                head_dim=self.head_dim,
                n_heads=self.n_heads,
                n_kv_heads=self.n_kv_heads,
                scale=self.softmax_scale,
                dropout_p=dropout,
                window_size=window_size_tuple,
                cache=cache,
            )
        self.kv_cache_manager: Optional[MLACacheManager] = None

    @property
    def cp_enabled(self) -> bool:
        """Return whether context parallelism is enabled on the training backend."""
        return self.backend.cp_enabled

    @staticmethod
    def _rms(x: torch.Tensor, eps: float) -> torch.Tensor:
        return rms(x, eps)

    def _project(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        if self.w_q is not None:
            q = self.w_q(x)
        else:
            assert self.w_q_a is not None and self.w_q_b is not None
            q_latent = self.w_q_a(x)
            if self.q_a_layernorm is not None:
                q_latent = self.q_a_layernorm(q_latent)
            q = self.w_q_b(q_latent)
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)
        if self.q_norm is not None:
            q = self.q_norm(q)
        q_nope, q_rope = q.split((self.nope_dim, self.rope_dim), dim=-1)

        compressed_kv = self.w_kv_a(x)
        latent, k_rope = compressed_kv.split((self.kv_lora_rank, self.rope_dim), dim=-1)
        if self.kv_a_layernorm is not None:
            latent = self.kv_a_layernorm(latent)
        k_rope = k_rope.view(batch_size, seq_len, 1, self.rope_dim)
        if self.k_rope_norm is not None:
            k_rope = self.k_rope_norm(k_rope)
        return q_nope, q_rope, latent, k_rope

    def _apply_rope_components(
        self,
        q_rope: torch.Tensor,
        k_rope: torch.Tensor,
        *,
        pos_sin: Optional[torch.Tensor],
        pos_cos: Optional[torch.Tensor],
        freqs_cis: Optional[torch.Tensor],
        cu_doc_lens: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.rope is None or self.rope_dim == 0:
            return q_rope, k_rope
        if self.cp_enabled and pos_sin is None and pos_cos is None and freqs_cis is None:
            raise RuntimeError("RoPE buffers must be supplied after context-parallel sharding")
        start_pos = (
            self.kv_cache_manager.current_position() if self.kv_cache_manager is not None else None
        )
        return self._apply_rope(
            q_rope,
            k_rope,
            start_pos,
            pos_sin,
            pos_cos,
            freqs_cis,
            cu_doc_lens,
        )

    def _prepare_cache_latents(self, latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return prepare_cache_latents(self, latent)

    def _materialize_kv(
        self, latent: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return materialize_kv(self, latent)

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
        """Apply MLA to an input of shape ``[batch, sequence, d_model]``."""
        batch_size, seq_len, _ = x.shape
        if self.kv_cache_manager is not None and self.cp_enabled:
            raise RuntimeError("Cached MLA decoding cannot be combined with context parallelism")
        if self.kv_cache_manager is not None and any(
            item is not None
            for item in (
                cu_doc_lens,
                cu_doc_lens_q,
                cu_doc_lens_k,
                max_doc_len,
                max_doc_len_q,
                max_doc_len_k,
                local_k_slice,
            )
        ):
            raise RuntimeError(
                "Packed-document/context-parallel inputs are not supported with cache"
            )

        q_nope, q_rope, latent, k_rope = self._project(x)
        if self.kv_cache_manager is not None:
            self.kv_cache_manager.ensure_compatible(latent)
            self.kv_cache_manager.record_leftpad(cache_leftpad)

        if self.kv_cache_manager is not None and self.kv_cache_manager.has_data:
            _, value_latent = self._prepare_cache_latents(latent)
            q_rope, k_rope = self._apply_rope_components(
                q_rope,
                k_rope,
                pos_sin=pos_sin,
                pos_cos=pos_cos,
                freqs_cis=freqs_cis,
                cu_doc_lens=None,
            )
            latent_cache, rope_cache = self.kv_cache_manager.append(value_latent, k_rope.squeeze(2))
            output = mqa_attention(self, q_nope, q_rope, latent_cache, rope_cache)
        else:
            k_nope, value, _, value_latent = self._materialize_kv(latent)
            if self.norm_type == MLANormType.standard_qk_norm:
                assert self.k_norm is not None
                expanded_k_rope = k_rope.expand(-1, -1, self.n_kv_heads, -1)
                normalized_k = self.k_norm(torch.cat((k_nope, expanded_k_rope), dim=-1))
                k_nope, materialized_k_rope = normalized_k.split(
                    (self.nope_dim, self.rope_dim), dim=-1
                )
            else:
                materialized_k_rope = k_rope

            q_rope, materialized_k_rope = self._apply_rope_components(
                q_rope,
                materialized_k_rope,
                pos_sin=pos_sin,
                pos_cos=pos_cos,
                freqs_cis=freqs_cis,
                cu_doc_lens=cu_doc_lens,
            )
            cache_rope_key = materialized_k_rope
            if materialized_k_rope.shape[2] == 1:
                materialized_k_rope = materialized_k_rope.expand(-1, -1, self.n_kv_heads, -1)
            q = torch.cat((q_nope, q_rope), dim=-1)
            k = torch.cat((k_nope, materialized_k_rope), dim=-1)
            output = materialized_attention(
                self,
                q,
                k,
                value,
                cu_doc_lens=cu_doc_lens,
                cu_doc_lens_q=cu_doc_lens_q,
                cu_doc_lens_k=cu_doc_lens_k,
                max_doc_len=max_doc_len,
                max_doc_len_q=max_doc_len_q,
                max_doc_len_k=max_doc_len_k,
                local_k_slice=local_k_slice,
                cache_leftpad=cache_leftpad,
            )
            if self.kv_cache_manager is not None:
                self.kv_cache_manager.append(value_latent, cache_rope_key.squeeze(2))

        return self.w_out(output.reshape(batch_size, seq_len, -1))

    def init_kv_cache_manager(self, batch_size: int, max_seq_len: int) -> None:
        """Initialize the compressed single-head MLA inference cache."""
        if self.cp_enabled:
            raise OLMoConfigurationError(
                "Cached MLA decoding cannot be combined with context parallelism"
            )
        if self.norm_type == MLANormType.standard_qk_norm:
            raise OLMoConfigurationError(
                "standard_qk_norm is not algebraically absorbable for MQA decoding"
            )
        self.kv_cache_manager = MLACacheManager(
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            latent_dim=self.kv_lora_rank,
            rope_dim=self.rope_dim,
            device=self.w_kv_a.weight.device,
            dtype=self.w_kv_a.weight.dtype,
        )

    def apply_tp(
        self,
        tp_mesh: DeviceMesh,
        input_layout: Optional[Placement] = None,
        output_layout: Optional[Placement] = None,
        use_local_output: bool = True,
        float8_enabled: bool = False,
    ) -> None:
        """Reject tensor parallelism until latent/head sharding is implemented explicitly."""
        del tp_mesh, input_layout, output_layout, use_local_output, float8_enabled
        raise NotImplementedError("Tensor parallelism is not implemented for MLAAttention")

    def apply_cp(
        self,
        cp_mesh: DeviceMesh,
        ring: Optional[RingContextParallelStyle] = None,
        uly: Optional[UlyssesContextParallelStyle] = None,
    ) -> None:
        """Apply OLMo-core context parallelism to the materialized training backend."""
        if self.kv_cache_manager is not None:
            raise OLMoConfigurationError(
                "Context parallelism cannot be enabled while an MLA cache is active"
            )
        self.backend.apply_cp(cp_mesh, ring=ring, uly=uly)

    def reset_parameters(self) -> None:
        """Reset MorphNorm gains and running statistics."""
        if self.gamma_k is not None:
            nn.init.ones_(self.gamma_k)
        self.morphnorm_scale.fill_(1.0)
        self.morphnorm_pending_sum.zero_()
        self.morphnorm_pending_count.zero_()

    def post_batch(self, dry_run: bool = False) -> None:
        """Commit accumulated MorphNorm statistics after the complete backward pass."""
        commit_morphnorm_stats(self, dry_run=dry_run)

    def init_weights(
        self,
        *,
        init_method: "InitMethod",
        d_model: int,
        block_idx: int,
        num_blocks: int,
        std: float = 0.02,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        """Initialize MLA projections with OLMo-core's configured initialization method."""
        from olmo_core.nn.transformer.init import InitMethod, init_linear

        projections = [
            module
            for module in (
                self.w_q,
                self.w_q_a,
                self.w_q_b,
                self.w_kv_a,
                self.w_k_b,
                self.w_v_b,
            )
            if module is not None
        ]
        for projection in projections:
            projection_std = std
            if init_method == InitMethod.fan_in:
                projection_std = projection.in_features**-0.5
            elif init_method == InitMethod.normalized:
                projection_std = d_model**-0.5
            init_linear(projection, std=projection_std, generator=generator)

        output_std = std
        if init_method == InitMethod.fan_in:
            output_std = self.w_out.in_features**-0.5
        elif init_method == InitMethod.llama:
            output_std = std / (2 * num_blocks) ** 0.5
        elif init_method == InitMethod.llama_depth:
            output_std = std / (2 * (block_idx + 1)) ** 0.5
        elif init_method == InitMethod.normalized:
            output_std = d_model**-0.5 / (2 * num_blocks) ** 0.5
        init_linear(self.w_out, std=output_std, generator=generator)
        self.reset_parameters()

    def num_flops_per_token(self, seq_len: int) -> int:
        """Estimate idealized training FLOPs per token."""
        param_flops = 6 * sum(parameter.numel() for parameter in self.parameters())
        effective_seq_len = min(self.window_size, seq_len) if self.window_size else seq_len
        attention_flops = (
            6 * self.n_heads * (self.head_dim + self.value_head_dim) * effective_seq_len
        )
        return param_flops + attention_flops
