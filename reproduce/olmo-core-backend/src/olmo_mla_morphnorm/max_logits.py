import importlib
import inspect
from dataclasses import dataclass
from types import ModuleType
from typing import Optional, Tuple, Union

import torch

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention import Attention, AttentionBackendName, AttentionConfig, AttentionType
from olmo_core.nn.attention.backend import FlashAttention3Backend, TorchAttentionBackend
from olmo_core.nn.attention.base import SequenceMixerConfig
from olmo_core.nn.attention.kv_cache import KVCacheManager
from olmo_core.nn.buffer_cache import BufferCache

_CUSTOM_FA3_OVERRIDE: Optional[ModuleType] = None
_CUSTOM_FA3_CACHE: Optional[ModuleType] = None
_VALIDATED_FA3_MODULE_IDS: set[int] = set()


def _get_custom_fa3() -> ModuleType:
    global _CUSTOM_FA3_CACHE
    if _CUSTOM_FA3_OVERRIDE is not None:
        return _CUSTOM_FA3_OVERRIDE
    if _CUSTOM_FA3_CACHE is not None:
        return _CUSTOM_FA3_CACHE
    for module_name in (
        "flash_attn_3.flash_attn_interface",
        "flash_attn_interface",
    ):
        try:
            _CUSTOM_FA3_CACHE = importlib.import_module(module_name)
            return _CUSTOM_FA3_CACHE
        except ImportError:
            continue
    raise RuntimeError(
        "Custom FlashAttention3 is required. Build the hopper/ package from "
        "flash-attention-max-logits instead of installing an upstream FA3 wheel."
    )


def _assert_max_logits_api(module: ModuleType) -> None:
    if id(module) in _VALIDATED_FA3_MODULE_IDS:
        return
    try:
        fixed_parameters = inspect.signature(module.flash_attn_func).parameters
        varlen_parameters = inspect.signature(module.flash_attn_varlen_func).parameters
    except (AttributeError, TypeError, ValueError) as error:
        raise RuntimeError("Unable to inspect the installed FlashAttention3 API") from error
    if "return_max_logits" not in fixed_parameters or "return_max_logits" not in varlen_parameters:
        raise RuntimeError(
            "The installed FlashAttention3 does not expose return_max_logits. "
            "Build the local flash-attention-max-logits fork with FLASH_ATTENTION_FORCE_BUILD=TRUE."
        )
    if module is not _CUSTOM_FA3_OVERRIDE:
        try:
            schema = str(torch.ops.flash_attn_3.fwd.default._schema)
        except (AttributeError, RuntimeError) as error:
            raise RuntimeError(
                "Unable to inspect the installed FlashAttention3 binary schema"
            ) from error
        if "max_logits" not in schema:
            raise RuntimeError(
                "FlashAttention3 has a max-logit Python wrapper but a stale binary extension. "
                "Force-reinstall the local flash-attention-max-logits fork."
            )
    _VALIDATED_FA3_MODULE_IDS.add(id(module))


def _normalize_varlen_args(
    *,
    cu_seqlens: Optional[torch.Tensor],
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    max_seqlen: Optional[int],
    max_seqlen_q: Optional[int],
    max_seqlen_k: Optional[int],
) -> Tuple[
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[int],
    Optional[int],
    bool,
]:
    if cu_seqlens is not None:
        cu_seqlens_q = cu_seqlens if cu_seqlens_q is None else cu_seqlens_q
        cu_seqlens_k = cu_seqlens if cu_seqlens_k is None else cu_seqlens_k
    if max_seqlen is not None:
        max_seqlen_q = max_seqlen if max_seqlen_q is None else max_seqlen_q
        max_seqlen_k = max_seqlen if max_seqlen_k is None else max_seqlen_k

    values = (cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k)
    if any(value is not None for value in values) and not all(
        value is not None for value in values
    ):
        raise ValueError("FA3 varlen attention requires complete Q and K length metadata")
    return (
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        all(value is not None for value in values),
    )


def _validate_result(result, n_heads: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError(
            "Custom FlashAttention3 returned an unexpected result for return_max_logits=True"
        )
    output, max_logits = result
    if max_logits.shape != (n_heads,) or max_logits.dtype != torch.float32:
        raise RuntimeError(
            f"Expected per-head FP32 max logits with shape {(n_heads,)}, "
            f"got {tuple(max_logits.shape)} and {max_logits.dtype}"
        )
    return output, max_logits


def dispatch_flash_attn_3_with_max_logits(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the custom FA3 forward and return output plus per-query-head max logits."""
    flash_attn_3 = _get_custom_fa3()
    _assert_max_logits_api(flash_attn_3)
    return _dispatch_flash_attn_3_with_max_logits(
        flash_attn_3,
        q,
        k,
        v,
        cu_seqlens=cu_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen=max_seqlen,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
    )


def _dispatch_flash_attn_3_with_max_logits(
    flash_attn_3: ModuleType,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
) -> Tuple[torch.Tensor, torch.Tensor]:
    (
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        varlen,
    ) = _normalize_varlen_args(
        cu_seqlens=cu_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen=max_seqlen,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
    )

    if varlen:
        assert cu_seqlens_q is not None and cu_seqlens_k is not None
        assert max_seqlen_q is not None and max_seqlen_k is not None
        result = flash_attn_3.flash_attn_varlen_func(
            q.flatten(0, 1),
            k.flatten(0, 1),
            v.flatten(0, 1),
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            return_max_logits=True,
        )
    else:
        result = flash_attn_3.flash_attn_func(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            return_max_logits=True,
        )
    return _validate_result(result, q.shape[-2])


class MaxLogitsFlashAttention3Backend(FlashAttention3Backend):
    """FA3 backend that accumulates scaled max QK logits per query head."""

    def __init__(
        self,
        *,
        head_dim: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        scale: Optional[float] = None,
        dropout_p: float = 0.0,
        window_size: Tuple[int, int] = (-1, -1),
        cache: Optional[BufferCache] = None,
        log_per_head: bool = True,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__(
            head_dim=head_dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            scale=scale,
            dropout_p=dropout_p,
            window_size=window_size,
            cache=cache,
        )
        self._flash_attn_3 = _get_custom_fa3()
        _assert_max_logits_api(self._flash_attn_3)
        self.log_per_head = log_per_head
        self.register_buffer(
            "max_logits_accumulator",
            torch.full((n_heads,), -torch.inf, dtype=torch.float32, device=device),
            persistent=False,
        )

    def reset_parameters(self) -> None:
        """Reset the nonpersistent max-logit accumulator."""
        self.max_logits_accumulator.fill_(-torch.inf)

    def reset_max_logits(self) -> None:
        """Clear accumulated max logits for a new full batch."""
        self.max_logits_accumulator.fill_(-torch.inf)

    def get_max_logits(self, reset: bool = True) -> torch.Tensor:
        """Return a snapshot of accumulated per-head maxima."""
        value = self.max_logits_accumulator.clone()
        if reset:
            self.reset_max_logits()
        return value

    def _record_max_logits(self, max_logits: torch.Tensor) -> None:
        if max_logits.shape != self.max_logits_accumulator.shape:
            raise RuntimeError(
                "Max-logit head sharding is not supported by this backend: "
                f"expected {tuple(self.max_logits_accumulator.shape)}, "
                f"got {tuple(max_logits.shape)}"
            )
        self.max_logits_accumulator.copy_(
            torch.maximum(self.max_logits_accumulator, max_logits.detach().float())
        )

    def forward(
        self,
        qkv: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        cu_doc_lens: Optional[torch.Tensor] = None,
        cu_doc_lens_q: Optional[torch.Tensor] = None,
        cu_doc_lens_k: Optional[torch.Tensor] = None,
        max_doc_len: Optional[int] = None,
        max_doc_len_q: Optional[int] = None,
        max_doc_len_k: Optional[int] = None,
        local_k_slice: Optional[slice] = None,
        kv_cache_manager: Optional[KVCacheManager] = None,
    ) -> torch.Tensor:
        # Metrics are training-only. Evaluation and generation retain OLMo's standard FA3 path.
        if not self.training:
            return super().forward(
                qkv,
                cu_doc_lens=cu_doc_lens,
                cu_doc_lens_q=cu_doc_lens_q,
                cu_doc_lens_k=cu_doc_lens_k,
                max_doc_len=max_doc_len,
                max_doc_len_q=max_doc_len_q,
                max_doc_len_k=max_doc_len_k,
                local_k_slice=local_k_slice,
                kv_cache_manager=kv_cache_manager,
            )
        if isinstance(qkv, torch.Tensor):
            raise RuntimeError("Max-logit monitoring currently requires separate Q, K, and V")
        if self.cp_enabled:
            raise RuntimeError("Max-logit monitoring is not implemented with context parallelism")
        if kv_cache_manager is not None:
            raise RuntimeError("Training with a KV cache is not supported")
        if local_k_slice is not None:
            raise RuntimeError("local_k_slice requires a context-parallel backend")

        q, k, v = qkv
        output, max_logits = _dispatch_flash_attn_3_with_max_logits(
            self._flash_attn_3,
            q,
            k,
            v,
            cu_seqlens=cu_doc_lens,
            cu_seqlens_q=cu_doc_lens_q,
            cu_seqlens_k=cu_doc_lens_k,
            max_seqlen=max_doc_len,
            max_seqlen_q=max_doc_len_q,
            max_seqlen_k=max_doc_len_k,
            softmax_scale=self.scale,
            causal=True,
            window_size=self.window_size,
        )
        self._record_max_logits(max_logits)
        return output


@SequenceMixerConfig.register("attention_max_logits")
@dataclass
class MaxLogitsAttentionConfig(AttentionConfig):
    """Standard OLMo attention configured with the custom max-logit FA3 backend."""

    log_max_logits_per_head: bool = True

    def build(
        self,
        d_model: int,
        *,
        layer_idx: int,
        n_layers: int,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ):
        if self.name != AttentionType.default:
            raise OLMoConfigurationError(
                "Max-logit monitoring currently supports separate-projection Attention only"
            )
        if self.backend != AttentionBackendName.flash_3:
            raise OLMoConfigurationError("MaxLogitsAttentionConfig requires backend='flash_3'")
        base_config = AttentionConfig(
            name=self.name,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim,
            bias=self.bias,
            gate=self.gate,
            rope=self.rope,
            clip_qkv=self.clip_qkv,
            qk_norm=self.qk_norm,
            dropout=self.dropout,
            use_flash=self.use_flash,
            backend=self.backend,
            dtype=self.dtype,
            sliding_window=self.sliding_window,
            use_head_qk_norm=self.use_head_qk_norm,
        )
        attention = base_config.build(
            d_model,
            layer_idx=layer_idx,
            n_layers=n_layers,
            init_device=init_device,
            cache=cache,
        )
        if not isinstance(attention, Attention):
            raise OLMoConfigurationError("Expected standard OLMo Attention")
        if isinstance(attention.backend, TorchAttentionBackend) and not torch.cuda.is_available():
            return attention
        if not isinstance(attention.backend, FlashAttention3Backend):
            raise OLMoConfigurationError(
                f"Expected FlashAttention3Backend, got {type(attention.backend).__name__}"
            )
        backend = attention.backend
        attention.backend = MaxLogitsFlashAttention3Backend(
            head_dim=backend.head_dim,
            n_heads=backend.n_heads,
            n_kv_heads=backend.n_kv_heads,
            scale=backend.scale,
            dropout_p=backend.dropout_p,
            window_size=backend.window_size,
            cache=backend.cache,
            log_per_head=self.log_max_logits_per_head,
            device=attention.w_q.weight.device,
        )
        return attention
