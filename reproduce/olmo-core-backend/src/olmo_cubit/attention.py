import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import DeviceMesh
from torch.distributed.tensor import Placement

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention import (
    Attention,
    AttentionBackendName,
    AttentionConfig,
    AttentionType,
    GateConfig,
    SlidingWindowAttentionConfig,
    TorchAttentionBackend,
)
from olmo_core.nn.attention.ring import RingContextParallelStyle, UlyssesContextParallelStyle
from olmo_core.nn.buffer_cache import BufferCache
from olmo_core.nn.layer_norm import LayerNormConfig
from olmo_core.nn.rope import RoPEConfig

from .krr import streaming_causal_krr_solve


def _check_finite(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise OLMoConfigurationError(f"{name} must be a finite number")


@dataclass
class CubitAttentionConfig(AttentionConfig):
    """OLMo attention configuration for the Cubit KRR token mixer."""

    share_reference: bool = False
    regularization: float = 1e-10
    lrr_lower: float = 0.5
    lrr_upper: float = 2.0
    reference_norm_eps: float = 1e-6
    krr_implementation: str = "streaming"
    krr_block_size: int = 64

    def __post_init__(self, type: Optional[str] = None) -> None:
        del type
        self.name = AttentionType(self.name)
        if self.backend is not None:
            self.backend = AttentionBackendName(self.backend)

        if self.name != AttentionType.default:
            raise OLMoConfigurationError(
                "Cubit requires OLMo's default (unfused-projection) attention topology"
            )
        if self.n_kv_heads is not None and self.n_kv_heads != self.n_heads:
            raise OLMoConfigurationError("Cubit v1 requires multi-head attention (n_kv_heads=n_heads)")
        if not isinstance(self.share_reference, bool):
            raise OLMoConfigurationError("share_reference must be a boolean")
        if self.dropout not in (None, 0, 0.0):
            raise OLMoConfigurationError("Cubit v1 does not support attention dropout")

        for name, value in (
            ("regularization", self.regularization),
            ("lrr_lower", self.lrr_lower),
            ("lrr_upper", self.lrr_upper),
            ("reference_norm_eps", self.reference_norm_eps),
        ):
            _check_finite(name, value)
        if self.regularization <= 0:
            raise OLMoConfigurationError("regularization must be greater than zero")
        if not 0 < self.lrr_lower < self.lrr_upper:
            raise OLMoConfigurationError("LRR bounds must satisfy 0 < lrr_lower < lrr_upper")
        if self.reference_norm_eps <= 0:
            raise OLMoConfigurationError("reference_norm_eps must be greater than zero")
        if self.krr_implementation not in ("streaming", "dense"):
            raise OLMoConfigurationError(
                "krr_implementation must be either 'streaming' or 'dense'"
            )
        if (
            isinstance(self.krr_block_size, bool)
            or not isinstance(self.krr_block_size, int)
            or self.krr_block_size <= 0
        ):
            raise OLMoConfigurationError("krr_block_size must be a positive integer")

    def num_params(self, d_model: int) -> int:
        params = super().num_params(d_model)
        head_dim = self.head_dim or d_model // self.n_heads
        bias = self.bias if self.bias is not None else True

        if not self.share_reference:
            params += d_model * self.n_heads * head_dim
            if bias:
                params += self.n_heads * head_dim

        params += d_model * self.n_heads
        if bias:
            params += self.n_heads

        # Per-head LRR lower/range, reference scale, and log(lambda).
        params += 4 * self.n_heads
        return params

    def build(
        self,
        d_model: int,
        *,
        layer_idx: int,
        n_layers: int,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ) -> "CubitAttention":
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        kwargs.pop("name")

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
            dtype=kwargs.pop("dtype").as_pt(),
            d_model=d_model,
            init_device=init_device,
            cache=cache,
        )
        try:
            return CubitAttention(**kwargs)
        except TypeError as exc:
            raise OLMoConfigurationError(
                f"invalid options for {self.__class__.__name__}, {exc}"
            ) from exc


class CubitAttention(Attention):
    """Cubit token mixer using a causal kernel-ridge-regression correction.

    By default the KRR solve streams over token blocks and recomputes kernel tiles in backward,
    avoiding a persistent ``[T, T]`` matrix. The final ``A @ solution`` aggregation is delegated
    to the configured OLMo attention backend; the supplied recipe selects FlashAttention 3.
    """

    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        bias: bool = True,
        gate: Optional[GateConfig] = None,
        rope: Optional[RoPEConfig] = None,
        clip_qkv: Optional[float] = None,
        qk_norm: Optional[LayerNormConfig] = None,
        dropout: float = 0.0,
        softmax_scale: Optional[float] = None,
        use_flash: Optional[bool] = None,
        backend: Optional[AttentionBackendName] = None,
        window_size: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
        use_head_qk_norm: bool = False,
        share_reference: bool = False,
        regularization: float = 1e-10,
        lrr_lower: float = 0.5,
        lrr_upper: float = 2.0,
        reference_norm_eps: float = 1e-6,
        krr_implementation: str = "streaming",
        krr_block_size: int = 64,
    ):
        resolved_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        if resolved_kv_heads != n_heads:
            raise OLMoConfigurationError("Cubit v1 requires multi-head attention (n_kv_heads=n_heads)")
        if dropout != 0.0:
            raise OLMoConfigurationError("Cubit v1 does not support attention dropout")

        super().__init__(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=resolved_kv_heads,
            head_dim=head_dim,
            bias=bias,
            gate=gate,
            rope=rope,
            clip_qkv=clip_qkv,
            qk_norm=qk_norm,
            dropout=dropout,
            softmax_scale=softmax_scale,
            use_flash=use_flash,
            backend=backend,
            window_size=window_size,
            dtype=dtype,
            init_device=init_device,
            cache=cache,
            use_head_qk_norm=use_head_qk_norm,
        )

        self.share_reference = share_reference
        self.reference_norm_eps = reference_norm_eps
        self.krr_implementation = krr_implementation
        self.krr_block_size = krr_block_size
        self._initial_regularization = regularization
        self._initial_lrr_lower = lrr_lower
        self._initial_lrr_range = lrr_upper - lrr_lower

        projection_dim = n_heads * self.head_dim
        self.w_r: Optional[nn.Linear]
        if share_reference:
            self.w_r = None
        else:
            self.w_r = nn.Linear(
                d_model,
                projection_dim,
                bias=bias,
                dtype=dtype,
                device=init_device,
            )
        self.w_lrr = nn.Linear(
            d_model,
            n_heads,
            bias=bias,
            dtype=dtype,
            device=init_device,
        )

        parameter_kwargs = {"dtype": dtype, "device": init_device}
        self.lrr_lower = nn.Parameter(torch.full((n_heads,), lrr_lower, **parameter_kwargs))
        self.lrr_range = nn.Parameter(
            torch.full((n_heads,), lrr_upper - lrr_lower, **parameter_kwargs)
        )
        self.reference_scale = nn.Parameter(torch.ones(n_heads, **parameter_kwargs))
        self.log_regularization = nn.Parameter(
            torch.full((n_heads,), math.log(regularization), **parameter_kwargs)
        )

    @staticmethod
    def _document_ids(
        cu_doc_lens: torch.Tensor,
        *,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        if cu_doc_lens.ndim != 1 or cu_doc_lens.numel() < 2:
            raise RuntimeError("cu_doc_lens must be a 1D tensor with at least two entries")

        boundaries = cu_doc_lens.detach().to(device="cpu", dtype=torch.long)
        if int(boundaries[0]) != 0 or bool((boundaries[1:] <= boundaries[:-1]).any()):
            raise RuntimeError("cu_doc_lens must start at zero and be strictly increasing")
        total_tokens = batch_size * seq_len
        if int(boundaries[-1]) != total_tokens:
            raise RuntimeError(
                f"cu_doc_lens must end at batch_size * seq_len ({total_tokens})"
            )
        boundary_set = set(boundaries.tolist())
        if any(row_end not in boundary_set for row_end in range(seq_len, total_tokens + 1, seq_len)):
            raise RuntimeError("every packed batch row must end at a document boundary")

        lengths = (boundaries[1:] - boundaries[:-1]).to(device=device)
        doc_ids = torch.repeat_interleave(
            torch.arange(lengths.numel(), device=device),
            lengths,
        )
        return doc_ids.view(batch_size, seq_len)

    def _build_attention_mask(
        self,
        *,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        cu_doc_lens: Optional[torch.Tensor],
    ) -> torch.Tensor:
        positions = torch.arange(seq_len, device=device)
        query_pos = positions[:, None]
        key_pos = positions[None, :]
        mask = key_pos <= query_pos
        if self.window_size is not None:
            mask = mask & (key_pos >= query_pos - (self.window_size - 1))
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)

        if cu_doc_lens is not None:
            doc_ids = self._document_ids(
                cu_doc_lens,
                batch_size=batch_size,
                seq_len=seq_len,
                device=device,
            )
            mask = mask & (doc_ids[:, :, None] == doc_ids[:, None, :])
        return mask

    @staticmethod
    def _masked_softmax(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return torch.softmax(scores.masked_fill(~mask[:, None, :, :], -torch.inf), dim=-1)

    def _dense_output_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        scale = self.backend.scale
        if scale is None:
            scale = self.head_dim**-0.5
        scores = torch.einsum("bthd,bshd->bhts", q.float(), k.float()) * scale
        weights = self._masked_softmax(scores, mask)
        output = torch.einsum("bhts,bshd->bthd", weights, v.float())
        return output.to(q.dtype)

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
        if self.kv_cache_manager is not None or cache_leftpad is not None:
            raise NotImplementedError("Cubit v1 does not support KV caching")
        if local_k_slice is not None:
            raise NotImplementedError("Cubit v1 does not support context-parallel key slices")
        if any(value is not None for value in (cu_doc_lens_q, cu_doc_lens_k)):
            raise NotImplementedError("Cubit v1 only supports the symmetric cu_doc_lens input")
        if any(value is not None for value in (max_doc_len_q, max_doc_len_k)):
            raise NotImplementedError("Cubit v1 only supports the symmetric max_doc_len input")
        if (cu_doc_lens is None) != (max_doc_len is None):
            raise RuntimeError("cu_doc_lens and max_doc_len must be provided together")

        batch_size, seq_len, _ = x.shape
        q, k, v = self.w_q(x), self.w_k(x), self.w_v(x)
        projected_r = None if self.share_reference else self.w_r(x)  # type: ignore[operator]

        if self.clip_qkv is not None:
            q = q.clamp(min=-self.clip_qkv, max=self.clip_qkv)
            k = k.clamp(min=-self.clip_qkv, max=self.clip_qkv)
            v = v.clamp(min=-self.clip_qkv, max=self.clip_qkv)
            if projected_r is not None:
                projected_r = projected_r.clamp(min=-self.clip_qkv, max=self.clip_qkv)

        if not self.use_head_qk_norm:
            if self.q_norm is not None:
                q = self.q_norm(q)
            if self.k_norm is not None:
                k = self.k_norm(k)

        q = q.view(batch_size, seq_len, -1, self.head_dim)
        k = k.view(batch_size, seq_len, -1, self.head_dim)
        v = v.view(batch_size, seq_len, -1, self.head_dim)

        if self.use_head_qk_norm:
            if self.q_norm is not None:
                q = self.q_norm(q)
            if self.k_norm is not None:
                k = self.k_norm(k)

        if self.share_reference:
            # Share the actual K representation used by OLMo, including QK norm.
            r = k
        else:
            assert projected_r is not None
            r = projected_r.view(batch_size, seq_len, -1, self.head_dim)

        reference_scale = self.reference_scale.float().view(1, 1, self.n_heads, 1)
        normalized_r = F.normalize(
            r.float(),
            p=2,
            dim=-1,
            eps=self.reference_norm_eps,
        ) * reference_scale

        if self.rope is not None:
            start_pos = None
            q, k = self._apply_rope(
                q,
                k,
                start_pos,
                pos_sin,
                pos_cos,
                freqs_cis,
                cu_doc_lens,
            )
            r, normalized_r = self._apply_rope(
                r,
                normalized_r,
                start_pos,
                pos_sin,
                pos_cos,
                freqs_cis,
                cu_doc_lens,
            )

        r_heads = r.float().transpose(1, 2)
        normalized_r_heads = normalized_r.float().transpose(1, 2)
        lrr_logits = self.w_lrr(x).float().transpose(1, 2).unsqueeze(-1)
        lrr = self.lrr_lower.float().view(1, self.n_heads, 1, 1)
        lrr = lrr + self.lrr_range.float().view(1, self.n_heads, 1, 1) * torch.sigmoid(
            lrr_logits
        )
        rhs = lrr * v.float().transpose(1, 2)
        regularization = self.log_regularization.float().exp()

        mask: Optional[torch.Tensor] = None
        if self.krr_implementation == "streaming":
            doc_ids = None
            if cu_doc_lens is not None:
                doc_ids = self._document_ids(
                    cu_doc_lens,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    device=x.device,
                )
            solution = streaming_causal_krr_solve(
                r_heads,
                normalized_r_heads,
                rhs,
                regularization,
                doc_ids=doc_ids,
                window_size=self.window_size,
                block_size=self.krr_block_size,
            )
        else:
            mask = self._build_attention_mask(
                batch_size=batch_size,
                seq_len=seq_len,
                device=x.device,
                cu_doc_lens=cu_doc_lens,
            )
            inverse_sigma = self._masked_softmax(
                r_heads @ normalized_r_heads.transpose(-2, -1),
                mask,
            )
            identity = torch.eye(seq_len, device=x.device, dtype=torch.float32)
            inverse_sigma = inverse_sigma + regularization.view(
                1, self.n_heads, 1, 1
            ) * identity.view(1, 1, seq_len, seq_len)
            solution = torch.linalg.solve_triangular(inverse_sigma, rhs, upper=False)

        # ``solve_triangular`` commonly returns a column-major-like layout where the
        # head dimension has a stride greater than one after transposing back to BTHD.
        # FlashAttention 3 requires the final (head) dimension of every input to be
        # contiguous, even when no dtype conversion is needed.
        solution = solution.transpose(1, 2).to(q.dtype).contiguous()

        if isinstance(self.backend, TorchAttentionBackend):
            if mask is None:
                mask = self._build_attention_mask(
                    batch_size=batch_size,
                    seq_len=seq_len,
                    device=x.device,
                    cu_doc_lens=cu_doc_lens,
                )
            att = self._dense_output_attention(q, k, solution, mask)
        else:
            att = self.sdpa(
                q.contiguous(),
                k.contiguous(),
                solution,
                cu_doc_lens=cu_doc_lens,
                max_doc_len=max_doc_len,
            )

        if self.gate is not None:
            assert self.w_g is not None
            gate_logits = self.w_g(x)
            if self.gate.full_precision:
                gate_logits = gate_logits.float()
            gate_values = torch.sigmoid(gate_logits).to(att.dtype)
            if self.gate.granularity.value == "headwise":
                att = att * gate_values.unsqueeze(-1)
            else:
                att = att.reshape(batch_size, seq_len, -1) * gate_values

        return self.w_out(att.reshape(batch_size, seq_len, -1))

    def init_weights(
        self,
        *,
        init_method,
        d_model: int,
        block_idx: int,
        num_blocks: int,
        std: float = 0.02,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        from olmo_core.nn.transformer.init import InitMethod, init_linear

        super().init_weights(
            init_method=init_method,
            d_model=d_model,
            block_idx=block_idx,
            num_blocks=num_blocks,
            std=std,
            generator=generator,
        )

        projection_std = d_model**-0.5 if init_method in (
            InitMethod.normalized,
            InitMethod.fan_in,
        ) else std
        if self.w_r is not None:
            init_linear(self.w_r, std=projection_std, generator=generator)
        init_linear(self.w_lrr, std=projection_std, generator=generator)

        self.lrr_lower.fill_(self._initial_lrr_lower)
        self.lrr_range.fill_(self._initial_lrr_range)
        self.reference_scale.fill_(1.0)
        self.log_regularization.fill_(math.log(self._initial_regularization))

    def init_kv_cache_manager(self, batch_size: int, max_seq_len: int):
        del batch_size, max_seq_len
        raise NotImplementedError("Cubit v1 does not support KV caching")

    def apply_tp(
        self,
        tp_mesh: DeviceMesh,
        input_layout: Optional[Placement] = None,
        output_layout: Optional[Placement] = None,
        use_local_output: bool = True,
        float8_enabled: bool = False,
    ) -> None:
        del tp_mesh, input_layout, output_layout, use_local_output, float8_enabled
        raise NotImplementedError("tensor parallelism is not implemented for Cubit v1")

    def apply_cp(
        self,
        cp_mesh: DeviceMesh,
        ring: Optional[RingContextParallelStyle] = None,
        uly: Optional[UlyssesContextParallelStyle] = None,
    ) -> None:
        del cp_mesh, ring, uly
        raise NotImplementedError("context parallelism is not implemented for Cubit v1")

    def num_flops_per_token(self, seq_len: int) -> int:
        # Dense v1: two score products, two weighted-value products, and a triangular solve.
        param_flops = 6 * sum(parameter.numel() for parameter in self.parameters())
        quadratic_flops = 24 * self.n_heads * self.head_dim * seq_len
        return param_flops + quadratic_flops
