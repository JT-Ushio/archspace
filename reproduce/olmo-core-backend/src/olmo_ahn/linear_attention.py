import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed.tensor import Placement

from olmo_core.config import DType, StrEnum
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention.base import SequenceMixer, SequenceMixerConfig
from olmo_core.nn.attention.recurrent import GatedDeltaNet, GatedDeltaNetConfig
from olmo_core.nn.attention.ring import (
    RingContextParallelStyle,
    UlyssesContextParallelStyle,
)
from olmo_core.nn.buffer_cache import BufferCache

if TYPE_CHECKING:
    from olmo_core.nn.transformer.init import InitMethod


class LinearAttentionType(StrEnum):
    """Linear sequence mixers supported by the AHN recipes."""

    gdn = "gdn"
    kda = "kda"
    gdn2 = "gdn2"


class PackedGatedDeltaNet(GatedDeltaNet):
    """Flatten batched packed documents before invoking OLMo's native GDN."""

    def forward(
        self,
        x: torch.Tensor,
        cu_doc_lens: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if cu_doc_lens is None or x.shape[0] == 1:
            return super().forward(x, cu_doc_lens=cu_doc_lens, **kwargs)

        batch_size, sequence_length, hidden_size = x.shape
        flattened = x.reshape(1, batch_size * sequence_length, hidden_size)
        output = super().forward(flattened, cu_doc_lens=cu_doc_lens, **kwargs)
        return output.reshape(batch_size, sequence_length, hidden_size)


class FLALinearAttention(SequenceMixer):
    """Adapt an FLA KDA or GDN2 layer to OLMo-core's sequence-mixer API."""

    def __init__(
        self,
        *,
        attention_type: LinearAttentionType,
        d_model: int,
        n_heads: int,
        n_v_heads: int,
        head_dim: int,
        expand_v: float,
        use_short_conv: bool,
        allow_neg_eigval: bool,
        conv_size: int,
        conv_bias: bool,
        norm_eps: float,
        safe_gate: bool,
        lower_bound: Optional[float],
        layer_idx: int,
        dtype: torch.dtype,
        init_device: str,
    ) -> None:
        super().__init__()
        if dtype != torch.float32:
            raise OLMoConfigurationError(
                "The FLA KDA/GDN2 adapters currently require float32 model parameters; "
                "use HSDP param_dtype for mixed-precision training"
            )

        try:
            from fla.layers import GatedDeltaNet2, KimiDeltaAttention
        except ImportError as exc:
            raise ImportError(
                "KDA and GDN2 require the pinned flash-linear-attention dependency. "
                "Install this package with its training dependencies on the GPU server."
            ) from exc

        self.attention_type = LinearAttentionType(attention_type)
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_v_heads = n_v_heads
        self.head_dim = head_dim
        self.expand_v = expand_v
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size

        common_kwargs = dict(
            hidden_size=d_model,
            expand_v=expand_v,
            head_dim=head_dim,
            num_heads=n_heads,
            num_v_heads=n_v_heads,
            mode="chunk",
            use_short_conv=use_short_conv,
            allow_neg_eigval=allow_neg_eigval,
            conv_size=conv_size,
            conv_bias=conv_bias,
            layer_idx=layer_idx,
            norm_eps=norm_eps,
        )
        with torch.device(init_device):
            if self.attention_type == LinearAttentionType.kda:
                self.layer = KimiDeltaAttention(
                    **common_kwargs,
                    safe_gate=safe_gate,
                    lower_bound=lower_bound,
                )
            elif self.attention_type == LinearAttentionType.gdn2:
                self.layer = GatedDeltaNet2(**common_kwargs)
            else:
                raise OLMoConfigurationError(
                    "FLALinearAttention only adapts KDA and GDN2; GDN uses OLMo-core's "
                    "native GatedDeltaNet"
                )

    def forward(
        self,
        x: torch.Tensor,
        cu_doc_lens: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        if cu_doc_lens is None or x.shape[0] == 1:
            output, _, _ = self.layer(hidden_states=x, cu_seqlens=cu_doc_lens)
            return output

        batch_size, sequence_length, hidden_size = x.shape
        flattened = x.reshape(1, batch_size * sequence_length, hidden_size)
        output, _, _ = self.layer(hidden_states=flattened, cu_seqlens=cu_doc_lens)
        return output.reshape(batch_size, sequence_length, hidden_size)

    def apply_tp(
        self,
        tp_mesh: DeviceMesh,
        input_layout: Optional[Placement] = None,
        output_layout: Optional[Placement] = None,
        use_local_output: bool = True,
        float8_enabled: bool = False,
    ) -> None:
        del tp_mesh, input_layout, output_layout, use_local_output, float8_enabled
        raise NotImplementedError("Tensor parallelism is not implemented for KDA/GDN2")

    def apply_cp(
        self,
        cp_mesh: DeviceMesh,
        ring: Optional[RingContextParallelStyle] = None,
        uly: Optional[UlyssesContextParallelStyle] = None,
    ) -> None:
        del cp_mesh, ring, uly
        raise NotImplementedError("Context parallelism is not implemented for KDA/GDN2")

    @torch.no_grad()
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
        from olmo_core.nn.transformer.init import InitMethod, init_linear

        del d_model, block_idx, num_blocks
        if init_method != InitMethod.normal:
            raise OLMoConfigurationError(
                "The KDA/GDN2 adapters currently support only OLMo InitMethod.normal"
            )

        for module in self.layer.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                init_linear(module, std=std, generator=generator)

        if self.attention_type == LinearAttentionType.kda and self.layer.safe_gate:
            self.layer.A_log.zero_()
        else:
            self.layer.A_log.copy_(
                nn.init.uniform_(self.layer.A_log, a=1, b=16, generator=generator).log()
            )

        dt_min, dt_max, dt_init_floor = 0.001, 0.1, 1e-4
        dt = torch.exp(
            nn.init.uniform_(self.layer.dt_bias, generator=generator)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        self.layer.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def num_flops_per_token(self, seq_len: int) -> int:
        del seq_len
        linear_flops = 2 * sum(
            module.weight.numel()
            for module in self.layer.modules()
            if isinstance(module, nn.Linear)
        )
        convolution_flops = 0
        if self.use_short_conv:
            key_dim = self.n_heads * self.head_dim
            value_dim = self.n_v_heads * int(self.head_dim * self.expand_v)
            convolution_flops = 2 * self.conv_size * (2 * key_dim + value_dim)
        state_size = self.n_v_heads * self.head_dim * int(self.head_dim * self.expand_v)
        recurrent_flops = 8 * state_size
        return linear_flops + convolution_flops + recurrent_flops


@SequenceMixerConfig.register("ahn_linear_attention")
@dataclass
class LinearAttentionConfig(SequenceMixerConfig[SequenceMixer]):
    """Serializable switch between GDN, KDA, and GDN2."""

    attention_type: LinearAttentionType = LinearAttentionType.gdn
    n_heads: int = 16
    n_v_heads: Optional[int] = None
    head_dim: Optional[int] = None
    expand_v: Optional[float] = None
    use_short_conv: bool = True
    allow_neg_eigval: bool = False
    conv_size: int = 4
    conv_bias: bool = False
    norm_eps: float = 1e-5
    safe_gate: bool = False
    lower_bound: Optional[float] = None
    dtype: DType = DType.float32

    def __post_init__(self, registered_type: Optional[str] = None) -> None:
        del registered_type
        self.attention_type = LinearAttentionType(self.attention_type)
        self.dtype = DType(self.dtype)
        if self.n_heads <= 0:
            raise OLMoConfigurationError("n_heads must be positive")
        if self.n_v_heads is not None:
            if self.n_v_heads < self.n_heads or self.n_v_heads % self.n_heads != 0:
                raise OLMoConfigurationError(
                    "n_v_heads must be greater than or equal to n_heads and divisible by it"
                )
        if self.head_dim is not None and self.head_dim <= 0:
            raise OLMoConfigurationError("head_dim must be positive")
        if self.expand_v is not None and self.expand_v <= 0:
            raise OLMoConfigurationError("expand_v must be positive")
        if self.conv_size <= 0:
            raise OLMoConfigurationError("conv_size must be positive")
        if self.attention_type == LinearAttentionType.gdn and not self.use_short_conv:
            raise OLMoConfigurationError("OLMo-core's native GDN requires use_short_conv=true")
        if self.safe_gate:
            if self.attention_type != LinearAttentionType.kda:
                raise OLMoConfigurationError("safe_gate is only supported by KDA")
            if self.lower_bound is None:
                raise OLMoConfigurationError("safe_gate requires lower_bound")
        if self.lower_bound is not None:
            if self.attention_type != LinearAttentionType.kda:
                raise OLMoConfigurationError("lower_bound is only supported by KDA")
            if not self.safe_gate:
                raise OLMoConfigurationError("lower_bound requires safe_gate=true")
            if not -5 <= self.lower_bound < 0:
                raise OLMoConfigurationError("lower_bound must be in FLA's safe range [-5, 0)")

    def _resolved_dimensions(self, d_model: int) -> tuple[int, int, int, float]:
        if self.head_dim is None and d_model % self.n_heads != 0:
            raise OLMoConfigurationError("d_model must be divisible by n_heads")
        n_v_heads = self.n_v_heads or self.n_heads
        head_dim = self.head_dim or d_model // self.n_heads
        expand_v = self.expand_v
        if expand_v is None:
            expand_v = 2.0 if self.attention_type == LinearAttentionType.gdn else 1.0
        head_v_dim = int(head_dim * expand_v)
        if not math.isclose(head_dim * expand_v, head_v_dim, rel_tol=1e-5):
            raise OLMoConfigurationError("head_dim * expand_v must be an integer")
        return n_v_heads, head_dim, head_v_dim, expand_v

    def _gdn_config(self, d_model: int) -> GatedDeltaNetConfig:
        n_v_heads, head_dim, _, expand_v = self._resolved_dimensions(d_model)
        return GatedDeltaNetConfig(
            n_heads=self.n_heads,
            n_v_heads=n_v_heads,
            head_dim=head_dim,
            expand_v=expand_v,
            allow_neg_eigval=self.allow_neg_eigval,
            conv_size=self.conv_size,
            conv_bias=self.conv_bias,
            norm_eps=self.norm_eps,
            dtype=self.dtype,
        )

    def num_params(self, d_model: int) -> int:
        if self.attention_type == LinearAttentionType.gdn:
            return self._gdn_config(d_model).num_params(d_model)

        n_v_heads, head_dim, head_v_dim, expand_v = self._resolved_dimensions(d_model)
        key_dim = self.n_heads * head_dim
        value_dim = n_v_heads * head_v_dim
        conv_params = 0
        if self.use_short_conv:
            conv_params = self.conv_size * (2 * key_dim + value_dim)
            if self.conv_bias:
                conv_params += 2 * key_dim + value_dim

        if self.attention_type == LinearAttentionType.kda:
            gate_dim = n_v_heads * head_dim
            return (
                2 * d_model * key_dim
                + d_model * value_dim
                + conv_params
                + d_model * head_v_dim
                + head_v_dim * gate_dim
                + d_model * n_v_heads
                + n_v_heads
                + gate_dim
                + d_model * head_v_dim
                + head_v_dim * value_dim
                + value_dim
                + head_v_dim
                + value_dim * d_model
            )

        assert self.attention_type == LinearAttentionType.gdn2
        return (
            2 * d_model * key_dim
            + d_model * value_dim
            + conv_params
            + d_model * head_v_dim
            + head_v_dim * key_dim
            + d_model * key_dim
            + d_model * value_dim
            + self.n_heads
            + key_dim
            + d_model * head_v_dim
            + head_v_dim * value_dim
            + value_dim
            + head_v_dim
            + value_dim * d_model
        )

    def build(
        self,
        d_model: int,
        *,
        layer_idx: int,
        n_layers: int,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ) -> SequenceMixer:
        if self.attention_type == LinearAttentionType.gdn:
            del layer_idx, n_layers, cache
            n_v_heads, head_dim, _, expand_v = self._resolved_dimensions(d_model)
            return PackedGatedDeltaNet(
                d_model=d_model,
                n_heads=self.n_heads,
                n_v_heads=n_v_heads,
                head_dim=head_dim,
                expand_v=expand_v,
                allow_neg_eigval=self.allow_neg_eigval,
                conv_size=self.conv_size,
                conv_bias=self.conv_bias,
                norm_eps=self.norm_eps,
                dtype=self.dtype.as_pt(),
                init_device=init_device,
            )

        del n_layers, cache
        n_v_heads, head_dim, _, expand_v = self._resolved_dimensions(d_model)
        return FLALinearAttention(
            attention_type=self.attention_type,
            d_model=d_model,
            n_heads=self.n_heads,
            n_v_heads=n_v_heads,
            head_dim=head_dim,
            expand_v=expand_v,
            use_short_conv=self.use_short_conv,
            allow_neg_eigval=self.allow_neg_eigval,
            conv_size=self.conv_size,
            conv_bias=self.conv_bias,
            norm_eps=self.norm_eps,
            safe_gate=self.safe_gate,
            lower_bound=self.lower_bound,
            layer_idx=layer_idx,
            dtype=self.dtype.as_pt(),
            init_device=init_device,
        )
