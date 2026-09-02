"""Hugging Face OLMo3 model that preserves the SiTU-Up SwiGLU forward pass."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers.models.olmo3.modeling_olmo3 import (
    Olmo3ForCausalLM,
    Olmo3MLP,
    Olmo3Model,
)

from .configuration_situ_olmo3 import SiTUOlmo3Config


class SiTUOlmo3MLP(Olmo3MLP):
    """Standard OLMo3 MLP weights with SiTU applied only to the Up branch.

    This exactly follows the training implementation::

        gate = gate_proj(x)
        up = up_proj(x)
        hidden = silu(gate) * (beta * tanh(up / beta))
        output = down_proj(hidden)

    OLMo Core evaluates the channel-control portion in fp32 for fp16/bf16
    projection outputs, then casts the product back before ``down_proj``.
    """

    config: SiTUOlmo3Config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        projection_dtype = gate.dtype

        if projection_dtype in (torch.float16, torch.bfloat16):
            gate = gate.float()
            up = up.float()

        beta = self.config.situ_beta_up
        up = beta * torch.tanh(up / beta)
        hidden = F.silu(gate) * up
        return self.down_proj(hidden.to(projection_dtype))


def _install_situ_up_mlp(model: Olmo3Model) -> None:
    """Change only the MLP forward implementation, preserving all parameters."""

    for layer_idx, layer in enumerate(model.layers):
        if not isinstance(layer.mlp, Olmo3MLP):
            raise TypeError(
                f"layer {layer_idx} has unsupported MLP type {type(layer.mlp).__name__}"
            )
        # SiTU adds no parameters. Rebinding the class avoids allocating or copying
        # the three projection matrices and leaves every state-dict key unchanged.
        layer.mlp.__class__ = SiTUOlmo3MLP


class SiTUOlmo3Model(Olmo3Model):
    config_class = SiTUOlmo3Config

    def __init__(self, config: SiTUOlmo3Config) -> None:
        super().__init__(config)
        _install_situ_up_mlp(self)


class SiTUOlmo3ForCausalLM(Olmo3ForCausalLM):
    config_class = SiTUOlmo3Config

    def __init__(self, config: SiTUOlmo3Config) -> None:
        super().__init__(config)
        _install_situ_up_mlp(self.model)


__all__ = [
    "SiTUOlmo3ForCausalLM",
    "SiTUOlmo3MLP",
    "SiTUOlmo3Model",
]
