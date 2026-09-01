from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Union

import torch

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.transformer import Transformer
from olmo_core.optim import MuonConfig
from olmo_core.optim.config import OptimGroupOverride


@dataclass
class SerializableMuonConfig(MuonConfig):
    """Muon config that supports depthwise-conv weights and stays serializable."""

    flatten: bool = True

    def categorize_parameters(self, model: torch.nn.Module) -> dict[str, list[str]]:
        if not isinstance(model, Transformer):
            raise OLMoConfigurationError("Muon requires an OLMo Transformer")

        embed_params = [
            f"embeddings.{name}"
            for name, parameter in model.embeddings.named_parameters()
            if parameter.ndim == 2
        ]
        matrix_params = [
            f"blocks.{name}"
            for name, parameter in model.blocks.named_parameters()
            if parameter.ndim == 2 or (self.flatten and parameter.ndim > 2)
        ]
        vector_params = [
            f"blocks.{name}"
            for name, parameter in model.blocks.named_parameters()
            if parameter.ndim < 2
        ]
        vector_params += [
            f"lm_head.{name}"
            for name, parameter in model.lm_head.named_parameters()
            if parameter.ndim < 2
        ]
        lm_head_params = [
            f"lm_head.{name}"
            for name, parameter in model.lm_head.named_parameters()
            if parameter.ndim == 2
        ]

        unsupported = [
            name
            for name, parameter in model.named_parameters()
            if parameter.ndim > 2 and not self.flatten
        ]
        if unsupported:
            raise OLMoConfigurationError(
                "linear-attention convolution weights require Muon flatten=true: "
                + ", ".join(unsupported)
            )

        categorized = embed_params + matrix_params + vector_params + lm_head_params
        all_params = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        if set(categorized) != all_params or len(categorized) != len(set(categorized)):
            raise OLMoConfigurationError("failed to categorize every parameter exactly once")
        return {
            "embed": embed_params,
            "matrix": matrix_params,
            "vector": vector_params,
            "lm_head": lm_head_params,
        }

    def default_group_overrides(self, model: torch.nn.Module) -> list[OptimGroupOverride]:
        overrides = super().default_group_overrides(model)
        no_decay = {
            name
            for name, _ in model.named_parameters()
            if name.endswith(".A_log") or name.endswith(".dt_bias")
        }
        if not no_decay:
            return overrides

        for override in overrides:
            override.params = [name for name in override.params if name not in no_decay]
        overrides.append(
            OptimGroupOverride(
                params=sorted(no_decay),
                opts={"algorithm": "adamw", "weight_decay": 0.0},
            )
        )
        return overrides

    def build_groups(
        self,
        model: torch.nn.Module,
        strict: bool = True,
    ) -> Union[Iterable[torch.Tensor], list[dict[str, Any]]]:
        try:
            return super().build_groups(model, strict=strict)
        finally:
            self.group_overrides = None
