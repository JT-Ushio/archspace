from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Union

import torch

from olmo_core.optim import MuonConfig
from olmo_core.optim.config import INITIAL_LR_FIELD


@dataclass
class SerializableMuonConfig(MuonConfig):
    """Muon config that clears its temporary parameter groups after construction."""

    # Legacy checkpoints may contain flatten=True, including pre-train step-0 saves.
    # Preserve the current recipe's value when restoring optimizer parameter groups.
    fixed_fields: tuple[str, ...] = (INITIAL_LR_FIELD, "flatten")

    def build_groups(
        self,
        model: torch.nn.Module,
        strict: bool = True,
    ) -> Union[Iterable[torch.Tensor], list[dict[str, Any]]]:
        try:
            return super().build_groups(model, strict=strict)
        finally:
            self.group_overrides = None
