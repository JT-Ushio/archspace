from collections.abc import Iterable
from typing import Any, Union

import torch

from olmo_core.optim import MuonConfig


class SerializableMuonConfig(MuonConfig):
    """Muon config that does not persist generated parameter-group overrides."""

    def build_groups(
        self,
        model: torch.nn.Module,
        strict: bool = True,
    ) -> Union[Iterable[torch.Tensor], list[dict[str, Any]]]:
        try:
            return super().build_groups(model, strict=strict)
        finally:
            self.group_overrides = None
