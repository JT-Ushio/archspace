"""Hugging Face configuration for OLMo3 with SiTU on the Up branch."""

from __future__ import annotations

import math

from transformers.models.olmo3.configuration_olmo3 import Olmo3Config


class SiTUOlmo3Config(Olmo3Config):
    """OLMo3 configuration with fixed, parameter-free SiTU-Up control."""

    model_type = "situ_olmo3"

    def __init__(
        self,
        situ_beta_up: float = 25.0,
        situ_control_scope: str = "up",
        **kwargs,
    ) -> None:
        if isinstance(situ_beta_up, bool) or not isinstance(situ_beta_up, (int, float)):
            raise TypeError("situ_beta_up must be a real number")
        if not math.isfinite(situ_beta_up) or situ_beta_up <= 0:
            raise ValueError("situ_beta_up must be finite and positive")
        if situ_control_scope != "up":
            raise ValueError("this model implementation only supports situ_control_scope='up'")

        super().__init__(**kwargs)
        self.situ_beta_up = float(situ_beta_up)
        self.situ_control_scope = situ_control_scope


__all__ = ["SiTUOlmo3Config"]
