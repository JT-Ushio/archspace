import functools
from typing import Optional

import torch
import torch.nn as nn

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.transformer.init import InitMethod, init_linear

from .feed_forward import MoSESwiGLU


def install_runtime_hooks() -> None:
    """Teach OLMo's initializer how to initialize every MoSE projection."""
    current = InitMethod.init_feed_forward
    if getattr(current, "_olmo_mose_hook", False):
        return

    @functools.wraps(current)
    def init_feed_forward(
        self: InitMethod,
        module,
        *,
        d_model: int,
        block_idx: int,
        num_blocks: int,
        std: float = 0.02,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        if not isinstance(module, MoSESwiGLU):
            return current(
                self,
                module,
                d_model=d_model,
                block_idx=block_idx,
                num_blocks=num_blocks,
                std=std,
                generator=generator,
            )

        if self != InitMethod.normal:
            raise OLMoConfigurationError(
                "MoSE-SwiGLU currently supports only OLMo InitMethod.normal"
            )
        for projection in module.projection_modules():
            init_linear(projection, std=std, generator=generator)
        for bias in (module.gate_bias, module.up_bias, module.down_bias):
            if bias is not None:
                nn.init.zeros_(bias)

    init_feed_forward._olmo_mose_hook = True  # type: ignore[attr-defined]
    InitMethod.init_feed_forward = init_feed_forward  # type: ignore[method-assign]
