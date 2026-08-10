import functools
import math
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
        # The linear and nonlinear experts jointly span R = r1 + r2, so both
        # kinds of U use the requested matrix std while every corresponding V
        # uses 1 / sqrt(R).
        gate_up_v_std = 1.0 / math.sqrt(module.r1 + module.r2)
        projection_stds = (
            (module.linear_u, std),
            (module.gate_linear_v, gate_up_v_std),
            (module.up_linear_v, gate_up_v_std),
            (module.nonlinear_u, std),
            (module.gate_nonlinear_v, gate_up_v_std),
            (module.up_nonlinear_v, gate_up_v_std),
        )
        for projection, projection_std in projection_stds:
            if projection is not None:
                init_linear(projection, std=projection_std, generator=generator)

        if module.down_is_mose:
            down_v_std = 1.0 / math.sqrt(module.down_r1 + module.down_r2)
            down_projection_stds = (
                (module.down_linear_u, std),
                (module.down_linear_v, down_v_std),
                (module.down_nonlinear_u, std),
                (module.down_nonlinear_v, down_v_std),
            )
            for projection, projection_std in down_projection_stds:
                if projection is not None:
                    init_linear(projection, std=projection_std, generator=generator)
        else:
            assert module.w_down is not None
            init_linear(module.w_down, std=std, generator=generator)
        for bias in (module.gate_bias, module.up_bias, module.down_bias):
            if bias is not None:
                nn.init.zeros_(bias)

    init_feed_forward._olmo_mose_hook = True  # type: ignore[attr-defined]
    InitMethod.init_feed_forward = init_feed_forward  # type: ignore[method-assign]
