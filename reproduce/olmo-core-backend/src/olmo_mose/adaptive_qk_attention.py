"""Experimental per-layer interpolation between Identity and SiLU in Q/K."""

import math
import re
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import DTensor

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.train.callbacks import Callback, CallbackConfig

from .attention import (
    LowRankAttention,
    LowRankAttentionConfig,
    LowRankAttentionSharingScope,
    NonlinearLowRankProjection,
)
from .feed_forward import MoSENonlinearity


class AdaptiveNonlinearityProjection(NonlinearLowRankProjection):
    """Reuse the original matrices; introduce one trainable scalar per projection."""

    def __init__(self, projection: NonlinearLowRankProjection, alpha_init: float):
        nn.Module.__init__(self)
        if projection.nonlinear_norm is not None or not math.isfinite(alpha_init):
            raise OLMoConfigurationError("adaptive projection requires no latent norm and finite alpha")
        self.in_features = projection.in_features
        self.out_features = projection.out_features
        self.rank = projection.rank
        self.nonlinearity = projection.nonlinearity
        self.rms_norm_learnable_weight = False
        self.nonlinear_norm = None
        self.v, self.u = projection.v, projection.u
        self.alpha_init = float(alpha_init)
        # One-element vectors work with FSDP sharding and enter Muon's AdamW group.
        self.alpha = nn.Parameter(
            torch.full((1,), self.alpha_init, dtype=torch.float32, device=self.v.weight.device)
        )

    @torch.no_grad()
    def reset_alpha(self):
        self.alpha.fill_(self.alpha_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.v(x)
        alpha = self.alpha.float()
        # This is a + alpha*(SiLU(a)-a), with exact Identity/SiLU endpoints.
        # Compute SiLU in the original activation dtype, then blend in FP32.
        blended = (1.0 - alpha) * latent.float() + alpha * F.silu(latent).float()
        return self.u(blended.to(latent.dtype))


class AdaptiveQKAttention(LowRankAttention):
    def __init__(self, *, q_alpha_init: float = 0.0, k_alpha_init: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        if self.share_scope != LowRankAttentionSharingScope.none:
            raise OLMoConfigurationError("adaptive Q/K requires independent Attention factors")
        self.w_q = AdaptiveNonlinearityProjection(self.w_q, q_alpha_init)
        self.w_k = AdaptiveNonlinearityProjection(self.w_k, k_alpha_init)

    def init_weights(self, **kwargs):
        super().init_weights(**kwargs)
        # The trainer materializes meta parameters with to_empty before initialization.
        self.w_q.reset_alpha()
        self.w_k.reset_alpha()


@dataclass
class AdaptiveQKAttentionConfig(LowRankAttentionConfig):
    q_alpha_init: float = 0.0
    k_alpha_init: float = 1.0

    def __post_init__(self, registered_name=None):
        super().__post_init__(registered_name)
        if self.share_scope != LowRankAttentionSharingScope.none:
            raise OLMoConfigurationError("adaptive Q/K requires independent Attention factors")
        if self.rms_norm_learnable_weight or (
            self.q_nonlinearity != MoSENonlinearity.identity
            or self.k_nonlinearity != MoSENonlinearity.silu
        ):
            raise OLMoConfigurationError("adaptive Q/K starts from C2: Q Identity, K SiLU")
        if not all(math.isfinite(a) for a in (self.q_alpha_init, self.k_alpha_init)):
            raise OLMoConfigurationError("alpha initialization must be finite")

    def num_params(self, d_model: int) -> int:
        return super().num_params(d_model) + 2

    def build(self, d_model, *, layer_idx, n_layers, init_device="cpu", cache=None):
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        kwargs.pop("name")
        sliding = kwargs.pop("sliding_window", None)
        if sliding is not None and sliding.should_use_swa(layer_idx, n_layers):
            kwargs["window_size"] = sliding.get_window_size(layer_idx, n_layers)
        elif kwargs.get("rope") is not None and kwargs["rope"].no_global_rope:
            kwargs["rope"] = None
        kwargs.update(
            dtype=kwargs.pop("dtype").as_pt(), d_model=d_model,
            init_device=init_device, cache=cache,
        )
        return AdaptiveQKAttention(**kwargs)


class QKAlphaMetricsCallback(Callback):
    def __init__(self, interval: int = 100):
        if interval <= 0:
            raise ValueError("alpha logging interval must be positive")
        self.interval = interval

    def post_step(self):
        if self.step % self.interval:
            return
        values = {"q": [], "k": []}
        for name, parameter in self.trainer.train_module.model.named_parameters():
            match = re.search(r"blocks\.(\d+)\..*w_([qk])\.alpha$", name)
            if match is None:
                continue
            layer, target = match.groups()
            value = parameter.detach()
            # All ranks must enter this collective, including ranks with empty shards.
            if isinstance(value, DTensor):
                value = value.full_tensor()
            value = value.float().reshape(())
            self.trainer.record_metric(f"qk_alpha/layer{layer}/{target}", value)
            values[target].append(value)
        if not values["q"] or len(values["q"]) != len(values["k"]):
            raise RuntimeError("expected one Q and one K alpha for every gated layer")
        for target, scalars in values.items():
            stacked = torch.stack(scalars)
            for statistic in ("mean", "min", "max"):
                self.trainer.record_metric(
                    f"qk_alpha/{target}_{statistic}", getattr(stacked, statistic)()
                )


@dataclass
class QKAlphaMetricsCallbackConfig(CallbackConfig):
    interval: int = 100

    def build(self, trainer):
        del trainer
        return QKAlphaMetricsCallback(self.interval)
