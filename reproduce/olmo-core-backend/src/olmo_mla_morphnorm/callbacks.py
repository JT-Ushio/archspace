from dataclasses import dataclass
from typing import Dict

from olmo_core.train.callbacks import WandBCallback


@dataclass
class MaxLogitsWandBCallback(WandBCallback):
    """Log max-attention-logit metrics to W&B at a separate interval."""

    max_logits_log_interval: int = 1

    def __post_init__(self) -> None:
        if self.max_logits_log_interval < 1:
            raise ValueError("max_logits_log_interval must be at least 1")

    def log_metrics(self, step: int, metrics: Dict[str, float]) -> None:
        if step % self.max_logits_log_interval != 0:
            metrics = {
                name: value for name, value in metrics.items() if "attention max logit" not in name
            }
        super().log_metrics(step, metrics)
