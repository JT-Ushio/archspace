"""Compare legacy input masks and valid target masks on the same LM forward pass.

Used only by standalone checkpoint evaluation. Live training and upstream
OLMo-core files are not modified. Context/tensor parallel evaluation is rejected.
"""

from types import MethodType

import torch

from olmo_core.data.utils import get_labels
from olmo_core.distributed.utils import all_reduce_value
from olmo_core.eval.lm_evaluator import LMEvaluator
from olmo_core.eval.metrics import MeanMetric


def align_eval_batch(train_module):
    if train_module.cp_enabled or train_module.tp_enabled:
        raise ValueError("This standalone adapter supports data parallelism only")
    original = train_module.eval_batch

    def evaluate(self, batch, labels=None):
        if labels is None:
            labels = get_labels(batch, label_ignore_index=self.label_ignore_index)
        valid_targets = labels.ne(self.label_ignore_index)
        legacy_mask = batch.get("label_mask")
        output = original(batch, labels=labels)
        if output.ce_loss is None or output.ce_loss.shape != valid_targets.shape:
            raise ValueError("Per-token loss and target mask must have identical shapes")
        # Change the scoring mask after forward; model inputs stay unchanged.
        batch["_mose_legacy_label_mask"] = legacy_mask
        batch["label_mask"] = valid_targets
        return output

    train_module.eval_batch = MethodType(evaluate, train_module)


class DualMaskLMEvaluator(LMEvaluator):
    def __init__(self, original):
        super().__init__(
            name=original.name, batches=original.batches,
            labels=list(original.metrics),
            device=original.device, deterministic=original.deterministic,
        )
        self.legacy_metrics = {label: MeanMetric(device=self.device) for label in self.metrics}
        self.corrected_counts = {label: torch.zeros((), dtype=torch.int64, device=self.device)
                                 for label in self.metrics}
        self.legacy_counts = {label: value.clone() for label, value in self.corrected_counts.items()}

    def update_metrics(self, batch, ce_loss, logits):
        if ce_loss is None:
            raise ValueError("Standalone LM evaluation requires per-token CE")
        if ce_loss.shape != batch["label_mask"].shape:
            raise ValueError("CE shape differs from aligned target mask")
        super().update_metrics(batch, ce_loss, logits)
        legacy_mask = batch["_mose_legacy_label_mask"]
        for idx, (metadata, values) in enumerate(zip(batch["metadata"], ce_loss)):
            label = metadata["label"]
            old_values = values if legacy_mask is None else values.masked_select(legacy_mask[idx])
            self.legacy_metrics[label].update(old_values)
            self.legacy_counts[label] += old_values.numel()
            self.corrected_counts[label] += values.masked_select(batch["label_mask"][idx]).numel()

    def compute_metrics(self):
        # all_reduce_value reduces tensors in place. Keep local accumulators
        # unchanged so collecting metrics twice cannot multiply token counts.
        result = {}
        for label in sorted(self.metrics):
            corrected = self.metrics[label]
            legacy = self.legacy_metrics[label]
            corrected_sum = all_reduce_value(corrected.weighted_sum.clone(), device=self.device)
            corrected_weight = all_reduce_value(corrected.weight.clone(), device=self.device)
            legacy_sum = all_reduce_value(legacy.weighted_sum.clone(), device=self.device)
            legacy_weight = all_reduce_value(legacy.weight.clone(), device=self.device)
            result[f"{label}/CE loss"] = corrected_sum / corrected_weight
            result[f"{label}/PPL"] = torch.exp(result[f"{label}/CE loss"])
            result[f"legacy/{label}/CE loss"] = legacy_sum / legacy_weight
            result[f"{label}/scored_tokens"] = all_reduce_value(
                self.corrected_counts[label].clone(), device=self.device)
            result[f"legacy/{label}/scored_tokens"] = all_reduce_value(
                self.legacy_counts[label].clone(), device=self.device)
        return result

    def reset_metrics(self):
        super().reset_metrics()
        for metric in self.legacy_metrics.values():
            metric.reset()
        for counts in (self.corrected_counts, self.legacy_counts):
            for value in counts.values():
                value.zero_()
