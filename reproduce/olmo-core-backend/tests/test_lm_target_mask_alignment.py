"""Target padding and prefix masks must select exactly the labels scored by CE."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def adapter():
    path = Path(__file__).resolve().parents[1] / "scripts/analysis/Corrected_LM_Evaluation/alignment.py"
    spec = importlib.util.spec_from_file_location("lm_mask_adapter_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("mask,loss,expected_old,expected_new,old_n,new_n", [
    ([True, True, True, False], [2., 4., 0., 0.], 2., 3., 3, 2),
    ([False, True, True, False], [2., 4., 0., 0.], 2., 3., 2, 2),
    (None, [2., 4., 6., 0.], 3., 4., 4, 3),
])
def test_masks_and_missing_final_target(mask, loss, expected_old, expected_new, old_n, new_n):
    module = adapter()
    batch = {"input_ids": torch.tensor([[1, 2, 3, 4]]), "metadata": [{"label": "x"}]}
    if mask is not None:
        batch["label_mask"] = torch.tensor([mask])
    before_mask = batch.get("label_mask")
    def forward(batch, labels):
        assert batch.get("label_mask") is before_mask
        assert labels.shape == (1, 4)
        return SimpleNamespace(ce_loss=torch.tensor([loss]))
    train_module = SimpleNamespace(cp_enabled=False, tp_enabled=False,
                                   label_ignore_index=-100, eval_batch=forward)
    module.align_eval_batch(train_module)
    output = train_module.eval_batch(batch)
    original = module.LMEvaluator(name="lm", batches=[], labels=["x"], device=torch.device("cpu"))
    evaluator = module.DualMaskLMEvaluator(original)
    evaluator.update_metrics(batch, output.ce_loss, None)
    result = evaluator.compute_metrics()
    assert result["x/CE loss"].item() == pytest.approx(expected_new)
    assert result["legacy/x/CE loss"].item() == pytest.approx(expected_old)
    assert result["x/scored_tokens"].item() == new_n
    assert result["legacy/x/scored_tokens"].item() == old_n
    evaluator.reset_metrics()
    assert evaluator.corrected_counts["x"].item() == 0


def test_no_silent_use_with_sequence_sharding():
    module = adapter()
    with pytest.raises(ValueError, match="data parallelism only"):
        module.align_eval_batch(SimpleNamespace(cp_enabled=True, tp_enabled=False))


def test_repeated_distributed_collection_preserves_local_accumulators(monkeypatch):
    module = adapter()
    original = module.LMEvaluator(name="lm", batches=[], labels=["x"], device=torch.device("cpu"))
    evaluator = module.DualMaskLMEvaluator(original)
    batch = {"metadata": [{"label": "x"}], "label_mask": torch.tensor([[True, True, False]]),
             "_mose_legacy_label_mask": torch.tensor([[True, True, True]])}
    evaluator.update_metrics(batch, torch.tensor([[2., 4., 0.]]), None)
    # Emulate the actual reducer's in-place behavior on two identical ranks.
    monkeypatch.setattr(module, "all_reduce_value", lambda value, device: value.mul_(2))
    for _ in range(2):
        result = evaluator.compute_metrics()
        assert result["x/scored_tokens"].item() == 4
        assert result["legacy/x/scored_tokens"].item() == 6
        assert result["x/CE loss"].item() == 3.0
    assert evaluator.corrected_counts["x"].item() == 2
    assert evaluator.metrics["x"].weighted_sum.item() == 6.0
