from typing import Dict, Optional, Tuple

import torch

from olmo_core.nn.transformer import MoETransformer, Transformer

from .max_logits import MaxLogitsFlashAttention3Backend
from .module import MLAAttention


def _get_max_logits_backend(block) -> Optional[MaxLogitsFlashAttention3Backend]:
    attention = getattr(block, "attention", None)
    backend = getattr(attention, "backend", None)
    return backend if isinstance(backend, MaxLogitsFlashAttention3Backend) else None


def _collect_max_logits_metrics(
    model: Transformer, reset: bool
) -> Dict[str, Tuple[torch.Tensor, object]]:
    from olmo_core.train.common import ReduceType

    output: Dict[str, Tuple[torch.Tensor, object]] = {}
    model_max: Optional[torch.Tensor] = None
    for block_key, block in model.blocks.items():
        backend = _get_max_logits_backend(block)
        if backend is None:
            continue
        max_logits = backend.get_max_logits(reset=reset)
        layer_idx = int(getattr(block, "block_idx", block_key))
        layer_max = max_logits.amax()
        output[f"block {layer_idx:02d}/attention max logit"] = (
            layer_max,
            ReduceType.max,
        )
        if backend.log_per_head:
            for head_idx, head_max in enumerate(max_logits.unbind()):
                output[f"block {layer_idx:02d}/attention max logit/head {head_idx:02d}"] = (
                    head_max,
                    ReduceType.max,
                )
        model_max = layer_max if model_max is None else torch.maximum(model_max, layer_max)
    if model_max is not None:
        output["attention max logit"] = (model_max, ReduceType.max)
    return output


def _reset_max_logits_metrics(model: Transformer) -> None:
    for block in model.blocks.values():
        backend = _get_max_logits_backend(block)
        if backend is not None:
            backend.reset_max_logits()


def _install_model_hooks(model_class) -> None:
    current_post_batch = model_class.post_batch
    if not getattr(current_post_batch, "_olmo_mla_morphnorm_hook", False):

        def post_batch(
            model: Transformer,
            dry_run: bool = False,
            _original_post_batch=current_post_batch,
        ) -> None:
            _original_post_batch(model, dry_run=dry_run)
            for module in model.modules():
                if isinstance(module, MLAAttention):
                    module.post_batch(dry_run=dry_run)

        setattr(post_batch, "_olmo_mla_morphnorm_hook", True)
        setattr(model_class, "post_batch", post_batch)

    current_compute = model_class.compute_auxiliary_metrics
    if not getattr(current_compute, "_olmo_max_logits_hook", False):

        def compute_auxiliary_metrics(
            model: Transformer,
            reset: bool = True,
            _original_compute=current_compute,
        ):
            output = _original_compute(model, reset=reset)
            output.update(_collect_max_logits_metrics(model, reset=reset))
            return output

        setattr(compute_auxiliary_metrics, "_olmo_max_logits_hook", True)
        setattr(model_class, "compute_auxiliary_metrics", compute_auxiliary_metrics)

    current_reset = model_class.reset_auxiliary_metrics
    if not getattr(current_reset, "_olmo_max_logits_hook", False):

        def reset_auxiliary_metrics(
            model: Transformer,
            _original_reset=current_reset,
        ) -> None:
            _original_reset(model)
            _reset_max_logits_metrics(model)

        setattr(reset_auxiliary_metrics, "_olmo_max_logits_hook", True)
        setattr(model_class, "reset_auxiliary_metrics", reset_auxiliary_metrics)


def install_runtime_hooks() -> None:
    """Install OLMo lifecycle hooks for MorphNorm state and max-logit metrics."""
    for model_class in (Transformer, MoETransformer):
        _install_model_hooks(model_class)
