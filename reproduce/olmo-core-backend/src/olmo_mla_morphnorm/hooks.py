from functools import wraps
from typing import Dict, Literal, Optional, Tuple, Union

import torch

from olmo_core.nn.transformer import MoETransformer, Transformer
from olmo_core.nn.transformer.block import ReorderedNormTransformerBlock
from olmo_core.utils import mark_dynamic, move_to_device

from .max_logits import MaxLogitsFlashAttention3Backend
from .module import MLAAttention


def _get_mla_attention(module) -> Optional[MLAAttention]:
    if isinstance(module, MLAAttention):
        return module
    return next((child for child in module.modules() if isinstance(child, MLAAttention)), None)


def _uses_ckv_layer_residual(model: Transformer) -> bool:
    blocks = list(model.blocks.values())
    attentions = [_get_mla_attention(block) for block in blocks]
    enabled = [
        attention
        for attention in attentions
        if attention is not None and attention.use_ckv_layer_residual
    ]
    if not enabled:
        return False
    if len(enabled) != len(attentions):
        raise RuntimeError(
            "use_ckv_layer_residual must be enabled on the MLA attention in every transformer "
            "layer"
        )
    if any(
        not any(isinstance(module, ReorderedNormTransformerBlock) for module in block.modules())
        for block in blocks
    ):
        raise RuntimeError(
            "use_ckv_layer_residual currently requires OLMo reordered-norm transformer blocks"
        )
    latent_dims = {attention.kv_lora_rank for attention in enabled}
    if len(latent_dims) != 1:
        raise RuntimeError(
            "Cross-layer cKV residual requires the same kv_lora_rank in every transformer layer; "
            f"got {sorted(latent_dims)}"
        )
    return True


def _forward_with_ckv_layer_residual(
    model: Transformer,
    input_ids: torch.Tensor,
    *,
    input_embeddings: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    loss_reduction: Literal["mean", "sum", "none"] = "mean",
    z_loss_multiplier: Optional[float] = None,
    loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
    return_logits: Optional[bool] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    **kwargs,
):
    if input_embeddings is not None and model._cp_load_balancer is not None:
        raise RuntimeError(
            "`input_embeddings` is not supported with context parallelism: `_prepare_inputs` "
            "shards `input_ids`/`labels`/RoPE while `input_embeddings` stays full-size, which "
            "would misalign the hidden states."
        )

    (
        input_ids,
        labels,
        all_block_kwargs,
        per_block_kwargs,
        lm_head_kwargs,
    ) = model._prepare_inputs(
        input_ids,
        labels,
        ignore_index=ignore_index,
        loss_reduction=loss_reduction,
        z_loss_multiplier=z_loss_multiplier,
        loss_div_factor=loss_div_factor,
        return_logits=return_logits,
        logits_to_keep=logits_to_keep,
        **kwargs,
    )

    if input_embeddings is not None:
        hidden_states = move_to_device(input_embeddings, model.device)
    else:
        hidden_states = (
            model.embeddings(input_ids) if model.embeddings is not None else input_ids
        )
        if model.embeddings is not None and model.embed_scale is not None:
            hidden_states = hidden_states * model.embed_scale
        if model.embedding_norm is not None:
            hidden_states = model.embedding_norm(hidden_states)

    prev_raw_ckv: Optional[torch.Tensor] = None
    for block_key, block in model.blocks.items():
        block_idx = int(block_key)
        block_kwargs = per_block_kwargs.get(block_idx, {})
        if model.compile_enabled:
            mark_dynamic(hidden_states, (0, 1), strict=False)
            if prev_raw_ckv is not None:
                mark_dynamic(prev_raw_ckv, (0, 1), strict=False)
        block_output = block(
            hidden_states,
            prev_raw_ckv=prev_raw_ckv,
            **all_block_kwargs,
            **block_kwargs,
        )
        if not isinstance(block_output, tuple) or len(block_output) != 2:
            raise RuntimeError(
                "A transformer block with use_ckv_layer_residual enabled must return "
                "(hidden_states, raw_ckv)"
            )
        hidden_states, prev_raw_ckv = block_output

    if model.lm_head is not None:
        if model.compile_enabled:
            mark_dynamic(hidden_states, (0, 1), strict=False)
            if labels is not None:
                mark_dynamic(labels, (0, 1), strict=False)
        if labels is not None:
            lm_head_kwargs["labels"] = labels
        return model.lm_head(hidden_states, **lm_head_kwargs)
    return hidden_states


def _install_ckv_block_hook() -> None:
    current_forward = ReorderedNormTransformerBlock.forward
    if getattr(current_forward, "_olmo_ckv_layer_residual_hook", False):
        return

    @wraps(current_forward)
    def forward(
        block: ReorderedNormTransformerBlock,
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        prev_raw_ckv: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        attention = _get_mla_attention(block.attention)
        if attention is None or not attention.use_ckv_layer_residual:
            return current_forward(block, x, loss_div_factor=loss_div_factor, **kwargs)

        del loss_div_factor
        attention_output = block.attention(x, prev_raw_ckv=prev_raw_ckv, **kwargs)
        if not isinstance(attention_output, tuple) or len(attention_output) != 2:
            raise RuntimeError(
                "MLAAttention with use_ckv_layer_residual enabled must return "
                "(attention_output, raw_ckv)"
            )
        projected_states, raw_ckv = attention_output
        hidden_states = block.attention_residual_stream(
            x, block.attention_norm(projected_states)
        )
        hidden_states = block.feed_forward_residual_stream(
            hidden_states,
            block.feed_forward_norm(block.feed_forward(hidden_states)),
        )
        return hidden_states, raw_ckv

    setattr(forward, "_olmo_ckv_layer_residual_hook", True)
    setattr(ReorderedNormTransformerBlock, "forward", forward)


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
    current_forward = model_class.forward
    if not getattr(current_forward, "_olmo_ckv_layer_residual_hook", False):

        @wraps(current_forward)
        def forward(
            model: Transformer,
            input_ids: torch.Tensor,
            *,
            input_embeddings: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            ignore_index: int = -100,
            loss_reduction: Literal["mean", "sum", "none"] = "mean",
            z_loss_multiplier: Optional[float] = None,
            loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
            return_logits: Optional[bool] = None,
            logits_to_keep: Union[int, torch.Tensor] = 0,
            **kwargs,
        ):
            if _uses_ckv_layer_residual(model):
                return _forward_with_ckv_layer_residual(
                    model,
                    input_ids,
                    input_embeddings=input_embeddings,
                    labels=labels,
                    ignore_index=ignore_index,
                    loss_reduction=loss_reduction,
                    z_loss_multiplier=z_loss_multiplier,
                    loss_div_factor=loss_div_factor,
                    return_logits=return_logits,
                    logits_to_keep=logits_to_keep,
                    **kwargs,
                )
            return current_forward(
                model,
                input_ids,
                input_embeddings=input_embeddings,
                labels=labels,
                ignore_index=ignore_index,
                loss_reduction=loss_reduction,
                z_loss_multiplier=z_loss_multiplier,
                loss_div_factor=loss_div_factor,
                return_logits=return_logits,
                logits_to_keep=logits_to_keep,
                **kwargs,
            )

        setattr(forward, "_olmo_ckv_layer_residual_hook", True)
        setattr(model_class, "forward", forward)

    current_apply_pp = model_class.apply_pp
    if not getattr(current_apply_pp, "_olmo_ckv_layer_residual_hook", False):

        @wraps(current_apply_pp)
        def apply_pp(
            model: Transformer,
            pp_mesh,
        ) -> None:
            if _uses_ckv_layer_residual(model):
                raise NotImplementedError(
                    "Pipeline parallelism is not supported with use_ckv_layer_residual because "
                    "raw_ckv is not part of OLMo's pipeline-stage interface"
                )
            current_apply_pp(model, pp_mesh)

        setattr(apply_pp, "_olmo_ckv_layer_residual_hook", True)
        setattr(model_class, "apply_pp", apply_pp)

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
    _install_ckv_block_hook()
    for model_class in (Transformer, MoETransformer):
        _install_model_hooks(model_class)
