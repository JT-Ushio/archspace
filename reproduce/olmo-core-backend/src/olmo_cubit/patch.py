from dataclasses import fields
from typing import Optional

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention import AttentionConfig, AttentionType
from olmo_core.nn.transformer import TransformerBlockConfig, TransformerConfig

from .attention import CubitAttentionConfig


def _convert_attention_config(
    attention: AttentionConfig,
    *,
    share_reference: bool,
    regularization: float,
    lrr_lower: float,
    lrr_upper: float,
    reference_norm_eps: float,
    krr_implementation: str,
    krr_block_size: int,
) -> CubitAttentionConfig:
    if type(attention) not in (AttentionConfig, CubitAttentionConfig):
        raise OLMoConfigurationError(
            "patch_cubit requires OLMo's dense AttentionConfig; "
            f"got {type(attention).__name__}"
        )
    if attention.name != AttentionType.default:
        raise OLMoConfigurationError("patch_cubit requires the default attention implementation")

    base_kwargs = {field.name: getattr(attention, field.name) for field in fields(AttentionConfig)}
    return CubitAttentionConfig(
        **base_kwargs,
        share_reference=share_reference,
        regularization=regularization,
        lrr_lower=lrr_lower,
        lrr_upper=lrr_upper,
        reference_norm_eps=reference_norm_eps,
        krr_implementation=krr_implementation,
        krr_block_size=krr_block_size,
    )


def patch_cubit(
    config: TransformerConfig,
    *,
    cubit_start_layer: int = 0,
    cubit_end_layer: Optional[int] = None,
    share_reference: bool = False,
    regularization: float = 1e-10,
    lrr_lower: float = 0.5,
    lrr_upper: float = 2.0,
    reference_norm_eps: float = 1e-6,
    krr_implementation: str = "streaming",
    krr_block_size: int = 64,
) -> TransformerConfig:
    """Return a copy using Cubit in the half-open layer range ``[start, end)``."""

    resolved_end_layer = config.n_layers if cubit_end_layer is None else cubit_end_layer
    if (
        isinstance(cubit_start_layer, bool)
        or not isinstance(cubit_start_layer, int)
        or isinstance(resolved_end_layer, bool)
        or not isinstance(resolved_end_layer, int)
        or not 0 <= cubit_start_layer < resolved_end_layer <= config.n_layers
    ):
        raise OLMoConfigurationError(
            "Cubit layer range must satisfy "
            f"0 <= cubit_start_layer < cubit_end_layer <= {config.n_layers}"
        )

    patched = config.copy()

    def patch_block(block: TransformerBlockConfig) -> None:
        if not isinstance(block.sequence_mixer, AttentionConfig):
            raise OLMoConfigurationError(
                "patch_cubit requires every selected block to use AttentionConfig"
            )
        block.sequence_mixer = _convert_attention_config(
            block.sequence_mixer,
            share_reference=share_reference,
            regularization=regularization,
            lrr_lower=lrr_lower,
            lrr_upper=lrr_upper,
            reference_norm_eps=reference_norm_eps,
            krr_implementation=krr_implementation,
            krr_block_size=krr_block_size,
        )

    if isinstance(patched.block, dict):
        if cubit_start_layer > 0 or resolved_end_layer < patched.n_layers:
            raise OLMoConfigurationError(
                "a partial Cubit layer range is not supported for named block patterns"
            )
        for block in patched.block.values():
            patch_block(block)
        return patched

    native_block = patched.block.copy()
    patch_block(patched.block)

    block_overrides = dict(patched.block_overrides or {})
    for block_idx, block in block_overrides.items():
        if cubit_start_layer <= block_idx < resolved_end_layer:
            patch_block(block)

    dense_indices = (
        *range(cubit_start_layer),
        *range(resolved_end_layer, patched.n_layers),
    )
    for block_idx in dense_indices:
        block_overrides.setdefault(block_idx, native_block.copy())
    patched.block_overrides = block_overrides or None
    return patched
