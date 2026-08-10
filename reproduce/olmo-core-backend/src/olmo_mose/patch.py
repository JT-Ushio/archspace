from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.feed_forward import ActivationFunction, FeedForwardConfig, FeedForwardType
from olmo_core.nn.transformer import TransformerBlockConfig, TransformerConfig

from .feed_forward import (
    ChannelControlledFeedForwardConfig,
    MoSENonlinearity,
    MoSESwiGLUConfig,
    SwiGLUChannelControl,
)


def _require_supported_feed_forward(feed_forward, caller: str) -> FeedForwardConfig:
    supported_types = (
        FeedForwardConfig,
        ChannelControlledFeedForwardConfig,
        MoSESwiGLUConfig,
    )
    if type(feed_forward) not in supported_types:
        raise OLMoConfigurationError(
            f"{caller} requires every patched block to have a supported dense "
            f"FeedForwardConfig; got {type(feed_forward).__name__}"
        )
    return feed_forward


def _convert_feed_forward_config(
    feed_forward: FeedForwardConfig,
    *,
    control: SwiGLUChannelControl,
) -> ChannelControlledFeedForwardConfig:
    if feed_forward.name != FeedForwardType.default:
        raise OLMoConfigurationError(
            "patch_swiglu_channel_control requires the default feed-forward implementation"
        )
    if feed_forward.activation != ActivationFunction.silu:
        raise OLMoConfigurationError(
            "patch_swiglu_channel_control requires a SwiGLU feed-forward"
        )

    return ChannelControlledFeedForwardConfig(
        hidden_size=feed_forward.hidden_size,
        name=feed_forward.name,
        bias=feed_forward.bias,
        dtype=feed_forward.dtype,
        activation=feed_forward.activation,
        control=control,
    )


def patch_swiglu_channel_control(
    config: TransformerConfig,
    *,
    control: SwiGLUChannelControl,
) -> TransformerConfig:
    """Return a copy of an OLMo config with dense SwiGLU channel control enabled."""
    control = SwiGLUChannelControl(control)
    patched = config.copy()

    def patch_block(block: TransformerBlockConfig) -> None:
        feed_forward = _require_supported_feed_forward(
            block.feed_forward,
            "patch_swiglu_channel_control",
        )
        if isinstance(feed_forward, MoSESwiGLUConfig):
            raise OLMoConfigurationError(
                "patch_swiglu_channel_control cannot change MoSE topology; "
                "use patch_mose_swiglu"
            )
        if control == SwiGLUChannelControl.standard:
            if isinstance(feed_forward, ChannelControlledFeedForwardConfig):
                block.feed_forward = FeedForwardConfig(
                    hidden_size=feed_forward.hidden_size,
                    name=feed_forward.name,
                    bias=feed_forward.bias,
                    dtype=feed_forward.dtype,
                    activation=feed_forward.activation,
                )
            return
        block.feed_forward = _convert_feed_forward_config(
            feed_forward,
            control=control,
        )

    if isinstance(patched.block, dict):
        for block in patched.block.values():
            patch_block(block)
    else:
        patch_block(patched.block)
    if patched.block_overrides is not None:
        for block in patched.block_overrides.values():
            patch_block(block)

    return patched


def patch_mose_swiglu(
    config: TransformerConfig,
    *,
    control: SwiGLUChannelControl,
    mose_start_layer: int = 0,
    r1: int = 880,
    r2: int = 880,
    down_r1: int = 880,
    down_r2: int = 880,
    gate_nonlinearity: MoSENonlinearity = MoSENonlinearity.silu,
    up_nonlinearity: MoSENonlinearity = MoSENonlinearity.silu,
    down_nonlinearity: MoSENonlinearity = MoSENonlinearity.silu,
) -> TransformerConfig:
    """Return a copy using MoSE-SwiGLU from ``mose_start_layer`` onward."""
    if (
        isinstance(mose_start_layer, bool)
        or not isinstance(mose_start_layer, int)
        or not 0 <= mose_start_layer < config.n_layers
    ):
        raise OLMoConfigurationError(
            f"mose_start_layer must be an integer in [0, {config.n_layers})"
        )
    control = SwiGLUChannelControl(control)
    gate_nonlinearity = MoSENonlinearity(gate_nonlinearity)
    up_nonlinearity = MoSENonlinearity(up_nonlinearity)
    down_nonlinearity = MoSENonlinearity(down_nonlinearity)
    patched = config.copy()

    def patch_block(block: TransformerBlockConfig) -> None:
        feed_forward = _require_supported_feed_forward(
            block.feed_forward,
            "patch_mose_swiglu",
        )
        if feed_forward.name != FeedForwardType.default:
            raise OLMoConfigurationError(
                "patch_mose_swiglu requires the default feed-forward implementation"
            )
        if feed_forward.activation != ActivationFunction.silu:
            raise OLMoConfigurationError("patch_mose_swiglu requires a SwiGLU feed-forward")

        block.feed_forward = MoSESwiGLUConfig(
            hidden_size=feed_forward.hidden_size,
            name=feed_forward.name,
            bias=True if feed_forward.bias is None else feed_forward.bias,
            dtype=feed_forward.dtype,
            activation=feed_forward.activation,
            r1=r1,
            r2=r2,
            down_r1=down_r1,
            down_r2=down_r2,
            control=control,
            gate_nonlinearity=gate_nonlinearity,
            up_nonlinearity=up_nonlinearity,
            down_nonlinearity=down_nonlinearity,
        )

    def apply_dense_control(block: TransformerBlockConfig) -> None:
        feed_forward = _require_supported_feed_forward(
            block.feed_forward,
            "patch_mose_swiglu",
        )
        if isinstance(feed_forward, MoSESwiGLUConfig):
            raise OLMoConfigurationError(
                "mose_start_layer > 0 requires a dense input feed-forward config"
            )
        if control == SwiGLUChannelControl.standard:
            if isinstance(feed_forward, ChannelControlledFeedForwardConfig):
                block.feed_forward = FeedForwardConfig(
                    hidden_size=feed_forward.hidden_size,
                    name=feed_forward.name,
                    bias=feed_forward.bias,
                    dtype=feed_forward.dtype,
                    activation=feed_forward.activation,
                )
        else:
            block.feed_forward = _convert_feed_forward_config(
                feed_forward,
                control=control,
            )

    if isinstance(patched.block, dict):
        if mose_start_layer > 0:
            raise OLMoConfigurationError(
                "mose_start_layer > 0 is not supported for named block patterns"
            )
        for block in patched.block.values():
            patch_block(block)
        return patched

    block_overrides = dict(patched.block_overrides or {})
    native_block = patched.block.copy()
    if mose_start_layer > 0:
        apply_dense_control(native_block)
    patch_block(patched.block)
    for block_idx, block in block_overrides.items():
        if block_idx >= mose_start_layer:
            patch_block(block)
        else:
            apply_dense_control(block)
    for block_idx in range(mose_start_layer):
        block_overrides.setdefault(block_idx, native_block.copy())
    patched.block_overrides = block_overrides or None

    return patched
