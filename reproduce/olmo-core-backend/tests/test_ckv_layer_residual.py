import pytest
import torch
import torch.nn.functional as F

from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.transformer import (
    TransformerActivationCheckpointingMode,
    TransformerConfig,
)
from olmo_mla_morphnorm import MLAAttention, MLANormType, patch_transformer_config


def _build_attention(*, use_ckv_layer_residual: bool) -> MLAAttention:
    return MLAAttention(
        d_model=8,
        n_heads=2,
        n_kv_heads=2,
        qk_head_dim=4,
        value_head_dim=4,
        q_lora_rank=None,
        kv_lora_rank=3,
        rope_dim=0,
        norm_type=MLANormType.baseline,
        use_q_a_layernorm=False,
        use_kv_a_layernorm=True,
        use_ckv_layer_residual=use_ckv_layer_residual,
        norm_config=LayerNormConfig(name=LayerNormType.rms, bias=False),
        rope=None,
        dropout=0.0,
        softmax_scale=None,
        use_flash=False,
        backend=AttentionBackendName.torch,
        window_size=None,
        dtype=torch.float32,
        init_device="cpu",
        cache=None,
        morphnorm_eps=1e-6,
        morphnorm_update_stats=False,
        return_max_logits=False,
        log_max_logits_per_head=False,
    )


def _build_model(*, use_ckv_layer_residual: bool):
    config = patch_transformer_config(
        TransformerConfig.olmo2_1M(
            vocab_size=32,
            n_layers=3,
            attn_backend=AttentionBackendName.torch,
        ),
        q_lora_rank=None,
        kv_lora_rank=4,
        rope_dim=0,
        value_head_dim=4,
        norm_type=MLANormType.baseline,
        use_q_a_layernorm=False,
        use_kv_a_layernorm=True,
        use_ckv_layer_residual=use_ckv_layer_residual,
    )
    config.block.sequence_mixer.qk_norm = LayerNormConfig(
        name=LayerNormType.rms, bias=False
    )
    return config.build(init_device="cpu")


def _model_attentions(model) -> list[MLAAttention]:
    return [module for module in model.modules() if isinstance(module, MLAAttention)]


def test_disabled_residual_ignores_previous_raw_ckv() -> None:
    torch.manual_seed(1)
    attention = _build_attention(use_ckv_layer_residual=False).eval()
    x = torch.randn(2, 3, attention.d_model)
    prev_raw_ckv = torch.randn(2, 3, attention.kv_lora_rank)

    output = attention(x)
    output_with_previous = attention(x, prev_raw_ckv=prev_raw_ckv)

    assert isinstance(output, torch.Tensor)
    torch.testing.assert_close(output_with_previous, output, rtol=0, atol=0)

    state_dict_before = tuple(attention.state_dict())
    parameter_count_before = sum(parameter.numel() for parameter in attention.parameters())
    attention.use_ckv_layer_residual = True
    assert tuple(attention.state_dict()) == state_dict_before
    assert sum(parameter.numel() for parameter in attention.parameters()) == parameter_count_before


def test_disabled_model_matches_original_transformer_forward() -> None:
    torch.manual_seed(11)
    model = _build_model(use_ckv_layer_residual=False).eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])

    output = model(input_ids)
    original_forward = type(model).forward.__wrapped__
    reference_output = original_forward(model, input_ids)

    torch.testing.assert_close(output, reference_output, rtol=0, atol=0)


def test_raw_ckv_accumulates_before_kv_norm_across_three_layers() -> None:
    torch.manual_seed(2)
    attentions = [_build_attention(use_ckv_layer_residual=True) for _ in range(3)]
    inputs = [torch.randn(2, 3, attention.d_model) for attention in attentions]
    norm_inputs: list[torch.Tensor] = []
    handles = []
    for attention in attentions:
        assert attention.kv_a_layernorm is not None
        handles.append(
            attention.kv_a_layernorm.register_forward_pre_hook(
                lambda _module, args: norm_inputs.append(args[0])
            )
        )

    raw_ckv_states = []
    prev_raw_ckv = None
    for attention, x in zip(attentions, inputs):
        _, _, _, _, raw_ckv = attention._project(x, prev_raw_ckv)
        raw_ckv_states.append(raw_ckv)
        prev_raw_ckv = raw_ckv

    for handle in handles:
        handle.remove()

    projected_ckv = [
        F.linear(x, attention.w_kv_a.weight)[..., : attention.kv_lora_rank]
        for attention, x in zip(attentions, inputs)
    ]
    expected_c0 = projected_ckv[0]
    expected_c1 = projected_ckv[1] + expected_c0
    expected_c2 = projected_ckv[2] + expected_c1

    torch.testing.assert_close(raw_ckv_states[0], expected_c0)
    torch.testing.assert_close(raw_ckv_states[1], expected_c1)
    torch.testing.assert_close(raw_ckv_states[2], expected_c2)
    for norm_input, raw_ckv in zip(norm_inputs, raw_ckv_states):
        assert norm_input is raw_ckv


def test_ckv_residual_rejects_shape_and_dtype_mismatches() -> None:
    attention = _build_attention(use_ckv_layer_residual=True)
    x = torch.randn(2, 3, attention.d_model)

    with pytest.raises(ValueError, match="shape mismatch"):
        attention._project(x, torch.randn(2, 2, attention.kv_lora_rank))
    with pytest.raises(ValueError, match="dtype mismatch"):
        attention._project(x, torch.randn(2, 3, attention.kv_lora_rank).double())


def test_model_passes_raw_ckv_between_transformer_layers() -> None:
    torch.manual_seed(3)
    model = _build_model(use_ckv_layer_residual=True).eval()
    attentions = _model_attentions(model)
    projected_ckv: list[torch.Tensor] = []
    norm_inputs: list[torch.Tensor] = []
    handles = []

    for attention in attentions:
        handles.append(
            attention.w_kv_a.register_forward_hook(
                lambda module, _args, output: projected_ckv.append(
                    output[..., : module.out_features]
                )
            )
        )
        assert attention.kv_a_layernorm is not None
        handles.append(
            attention.kv_a_layernorm.register_forward_pre_hook(
                lambda _module, args: norm_inputs.append(args[0])
            )
        )

    output = model(torch.tensor([[1, 2, 3, 4]]))
    for handle in handles:
        handle.remove()

    assert isinstance(output, torch.Tensor)
    assert len(projected_ckv) == len(norm_inputs) == 3
    accumulated = projected_ckv[0]
    torch.testing.assert_close(norm_inputs[0], accumulated)
    for layer_idx in range(1, 3):
        accumulated = projected_ckv[layer_idx] + accumulated
        torch.testing.assert_close(norm_inputs[layer_idx], accumulated)


def test_final_normalized_ckv_gradient_reaches_first_projection() -> None:
    torch.manual_seed(4)
    attentions = [_build_attention(use_ckv_layer_residual=True) for _ in range(3)]
    inputs = [torch.randn(2, 3, attention.d_model) for attention in attentions]

    prev_raw_ckv = None
    final_normalized_ckv = None
    for attention, x in zip(attentions, inputs):
        _, _, final_normalized_ckv, _, prev_raw_ckv = attention._project(x, prev_raw_ckv)

    assert final_normalized_ckv is not None
    loss = attentions[-1].w_v_b(final_normalized_ckv).square().mean()
    loss.backward()

    first_grad = attentions[0].w_kv_a.weight.grad
    assert first_grad is not None
    assert torch.count_nonzero(first_grad).item() > 0


def test_full_activation_checkpointing_preserves_ckv_gradient() -> None:
    torch.manual_seed(5)
    model = _build_model(use_ckv_layer_residual=True).train()
    model.apply_activation_checkpointing(TransformerActivationCheckpointingMode.full)

    output = model(torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]]))
    output.float().square().mean().backward()

    first_grad = _model_attentions(model)[0].w_kv_a.weight.grad
    assert first_grad is not None
    assert torch.count_nonzero(first_grad).item() > 0


def test_cached_prefill_and_decode_preserve_model_api() -> None:
    model = _build_model(use_ckv_layer_residual=True).eval()
    for attention in _model_attentions(model):
        attention.init_kv_cache_manager(batch_size=1, max_seq_len=8)

    prefill_output = model(torch.tensor([[1, 2, 3]]))
    decode_output = model(torch.tensor([[4]]))

    assert prefill_output.shape == (1, 3, 32)
    assert decode_output.shape == (1, 1, 32)
