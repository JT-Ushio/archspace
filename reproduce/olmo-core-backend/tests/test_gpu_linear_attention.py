import pytest
import torch

from olmo_ahn import LinearAttentionConfig, LinearAttentionType

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/Triton")


@pytest.mark.parametrize("attention_type", list(LinearAttentionType))
def test_linear_attention_cuda_forward_backward_and_document_reset(
    attention_type: LinearAttentionType,
) -> None:
    pytest.importorskip("fla")
    from olmo_core.nn.transformer.init import InitMethod

    config = LinearAttentionConfig(
        attention_type=attention_type,
        n_heads=2,
        n_v_heads=2,
        head_dim=64,
        expand_v=1.0,
    )
    module = config.build(128, layer_idx=0, n_layers=2, init_device="meta")
    module.to_empty(device=torch.device("cuda"))
    for child in module.modules():
        if hasattr(child, "reset_parameters"):
            child.reset_parameters()
    module.init_weights(
        init_method=InitMethod.normal,
        d_model=128,
        block_idx=0,
        num_blocks=2,
    )
    module.to(dtype=torch.bfloat16)
    module.train()

    assert sum(parameter.numel() for parameter in module.parameters()) == config.num_params(128)

    cu_doc_lens = torch.tensor([0, 4, 8, 12, 16], device="cuda", dtype=torch.int32)
    inputs = torch.randn(2, 8, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    changed_suffix = inputs.detach().clone()
    changed_suffix.reshape(-1, 128)[4:] = torch.randn_like(
        changed_suffix.reshape(-1, 128)[4:]
    )

    output = module(inputs, cu_doc_lens=cu_doc_lens)
    changed_output = module(changed_suffix, cu_doc_lens=cu_doc_lens)

    assert output.shape == inputs.shape
    assert torch.isfinite(output).all()
    torch.testing.assert_close(
        output.reshape(-1, 128)[:4],
        changed_output.reshape(-1, 128)[:4],
    )

    output.float().square().mean().backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in module.parameters()
    )
