import math

import pytest
import torch

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention import AttentionBackendName
from olmo_cubit import CubitAttention, CubitAttentionConfig


def _tiny_cubit(**kwargs) -> CubitAttention:
    return CubitAttention(
        d_model=8,
        n_heads=2,
        bias=False,
        backend=AttentionBackendName.torch,
        regularization=0.2,
        lrr_lower=0.5,
        lrr_upper=2.0,
        **kwargs,
    )


def _paper_reference(module: CubitAttention, x: torch.Tensor) -> torch.Tensor:
    batch_size, seq_len, _ = x.shape
    q = module.w_q(x).view(batch_size, seq_len, module.n_heads, module.head_dim)
    k = module.w_k(x).view(batch_size, seq_len, module.n_heads, module.head_dim)
    v = module.w_v(x).view(batch_size, seq_len, module.n_heads, module.head_dim)
    assert module.w_r is not None
    r = module.w_r(x).view(batch_size, seq_len, module.n_heads, module.head_dim)

    normalized_r = torch.nn.functional.normalize(
        r.float(), dim=-1, eps=module.reference_norm_eps
    )
    normalized_r = normalized_r * module.reference_scale.float().view(1, 1, -1, 1)
    mask = torch.ones(seq_len, seq_len, dtype=torch.bool).tril().view(1, 1, seq_len, seq_len)

    r_heads = r.float().transpose(1, 2)
    normalized_r_heads = normalized_r.transpose(1, 2)
    inverse_sigma = r_heads @ normalized_r_heads.transpose(-2, -1)
    inverse_sigma = torch.softmax(inverse_sigma.masked_fill(~mask, -torch.inf), dim=-1)
    inverse_sigma = inverse_sigma + torch.diag_embed(
        module.log_regularization.float().exp().view(1, -1, 1).expand(1, -1, seq_len)
    )

    lrr_logits = module.w_lrr(x).float().transpose(1, 2).unsqueeze(-1)
    lrr = module.lrr_lower.float().view(1, -1, 1, 1)
    lrr = lrr + module.lrr_range.float().view(1, -1, 1, 1) * torch.sigmoid(lrr_logits)
    solution = torch.linalg.solve_triangular(
        inverse_sigma,
        lrr * v.float().transpose(1, 2),
        upper=False,
    )

    q_heads = q.float().transpose(1, 2)
    k_heads = k.float().transpose(1, 2)
    scores = q_heads @ k_heads.transpose(-2, -1) / math.sqrt(module.head_dim)
    attention = torch.softmax(scores.masked_fill(~mask, -torch.inf), dim=-1)
    output = (attention @ solution).transpose(1, 2).reshape(batch_size, seq_len, -1)
    return module.w_out(output.to(x.dtype))


def test_forward_matches_appendix_h_formula() -> None:
    torch.manual_seed(7)
    module = _tiny_cubit()
    x = torch.randn(2, 5, 8)

    expected = _paper_reference(module, x)
    actual = module(x)

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


def test_backward_is_finite_for_every_parameter() -> None:
    torch.manual_seed(11)
    module = _tiny_cubit()
    x = torch.randn(2, 6, 8, requires_grad=True)

    module(x).square().mean().backward()

    assert torch.isfinite(x.grad).all()
    for parameter in module.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_causal_mask_prevents_future_information_leakage() -> None:
    torch.manual_seed(17)
    module = _tiny_cubit()
    x = torch.randn(1, 6, 8)
    changed = x.clone()
    changed[:, 4:] = torch.randn_like(changed[:, 4:]) * 10

    torch.testing.assert_close(module(x)[:, :4], module(changed)[:, :4], rtol=0, atol=0)


def test_packed_documents_are_isolated_on_cpu_fallback() -> None:
    torch.manual_seed(19)
    module = _tiny_cubit()
    x = torch.randn(1, 6, 8)
    changed = x.clone()
    changed[:, 3:] = torch.randn_like(changed[:, 3:]) * 10
    cu_doc_lens = torch.tensor([0, 3, 6], dtype=torch.int32)

    first = module(x, cu_doc_lens=cu_doc_lens, max_doc_len=3)
    second = module(changed, cu_doc_lens=cu_doc_lens, max_doc_len=3)

    torch.testing.assert_close(first[:, :3], second[:, :3], rtol=0, atol=0)


def test_sliding_window_is_applied_to_krr_and_output_masks() -> None:
    module = _tiny_cubit(window_size=3)
    mask = module._build_attention_mask(
        batch_size=1,
        seq_len=5,
        device=torch.device("cpu"),
        cu_doc_lens=None,
    )

    expected = torch.tensor(
        [
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 1, 1],
        ],
        dtype=torch.bool,
    )
    torch.testing.assert_close(mask[0], expected)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"regularization": 0.0},
        {"lrr_lower": 0.0},
        {"lrr_lower": 2.0, "lrr_upper": 1.0},
        {"reference_norm_eps": 0.0},
        {"n_heads": 4, "n_kv_heads": 2},
    ],
)
def test_config_rejects_invalid_cubit_settings(kwargs) -> None:
    with pytest.raises(OLMoConfigurationError):
        CubitAttentionConfig(**kwargs)
