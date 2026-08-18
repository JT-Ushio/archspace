import pytest
import torch

from olmo_cubit import streaming_causal_krr_solve


def _dense_causal_krr_solve(
    reference_queries: torch.Tensor,
    reference_keys: torch.Tensor,
    rhs: torch.Tensor,
    regularization: torch.Tensor,
    *,
    doc_ids: torch.Tensor | None,
    window_size: int | None,
) -> torch.Tensor:
    _, n_heads, seq_len, _ = reference_queries.shape
    query_positions = torch.arange(seq_len)[:, None]
    key_positions = torch.arange(seq_len)[None, :]
    mask = key_positions <= query_positions
    if window_size is not None:
        mask = mask & (key_positions >= query_positions - (window_size - 1))
    mask = mask.view(1, 1, seq_len, seq_len)
    if doc_ids is not None:
        mask = mask & (doc_ids[:, :, None] == doc_ids[:, None, :])[:, None]

    scores = reference_queries @ reference_keys.transpose(-2, -1)
    kernel = torch.softmax(scores.masked_fill(~mask, -torch.inf), dim=-1)
    identity = torch.eye(seq_len, dtype=kernel.dtype).view(1, 1, seq_len, seq_len)
    system = kernel + regularization.view(1, n_heads, 1, 1) * identity
    return torch.linalg.solve_triangular(system, rhs, upper=False)


@pytest.mark.parametrize(
    ("doc_ids", "window_size"),
    [
        (None, None),
        (torch.tensor([[0, 0, 0, 1, 1, 1]]), None),
        (None, 3),
    ],
)
def test_streaming_krr_matches_dense_forward_and_backward(
    doc_ids: torch.Tensor | None,
    window_size: int | None,
) -> None:
    torch.manual_seed(23)
    inputs = [
        torch.randn(1, 2, 6, 3, dtype=torch.float64, requires_grad=True),
        torch.randn(1, 2, 6, 3, dtype=torch.float64, requires_grad=True),
        torch.randn(1, 2, 6, 4, dtype=torch.float64, requires_grad=True),
        torch.full((2,), 0.2, dtype=torch.float64, requires_grad=True),
    ]
    expected = _dense_causal_krr_solve(
        *inputs,
        doc_ids=doc_ids,
        window_size=window_size,
    )
    output_grad = torch.randn_like(expected)
    expected_grads = torch.autograd.grad(expected, inputs, output_grad)

    streaming_inputs = [value.detach().clone().requires_grad_() for value in inputs]
    actual = streaming_causal_krr_solve(
        *streaming_inputs,
        doc_ids=doc_ids,
        window_size=window_size,
        block_size=2,
    )
    actual_grads = torch.autograd.grad(actual, streaming_inputs, output_grad)

    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-10, atol=1e-10)


def test_streaming_krr_saved_state_is_linear_in_sequence_length() -> None:
    seq_len = 16
    saved_shapes: list[torch.Size] = []

    def pack(tensor: torch.Tensor) -> torch.Tensor:
        saved_shapes.append(tensor.shape)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        output = streaming_causal_krr_solve(
            torch.randn(1, 1, seq_len, 3, requires_grad=True),
            torch.randn(1, 1, seq_len, 3, requires_grad=True),
            torch.randn(1, 1, seq_len, 4, requires_grad=True),
            torch.full((1,), 0.2, requires_grad=True),
            block_size=4,
        )

    assert output.shape == (1, 1, seq_len, 4)
    assert all(shape[-2:] != torch.Size((seq_len, seq_len)) for shape in saved_shapes)
