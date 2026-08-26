"""Optional Triton forward kernel for Cubit's streaming KRR solve."""

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _krr_forward_block_kernel(
    reference_queries,
    reference_keys,
    solution,
    doc_ids,
    previous_output,
    row_logsumexp_output,
    diagonal_output,
    n_heads: tl.constexpr,
    n_tokens: tl.constexpr,
    head_dim: tl.constexpr,
    q_start,
    q_end,
    first_key,
    block_len: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_d: tl.constexpr,
    block_diag: tl.constexpr,
    has_doc_ids: tl.constexpr,
    window_size: tl.constexpr,
):
    """Fuse a kernel row's online softmax and solved-prefix aggregation."""
    query_tile = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch_idx = batch_head // n_heads

    query_offsets = query_tile * block_m + tl.arange(0, block_m)
    query_positions = q_start + query_offsets
    dim_offsets = tl.arange(0, block_d)
    valid_queries = query_offsets < block_len
    valid_dims = dim_offsets < head_dim

    query_ptrs = (
        (batch_head * n_tokens + query_positions[:, None]) * head_dim
        + dim_offsets[None, :]
    )
    queries = tl.load(
        reference_queries + query_ptrs,
        mask=valid_queries[:, None] & valid_dims[None, :],
        other=0.0,
    )

    if has_doc_ids:
        query_doc_ids = tl.load(
            doc_ids + batch_idx * n_tokens + query_positions,
            mask=valid_queries,
            other=-1,
        )

    row_max = tl.full((block_m,), -float("inf"), tl.float32)
    row_sum = tl.zeros((block_m,), tl.float32)
    previous = tl.zeros((block_m, block_d), tl.float32)

    for key_start in tl.range(first_key, q_end, block_n):
        key_offsets = tl.arange(0, block_n)
        key_positions = key_start + key_offsets
        valid_keys = key_positions < q_end
        key_ptrs = (
            (batch_head * n_tokens + key_positions[:, None]) * head_dim
            + dim_offsets[None, :]
        )
        keys = tl.load(
            reference_keys + key_ptrs,
            mask=valid_keys[:, None] & valid_dims[None, :],
            other=0.0,
        )
        scores = tl.dot(queries, tl.trans(keys), input_precision="tf32")

        valid_scores = (
            valid_queries[:, None]
            & valid_keys[None, :]
            & (key_positions[None, :] <= query_positions[:, None])
        )
        if window_size > 0:
            valid_scores = valid_scores & (
                key_positions[None, :]
                >= query_positions[:, None] - (window_size - 1)
            )
        if has_doc_ids:
            key_doc_ids = tl.load(
                doc_ids + batch_idx * n_tokens + key_positions,
                mask=valid_keys,
                other=-2,
            )
            valid_scores = valid_scores & (
                query_doc_ids[:, None] == key_doc_ids[None, :]
            )
        scores = tl.where(valid_scores, scores, -float("inf"))

        tile_max = tl.max(scores, axis=1)
        next_max = tl.maximum(row_max, tile_max)
        correction = tl.exp(row_max - next_max)
        unnormalized = tl.exp(scores - next_max[:, None])
        row_sum = row_sum * correction + tl.sum(unnormalized, axis=1)

        previous_key_mask = valid_keys & (key_positions < q_start)
        value_ptrs = (
            (batch_head * n_tokens + key_positions[:, None]) * head_dim
            + dim_offsets[None, :]
        )
        previous_values = tl.load(
            solution + value_ptrs,
            mask=previous_key_mask[:, None] & valid_dims[None, :],
            other=0.0,
        )
        previous_weights = tl.where(
            valid_scores & (key_positions[None, :] < q_start),
            unnormalized,
            0.0,
        )
        previous = previous * correction[:, None] + tl.dot(
            previous_weights,
            previous_values,
            input_precision="tf32",
        )
        row_max = next_max

    row_logsumexp = row_max + tl.log(row_sum)
    previous = previous / row_sum[:, None]

    previous_ptrs = (
        (batch_head * block_len + query_offsets[:, None]) * head_dim
        + dim_offsets[None, :]
    )
    tl.store(
        previous_output + previous_ptrs,
        previous,
        mask=valid_queries[:, None] & valid_dims[None, :],
    )
    tl.store(
        row_logsumexp_output + batch_head * block_len + query_offsets,
        row_logsumexp,
        mask=valid_queries,
    )

    diagonal_offsets = tl.arange(0, block_diag)
    diagonal_positions = q_start + diagonal_offsets
    valid_diagonal_keys = diagonal_offsets < block_len
    diagonal_key_ptrs = (
        (batch_head * n_tokens + diagonal_positions[:, None]) * head_dim
        + dim_offsets[None, :]
    )
    diagonal_keys = tl.load(
        reference_keys + diagonal_key_ptrs,
        mask=valid_diagonal_keys[:, None] & valid_dims[None, :],
        other=0.0,
    )
    diagonal_scores = tl.dot(
        queries,
        tl.trans(diagonal_keys),
        input_precision="tf32",
    )
    valid_diagonal = (
        valid_queries[:, None]
        & valid_diagonal_keys[None, :]
        & (diagonal_positions[None, :] <= query_positions[:, None])
    )
    if window_size > 0:
        valid_diagonal = valid_diagonal & (
            diagonal_positions[None, :]
            >= query_positions[:, None] - (window_size - 1)
        )
    if has_doc_ids:
        diagonal_doc_ids = tl.load(
            doc_ids + batch_idx * n_tokens + diagonal_positions,
            mask=valid_diagonal_keys,
            other=-2,
        )
        valid_diagonal = valid_diagonal & (
            query_doc_ids[:, None] == diagonal_doc_ids[None, :]
        )
    diagonal_weights = tl.where(
        valid_diagonal,
        tl.exp(diagonal_scores - row_logsumexp[:, None]),
        0.0,
    )
    diagonal_ptrs = (
        (batch_head * block_len + query_offsets[:, None]) * block_len
        + diagonal_offsets[None, :]
    )
    tl.store(
        diagonal_output + diagonal_ptrs,
        diagonal_weights,
        mask=valid_queries[:, None] & valid_diagonal_keys[None, :],
    )


def triton_krr_forward_block(
    reference_queries: torch.Tensor,
    reference_keys: torch.Tensor,
    solution: torch.Tensor,
    *,
    q_start: int,
    q_end: int,
    doc_ids: Optional[torch.Tensor],
    window_size: Optional[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute one forward-solve block without materializing its prefix kernel tile."""
    if not reference_queries.is_cuda:
        raise RuntimeError("the Triton KRR backend requires CUDA tensors")
    if reference_queries.dtype != torch.float32:
        raise RuntimeError("the Triton KRR backend requires float32 reference tensors")
    if not (
        reference_queries.is_contiguous()
        and reference_keys.is_contiguous()
        and solution.is_contiguous()
    ):
        raise RuntimeError("the Triton KRR backend requires contiguous tensors")
    if reference_queries.shape != reference_keys.shape:
        raise RuntimeError("reference query and key tensors must have identical shapes")

    batch_size, n_heads, n_tokens, head_dim = reference_queries.shape
    if solution.shape != reference_queries.shape:
        raise RuntimeError(
            "the Triton KRR backend currently requires value_dim to equal head_dim"
        )
    if head_dim > 256:
        raise RuntimeError("the Triton KRR backend currently supports head_dim <= 256")

    block_len = q_end - q_start
    if block_len > 256:
        raise RuntimeError("the Triton KRR backend currently supports block_size <= 256")
    first_key = 0 if window_size is None else max(0, q_start - window_size + 1)
    block_m = 16
    block_n = 64
    block_d = max(16, triton.next_power_of_2(head_dim))
    block_diag = max(16, triton.next_power_of_2(block_len))

    previous = torch.empty(
        batch_size,
        n_heads,
        block_len,
        head_dim,
        dtype=torch.float32,
        device=reference_queries.device,
    )
    row_logsumexp = torch.empty(
        batch_size,
        n_heads,
        block_len,
        dtype=torch.float32,
        device=reference_queries.device,
    )
    diagonal = torch.empty(
        batch_size,
        n_heads,
        block_len,
        block_len,
        dtype=torch.float32,
        device=reference_queries.device,
    )
    doc_ids_arg = (
        doc_ids.contiguous()
        if doc_ids is not None
        else torch.empty(1, dtype=torch.long, device=reference_queries.device)
    )

    grid = (triton.cdiv(block_len, block_m), batch_size * n_heads)
    _krr_forward_block_kernel[grid](
        reference_queries,
        reference_keys,
        solution,
        doc_ids_arg,
        previous,
        row_logsumexp,
        diagonal,
        n_heads=n_heads,
        n_tokens=n_tokens,
        head_dim=head_dim,
        q_start=q_start,
        q_end=q_end,
        first_key=first_key,
        block_len=block_len,
        block_m=block_m,
        block_n=block_n,
        block_d=block_d,
        block_diag=block_diag,
        has_doc_ids=doc_ids is not None,
        window_size=0 if window_size is None else window_size,
        num_warps=4,
    )
    return previous, row_logsumexp, diagonal
