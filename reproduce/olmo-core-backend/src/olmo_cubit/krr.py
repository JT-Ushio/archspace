"""Memory-efficient causal kernel-ridge-regression primitives for Cubit."""

from typing import Optional

import torch


def _batch_matmul(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Multiply ``[B, H, M, K]`` by ``[B, H, K, N]`` with strided batched GEMM."""
    batch_size, n_heads, _, inner_dim = left.shape
    if right.shape[:2] != (batch_size, n_heads) or right.shape[2] != inner_dim:
        raise RuntimeError("incompatible batched matrix multiplication shapes")
    return torch.matmul(left, right)


def _score_mask(
    *,
    batch_size: int,
    q_start: int,
    q_end: int,
    k_start: int,
    k_end: int,
    device: torch.device,
    doc_ids: Optional[torch.Tensor],
    window_size: Optional[int],
) -> torch.Tensor:
    query_positions = torch.arange(q_start, q_end, device=device)
    key_positions = torch.arange(k_start, k_end, device=device)
    mask = key_positions[None, :] <= query_positions[:, None]
    if window_size is not None:
        mask = mask & (
            key_positions[None, :] >= query_positions[:, None] - (window_size - 1)
        )
    mask = mask.view(1, 1, q_end - q_start, k_end - k_start)
    if doc_ids is not None:
        same_document = (
            doc_ids[:, q_start:q_end, None] == doc_ids[:, None, k_start:k_end]
        )
        mask = mask & same_document[:, None, :, :]
    return mask.expand(batch_size, 1, -1, -1)


def _kernel_weights(
    reference_queries: torch.Tensor,
    reference_keys: torch.Tensor,
    *,
    q_start: int,
    k_start: int,
    doc_ids: Optional[torch.Tensor],
    window_size: Optional[int],
    logsumexp: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one causal kernel tile and its row log-normalizers."""
    q_end = q_start + reference_queries.shape[2]
    k_end = k_start + reference_keys.shape[2]
    scores = _batch_matmul(reference_queries, reference_keys.transpose(-2, -1))
    mask = _score_mask(
        batch_size=reference_queries.shape[0],
        q_start=q_start,
        q_end=q_end,
        k_start=k_start,
        k_end=k_end,
        device=reference_queries.device,
        doc_ids=doc_ids,
        window_size=window_size,
    )
    scores = scores.masked_fill(~mask, -torch.inf)
    if logsumexp is None:
        logsumexp = torch.logsumexp(scores, dim=-1)
    weights = torch.exp(scores - logsumexp.unsqueeze(-1)).masked_fill(~mask, 0.0)
    return weights, logsumexp


def _first_key_for_block(q_start: int, window_size: Optional[int]) -> int:
    if window_size is None:
        return 0
    return max(0, q_start - window_size + 1)


def _forward_solve(
    reference_queries: torch.Tensor,
    reference_keys: torch.Tensor,
    rhs: torch.Tensor,
    regularization: torch.Tensor,
    *,
    doc_ids: Optional[torch.Tensor],
    window_size: Optional[int],
    block_size: int,
    kernel_backend: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, n_heads, seq_len, _ = rhs.shape
    solution = torch.empty_like(rhs)
    row_logsumexp = torch.empty(
        batch_size,
        n_heads,
        seq_len,
        dtype=reference_queries.dtype,
        device=reference_queries.device,
    )

    for q_start in range(0, seq_len, block_size):
        q_end = min(q_start + block_size, seq_len)
        if kernel_backend == "triton":
            try:
                from .krr_triton import triton_krr_forward_block
            except ImportError as exc:
                raise RuntimeError(
                    "the Triton KRR backend requires the 'triton' package"
                ) from exc

            previous, block_logsumexp, diagonal_block = triton_krr_forward_block(
                reference_queries,
                reference_keys,
                solution,
                q_start=q_start,
                q_end=q_end,
                doc_ids=doc_ids,
                window_size=window_size,
            )
            residual = rhs[:, :, q_start:q_end] - previous
        else:
            key_start = _first_key_for_block(q_start, window_size)
            weights, block_logsumexp = _kernel_weights(
                reference_queries[:, :, q_start:q_end],
                reference_keys[:, :, key_start:q_end],
                q_start=q_start,
                k_start=key_start,
                doc_ids=doc_ids,
                window_size=window_size,
            )
            previous_end = q_start - key_start
            residual = rhs[:, :, q_start:q_end]
            if previous_end > 0:
                residual = residual - _batch_matmul(
                    weights[:, :, :, :previous_end],
                    solution[:, :, key_start:q_start],
                )
            diagonal_block = weights[:, :, :, previous_end:]

        row_logsumexp[:, :, q_start:q_end] = block_logsumexp
        block_len = q_end - q_start
        identity = torch.eye(
            block_len,
            dtype=diagonal_block.dtype,
            device=diagonal_block.device,
        ).view(1, 1, block_len, block_len)
        diagonal_block = diagonal_block + regularization.view(1, n_heads, 1, 1) * identity
        solution[:, :, q_start:q_end] = torch.linalg.solve_triangular(
            diagonal_block,
            residual,
            upper=False,
        )

    return solution, row_logsumexp


def _transpose_solve(
    reference_queries: torch.Tensor,
    reference_keys: torch.Tensor,
    rhs: torch.Tensor,
    regularization: torch.Tensor,
    row_logsumexp: torch.Tensor,
    *,
    doc_ids: Optional[torch.Tensor],
    window_size: Optional[int],
    block_size: int,
) -> torch.Tensor:
    """Solve ``(K + lambda I)^T x = rhs`` without materializing ``K``."""
    _, n_heads, seq_len, _ = rhs.shape
    solution = torch.empty_like(rhs)

    for q_start in range(((seq_len - 1) // block_size) * block_size, -1, -block_size):
        q_end = min(q_start + block_size, seq_len)
        residual = rhs[:, :, q_start:q_end]

        future_end = seq_len
        if window_size is not None:
            future_end = min(seq_len, q_end + window_size - 1)
        if q_end < future_end:
            future_weights, _ = _kernel_weights(
                reference_queries[:, :, q_end:future_end],
                reference_keys[:, :, q_start:q_end],
                q_start=q_end,
                k_start=q_start,
                doc_ids=doc_ids,
                window_size=window_size,
                logsumexp=row_logsumexp[:, :, q_end:future_end],
            )
            residual = residual - _batch_matmul(
                future_weights.transpose(-2, -1),
                solution[:, :, q_end:future_end],
            )

        diagonal_weights, _ = _kernel_weights(
            reference_queries[:, :, q_start:q_end],
            reference_keys[:, :, q_start:q_end],
            q_start=q_start,
            k_start=q_start,
            doc_ids=doc_ids,
            window_size=window_size,
            logsumexp=row_logsumexp[:, :, q_start:q_end],
        )
        block_len = q_end - q_start
        identity = torch.eye(
            block_len,
            dtype=diagonal_weights.dtype,
            device=diagonal_weights.device,
        ).view(1, 1, block_len, block_len)
        diagonal_weights = (
            diagonal_weights + regularization.view(1, n_heads, 1, 1) * identity
        )
        solution[:, :, q_start:q_end] = torch.linalg.solve_triangular(
            diagonal_weights.transpose(-2, -1),
            residual,
            upper=True,
        )

    return solution


def _kernel_gradients(
    reference_queries: torch.Tensor,
    reference_keys: torch.Tensor,
    solution: torch.Tensor,
    adjoint: torch.Tensor,
    row_logsumexp: torch.Tensor,
    *,
    doc_ids: Optional[torch.Tensor],
    window_size: Optional[int],
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiate the row-softmax kernel by recomputing one row tile at a time."""
    seq_len = solution.shape[2]
    grad_queries = torch.zeros_like(reference_queries)
    grad_keys = torch.zeros_like(reference_keys)

    for q_start in range(0, seq_len, block_size):
        q_end = min(q_start + block_size, seq_len)
        key_start = _first_key_for_block(q_start, window_size)
        weights, _ = _kernel_weights(
            reference_queries[:, :, q_start:q_end],
            reference_keys[:, :, key_start:q_end],
            q_start=q_start,
            k_start=key_start,
            doc_ids=doc_ids,
            window_size=window_size,
            logsumexp=row_logsumexp[:, :, q_start:q_end],
        )
        grad_kernel = -_batch_matmul(
            adjoint[:, :, q_start:q_end],
            solution[:, :, key_start:q_end].transpose(-2, -1),
        )
        softmax_dot = (weights * grad_kernel).sum(dim=-1, keepdim=True)
        grad_scores = weights * (grad_kernel - softmax_dot)

        grad_queries[:, :, q_start:q_end] = _batch_matmul(
            grad_scores,
            reference_keys[:, :, key_start:q_end],
        )
        grad_keys[:, :, key_start:q_end] += _batch_matmul(
            grad_scores.transpose(-2, -1),
            reference_queries[:, :, q_start:q_end],
        )

    return grad_queries, grad_keys


class _StreamingCausalKRRSolve(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def forward(
        ctx,
        reference_queries: torch.Tensor,
        reference_keys: torch.Tensor,
        rhs: torch.Tensor,
        regularization: torch.Tensor,
        doc_ids: Optional[torch.Tensor],
        window_size: Optional[int],
        block_size: int,
        kernel_backend: str,
    ) -> torch.Tensor:
        solution, row_logsumexp = _forward_solve(
            reference_queries,
            reference_keys,
            rhs,
            regularization,
            doc_ids=doc_ids,
            window_size=window_size,
            block_size=block_size,
            kernel_backend=kernel_backend,
        )
        saved_doc_ids = (
            doc_ids
            if doc_ids is not None
            else torch.empty(0, dtype=torch.long, device=reference_queries.device)
        )
        ctx.save_for_backward(
            reference_queries,
            reference_keys,
            solution,
            regularization,
            row_logsumexp,
            saved_doc_ids,
        )
        ctx.has_doc_ids = doc_ids is not None
        ctx.window_size = window_size
        ctx.block_size = block_size
        return solution

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output: torch.Tensor):
        (
            reference_queries,
            reference_keys,
            solution,
            regularization,
            row_logsumexp,
            saved_doc_ids,
        ) = ctx.saved_tensors
        doc_ids = saved_doc_ids if ctx.has_doc_ids else None
        adjoint = _transpose_solve(
            reference_queries,
            reference_keys,
            grad_output,
            regularization,
            row_logsumexp,
            doc_ids=doc_ids,
            window_size=ctx.window_size,
            block_size=ctx.block_size,
        )
        grad_queries, grad_keys = _kernel_gradients(
            reference_queries,
            reference_keys,
            solution,
            adjoint,
            row_logsumexp,
            doc_ids=doc_ids,
            window_size=ctx.window_size,
            block_size=ctx.block_size,
        )
        grad_regularization = -(adjoint * solution).sum(dim=(0, 2, 3))
        return (
            grad_queries,
            grad_keys,
            adjoint,
            grad_regularization,
            None,
            None,
            None,
            None,
        )


def streaming_causal_krr_solve(
    reference_queries: torch.Tensor,
    reference_keys: torch.Tensor,
    rhs: torch.Tensor,
    regularization: torch.Tensor,
    *,
    doc_ids: Optional[torch.Tensor] = None,
    window_size: Optional[int] = None,
    block_size: int = 64,
    kernel_backend: str = "torch",
) -> torch.Tensor:
    """Solve Cubit's causal KRR system with ``O(T * D + T * block_size)`` memory.

    The computation remains exactly quadratic in sequence length. Like FlashAttention, kernel
    rows are constructed in blocks and recomputed during backward instead of storing a full
    ``[T, T]`` matrix.
    """
    if reference_queries.ndim != 4:
        raise RuntimeError("reference tensors must have shape [batch, heads, tokens, dim]")
    if reference_queries.shape != reference_keys.shape:
        raise RuntimeError("reference query and key tensors must have identical shapes")
    if rhs.ndim != 4 or rhs.shape[:3] != reference_queries.shape[:3]:
        raise RuntimeError("rhs must have shape [batch, heads, tokens, value_dim]")
    if regularization.shape != (reference_queries.shape[1],):
        raise RuntimeError("regularization must have shape [heads]")
    if doc_ids is not None and doc_ids.shape != (
        reference_queries.shape[0],
        reference_queries.shape[2],
    ):
        raise RuntimeError("doc_ids must have shape [batch, tokens]")
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size <= 0:
        raise RuntimeError("block_size must be a positive integer")
    if window_size is not None and window_size <= 0:
        raise RuntimeError("window_size must be positive")
    if kernel_backend not in ("torch", "triton"):
        raise RuntimeError("kernel_backend must be either 'torch' or 'triton'")
    if kernel_backend == "triton" and not reference_queries.is_cuda:
        raise RuntimeError("the Triton KRR backend requires CUDA tensors")

    return _StreamingCausalKRRSolve.apply(
        reference_queries.contiguous(),
        reference_keys.contiguous(),
        rhs.contiguous(),
        regularization,
        doc_ids,
        window_size,
        block_size,
        kernel_backend,
    )
