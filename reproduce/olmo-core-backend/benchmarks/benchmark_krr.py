"""Benchmark Cubit's dense and streaming causal KRR solvers on CUDA."""

import argparse
import math
import time

import torch

from olmo_cubit import streaming_causal_krr_solve


def dense_causal_krr_solve(
    reference_queries: torch.Tensor,
    reference_keys: torch.Tensor,
    rhs: torch.Tensor,
    regularization: torch.Tensor,
) -> torch.Tensor:
    _, n_heads, seq_len, _ = reference_queries.shape
    mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=rhs.device).tril()
    scores = reference_queries @ reference_keys.transpose(-2, -1)
    kernel = torch.softmax(scores.masked_fill(~mask.view(1, 1, seq_len, seq_len), -torch.inf), -1)
    identity = torch.eye(seq_len, dtype=kernel.dtype, device=rhs.device)
    system = kernel + regularization.view(1, n_heads, 1, 1) * identity.view(
        1, 1, seq_len, seq_len
    )
    return torch.linalg.solve_triangular(system, rhs, upper=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation", choices=("streaming", "dense"), default="streaming")
    parser.add_argument("--kernel-backend", choices=("torch", "triton"), default="torch")
    parser.add_argument(
        "--mode",
        choices=("forward", "forward-backward"),
        default="forward-backward",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires CUDA")
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(0)
    device = torch.device("cuda")

    shape = (args.batch_size, args.heads, args.seq_len, args.head_dim)
    reference_queries = torch.randn(shape, device=device, dtype=torch.float32, requires_grad=True)
    reference_keys = torch.nn.functional.normalize(
        torch.randn(shape, device=device, dtype=torch.float32),
        dim=-1,
    ).requires_grad_()
    rhs = torch.randn(shape, device=device, dtype=torch.float32, requires_grad=True)
    regularization = torch.full(
        (args.heads,),
        1e-3,
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )
    inputs = (reference_queries, reference_keys, rhs, regularization)

    def step() -> None:
        for value in inputs:
            value.grad = None
        if args.implementation == "streaming":
            output = streaming_causal_krr_solve(
                *inputs,
                block_size=args.block_size,
                kernel_backend=args.kernel_backend,
            )
        else:
            output = dense_causal_krr_solve(*inputs)
        if args.mode == "forward-backward":
            output.square().mean().backward()

    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    for _ in range(args.steps):
        step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak_gib = torch.cuda.max_memory_allocated() / math.pow(1024, 3)

    print(f"implementation: {args.implementation}")
    print(f"kernel_backend: {args.kernel_backend}")
    print(f"mode: {args.mode}")
    print(f"shape: {shape}")
    print(f"block_size: {args.block_size}")
    print(f"time: {elapsed * 1000 / args.steps:.2f} ms")
    print(f"peak allocated: {peak_gib:.2f} GiB")


if __name__ == "__main__":
    main()
