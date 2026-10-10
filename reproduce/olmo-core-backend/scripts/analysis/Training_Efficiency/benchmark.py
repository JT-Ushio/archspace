"""Measure actual distributed training steps on pre-generated synthetic inputs.

Uses fresh weights and optimizer state in a separate allocation. No historical
checkpoint is loaded or saved. Results exclude dataset I/O and checkpoint/eval.
"""

import argparse
import hashlib
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

from olmo_core.distributed.utils import get_rank, get_world_size
from olmo_core.train import prepare_training_environment, teardown_training_environment
from olmo_core.train.callbacks import CheckpointerCallback
from olmo_core.utils import seed_all

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parents[2]
CASES = {
    "dense": ("OLMo3-1B-DenseBaseline.py", 1_484_916_736),
    "c0": ("Rank512Matched_Combinations/c0_k_silu_down_rms.py", 797_050_880),
    "c6": ("Rank512Matched_Combinations/c6_qk_silu_down_rms_gateup.py", 797_050_880),
}
SEQUENCE_LENGTH = 8192
GLOBAL_TOKENS = 256 * SEQUENCE_LENGTH
MICROBATCH_TOKENS = 4 * SEQUENCE_LENGTH


def reject_checkpoint(*args, **kwargs):
    raise RuntimeError("Synthetic efficiency benchmark forbids checkpoint load/save")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=10)
    args = parser.parse_args()
    if args.warmup_steps < 5 or args.measure_steps < 10:
        raise ValueError("Require at least5warmup and10measured full optimizer steps")
    relative, expected_params = CASES[args.case]
    config_path = BACKEND / "cfps/BS256_Seq8K_30B_WSD_FP8" / relative
    sys.path.insert(0, str(config_path.parent))
    spec = importlib.util.spec_from_file_location("efficiency_recipe", config_path)
    recipe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recipe)
    opts = argparse.Namespace(name="isolated-efficiency-" + args.case,
                              sequence_length=SEQUENCE_LENGTH, data_root=args.data_root,
                              save_folder=str(args.runtime_dir / "unused-checkpoints"),
                              work_dir=str(args.runtime_dir / "dataset-cache"))
    config = recipe.build_config(opts, [])
    if config.model.num_params != expected_params:
        raise ValueError("Actual model parameter budget differs from the reference")
    # Explicitly match microbatch to new low-rank runs; historical dense used8seq.
    config.train_module.rank_microbatch_size = MICROBATCH_TOKENS
    if (config.data_loader.global_batch_size != GLOBAL_TOKENS
            or not config.train_module.float8_config.enabled
            or config.trainer.max_duration.value != 30_000_000_000):
        raise ValueError("Preserve the original global batch, FP8 and scheduler horizon")
    config.trainer.callbacks = {"checkpointer": CheckpointerCallback(enabled=False)}
    config.trainer.async_bookkeeping = False
    config.trainer.work_dir = str(args.runtime_dir / "trainer")
    prepare_training_environment(shared_filesystem=True, backend="cpu:gloo,cuda:nccl")
    trainer = None
    try:
        if get_world_size() != 8:
            raise ValueError("Compare all cases on one8GPU node")
        seed_all(config.init_seed)
        model = config.model.build(init_device="meta")
        module = config.train_module.build(model)
        if module.cp_enabled or module.tp_enabled or get_world_size(module.dp_process_group) != 8:
            raise ValueError("Require the same8-rank DP/FSDP layout for all cases")
        dataset = config.dataset.build()
        loader = config.data_loader.build(dataset, dp_process_group=module.dp_process_group)
        trainer = config.trainer.build(module, loader)
        trainer.load_checkpoint = reject_checkpoint
        trainer.save_checkpoint = reject_checkpoint
        trainer.save_checkpoint_async = reject_checkpoint
        module.pre_train()
        rank_sequences = GLOBAL_TOKENS // 8 // SEQUENCE_LENGTH
        generator = torch.Generator(device="cpu").manual_seed(34521 + get_rank())
        input_ids = torch.randint(config.model.vocab_size, (rank_sequences, SEQUENCE_LENGTH),
                                  dtype=torch.int64, generator=generator)
        captured = []
        original_record = trainer.record_metric

        def capture(name, value, *positional, **kwargs):
            full_name = (kwargs.get("namespace") + "/" + name
                         if kwargs.get("namespace") else name)
            if full_name in ("train/CE loss", "optim/total grad norm"):
                captured.append(value.detach())
            return original_record(name, value, *positional, **kwargs)

        trainer.record_metric = capture
        elapsed = []
        for step in range(1, args.warmup_steps + args.measure_steps + 1):
            trainer.global_step = step
            module.zero_grads()
            if step == args.warmup_steps + 1:
                torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            dist.barrier()
            start = time.perf_counter()
            module.train_batch({"input_ids": input_ids})
            module.optim_step()
            module.zero_grads()
            torch.cuda.synchronize()
            duration = torch.tensor(time.perf_counter() - start, device=module.device,
                                    dtype=torch.float64)
            dist.all_reduce(duration, op=dist.ReduceOp.MAX)
            trainer.global_train_tokens_seen += GLOBAL_TOKENS
            if step > args.warmup_steps:
                elapsed.append(float(duration.item()))
        if not captured:
            raise RuntimeError("Training CE/gradient metrics were not observed")
        finite = torch.tensor(int(all(torch.isfinite(value).all().item() for value in captured)),
                              device=module.device)
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not finite.item():
            raise ValueError("Nonfinite synthetic training CE or gradient norm")
        memory = torch.tensor([torch.cuda.max_memory_allocated(),
                               torch.cuda.max_memory_reserved()],
                              dtype=torch.int64, device=module.device)
        dist.all_reduce(memory, op=dist.ReduceOp.MAX)
        mean_time = statistics.mean(elapsed)
        properties = torch.cuda.get_device_properties(module.device)
        result = {
            "case": args.case, "num_params": expected_params, "world_size": 8,
            "global_tokens_per_step": GLOBAL_TOKENS, "sequence_length": SEQUENCE_LENGTH,
            "microbatch_tokens_per_rank": MICROBATCH_TOKENS,
            "microbatches_per_rank": GLOBAL_TOKENS // 8 // MICROBATCH_TOKENS,
            "warmup_steps_excluded": args.warmup_steps, "measured_steps": args.measure_steps,
            "max_rank_step_seconds": elapsed, "mean_step_seconds": mean_time,
            "median_step_seconds": statistics.median(elapsed),
            "global_tokens_per_second": GLOBAL_TOKENS / mean_time,
            "tokens_per_second_per_gpu": GLOBAL_TOKENS / mean_time / 8,
            "max_rank_peak_allocated_bytes": int(memory[0].item()),
            "max_rank_peak_reserved_bytes": int(memory[1].item()),
            "gpu": properties.name, "gpu_total_memory_bytes": properties.total_memory,
            "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
            "init_seed": config.init_seed, "input_seed_per_rank": "34521 + rank",
            "finite_ce_and_gradients": True,
            "scope": "Synthetic full training+optimizer steps; no I/O/eval/checkpoint; compilation warmup excluded; no quality claim",
            "historical_dense_microbatch": "65536; explicitly32768 here for comparison",
        }
        encoded = json.dumps(result, ensure_ascii=False, allow_nan=False)
        if get_rank() == 0:
            prefix = "MOSE_EFFICIENCY_" + args.case.upper()
            print(prefix + "_SHA256 " + hashlib.sha256(encoded.encode()).hexdigest(), flush=True)
            print(prefix + "_JSON " + encoded, flush=True)
    finally:
        if trainer is not None:
            trainer._shutdown()
        teardown_training_environment()


if __name__ == "__main__":
    main()
