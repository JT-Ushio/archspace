"""Generate paired exact-SVD factors, then measure A/B/C step0 on fixed data.

Scheduled GPU job only. Each rank owns disjoint projections for SVD and disjoint
sequences for forward analysis. No optimizer, checkpoint training or W&B writes.
"""
import argparse
from datetime import timedelta
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import traceback

import torch
import torch.distributed as dist
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parents[2]
sys.path[:0] = [str(BACKEND), str(BACKEND / "src"), str(HERE.parent / "QK_Combination_Analysis")]
from analyze import prepare_samples
from checkpoint_io import write_csv, write_json
from olmo_mose import MoSENonlinearity
from olmo_mose.svd_initialization import (MODES, apply_artifacts, config_fingerprint, file_hash,
                                        make_factors, projection_pairs, rms, stable_seed, tensor_hash)
from cfps.BS256_Seq8K_30B_WSD_FP8.Down_Identity_Ablations._common import build_down_activation_config

LOG = logging.getLogger("init_probe")


def other_hashes(model):
    targets = {id(m.weight) for pair in projection_pairs(model).values() for m in pair}
    return {name: tensor_hash(value) for name, value in model.named_parameters() if id(value) not in targets}


def weight_metrics(enc, dec):
    # Nonzero spectrum of D E without forming the potentially large dense product.
    _, re = torch.linalg.qr(enc.T, mode="reduced")
    _, rd = torch.linalg.qr(dec, mode="reduced")
    se = torch.linalg.svdvals(re)
    sd = torch.linalg.svdvals(rd)
    product = torch.linalg.svdvals(rd @ re.T)
    if not torch.isfinite(product).all() or product[-1] <= 0:
        raise ValueError("Invalid product spectrum")
    return dict(E_rms=rms(enc).item(), D_rms=rms(dec).item(),
                DE_frobenius=product.square().sum().sqrt().item(),
                DE_condition_nonzero=(product[0]/product[-1]).item(),
                E_singular_values=se.cpu(), D_singular_values=sd.cpu(), DE_singular_values=product.cpu())


def activation_metrics(value):
    value = value.detach().float()
    if not torch.isfinite(value).all():
        raise ValueError("Nonfinite activation")
    absolute = value.abs().flatten()
    quantiles = torch.quantile(absolute, torch.tensor([.5, .99, .999], device=value.device))
    channel_rms = value.reshape(-1, value.shape[-1]).square().mean(0).sqrt()
    return dict(mean=value.mean().item(), rms=rms(value).item(), negative_fraction=(value < 0).float().mean().item(),
                abs_p50=quantiles[0].item(), abs_p99=quantiles[1].item(), abs_p999=quantiles[2].item(),
                abs_max=absolute.max().item(), channel_rms_min=channel_rms.min().item(),
                channel_rms_max=channel_rms.max().item())


def forward_probe(model, samples, mode, precision):
    rows, ce_rows, handles, context = [], [], [], {}
    def record(name, stage, value):
        sample = context["sample"]
        selected = value[:, sample["positions"].to(value.device), :]
        rows.append(dict(mode=mode, precision=precision, pair=name, stage=stage,
                         sample=sample["meta"]["sample"], **activation_metrics(selected)))
    for name, (enc, dec) in projection_pairs(model).items():
        handles.append(enc.register_forward_hook(lambda m, x, y, name=name: record(name, "a", y)))
        handles.append(dec.register_forward_pre_hook(lambda m, x, name=name: record(name, "z", x[0])))
        handles.append(dec.register_forward_hook(lambda m, x, y, name=name: record(name, "y", y)))
    try:
        with torch.no_grad():
            for sample in samples:
                context["sample"] = sample
                positions = sample["positions"].cuda()
                logits = model(sample["ids"][None].cuda(), logits_to_keep=positions[None])
                loss = F.cross_entropy(logits[0].float(), sample["labels"].cuda(), reduction="none")
                if not torch.isfinite(loss).all():
                    raise ValueError("Nonfinite initial loss")
                ce_rows.append(dict(mode=mode, precision=precision, sample=sample["meta"]["sample"],
                                    source=sample["meta"]["source"], tokens=len(loss), ce=loss.mean().item()))
                LOG.info("INITIAL CE %s %s sample=%s CE=%.6f", mode, precision, sample["meta"]["sample"], loss.mean())
    finally:
        for handle in handles:
            handle.remove()
    return rows, ce_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=34521, help="SVD/data sampling seed; model init seed is unchanged")
    parser.add_argument("--num-sequences", type=int, default=32)
    parser.add_argument("--tokens-per-sequence", type=int, default=128)
    args = parser.parse_args()
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s rank{rank} %(message)s")
    dist.init_process_group("nccl", timeout=timedelta(minutes=90))
    output = args.output_dir
    try:
        if rank == 0:
            output.mkdir(parents=True, exist_ok=False)
        dist.barrier()
        opts = argparse.Namespace(sequence_length=8192, data_root=args.data_root, work_dir=str(args.work_dir),
                                  save_folder=str(output / "unused-checkpoints"), name="svd-step0-probe")
        config = build_down_activation_config(opts, [], variant="svd-step0-probe",
                                              down_nonlinearity=MoSENonlinearity.rms_norm)
        cfg = config.model
        def new_model():
            model = cfg.build(init_device="meta")
            model.init_weights(device=device, max_seq_len=8192)
            return model
        model = new_model()
        reference_other = other_hashes(model)
        entries, weight_rows, spectra = {}, [], {}
        for index, (name, (encoder, decoder)) in enumerate(projection_pairs(model).items()):
            if index % world != rank:
                continue
            enc, dec = encoder.weight.detach(), decoder.weight.detach()
            LOG.info("SVD START %s E=%s D=%s", name, tuple(enc.shape), tuple(dec.shape))
            modes, entry = make_factors(enc, dec, std=cfg.init_std, seed=stable_seed(args.seed, name))
            path = output / f"{name}.pt"
            torch.save({mode: {key: val.cpu() for key, val in values.items()} for mode, values in modes.items()}, path)
            entry.update(file=path.name, sha256=file_hash(path))
            entries[name] = entry
            for mode in MODES:
                values = {"E": enc, "D": dec} if mode == "independent" else modes[mode]
                measured = weight_metrics(values["E"], values["D"])
                spectra[f"{name}/{mode}"] = {key: measured.pop(key) for key in
                    ("E_singular_values", "D_singular_values", "DE_singular_values")}
                weight_rows.append(dict(pair=name, mode=mode, **measured, **entry))
            if not torch.allclose(rms(modes["svd_scale_matched"]["E"]), rms(enc), rtol=2e-6) or not torch.allclose(
                    rms(modes["svd_scale_matched"]["D"]), rms(dec), rtol=2e-6):
                raise ValueError("Factor RMS matching failed")
            LOG.info("SVD COMPLETE %s retained_energy=%.4f", name, entry["retained_energy_fraction"])
        write_json(output / f"pairs-rank{rank}.json", entries)
        write_json(output / f"other-rank{rank}.json", reference_other)
        write_csv(output / f"weights-rank{rank}.csv", weight_rows)
        torch.save(spectra, output / f"spectra-rank{rank}.pt")
        # Drop references before allocating another full model.
        del model, encoder, decoder, enc, dec, modes, values
        gc.collect()
        torch.cuda.empty_cache()
        dist.barrier()
        if rank == 0:
            merged = {}
            for worker in range(world):
                new = json.loads((output / f"pairs-rank{worker}.json").read_text())
                if set(new) & set(merged):
                    raise ValueError("Duplicate artifact pair")
                merged.update(new)
                if json.loads((output / f"other-rank{worker}.json").read_text()) != reference_other:
                    raise ValueError("Non-target parameters differ across ranks")
            if len(merged) != cfg.n_layers * 7:
                raise ValueError("Missing SVD pairs")
            sources = [Path(__file__), BACKEND / "src/olmo_mose/svd_initialization.py",
                       BACKEND / "src/olmo_mose/attention.py", BACKEND / "src/olmo_mose/hooks.py"]
            write_json(output / "manifest.json", dict(status="complete", model_sha256=config_fingerprint(cfg),
                       svd_seed=args.seed, model_init_seed=cfg.init_seed, pairs=merged, world_size=world,
                       torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
                       device_name=torch.cuda.get_device_name(device), svd_algorithm="FP32 reduced gesvd",
                       other_parameter_hashes=reference_other,
                       source_sha256={str(p.relative_to(BACKEND)): file_hash(p) for p in sources}))
            write_json(output / "config.json", config.as_config_dict())
        # Dataset.prepare contains its own distributed collectives: every rank
        # must participate in the same order, even though only rank0 saves.
        args.queries = 32
        samples, sample_meta, data_config = prepare_samples(config.as_config_dict(), args)
        if rank == 0:
            torch.save(samples, output / "samples.pt")
            write_json(output / "samples.json", dict(samples=sample_meta, dataset=data_config))
        dist.barrier()
        samples = torch.load(output / "samples.pt", weights_only=True)[rank::world]
        if not samples:
            raise ValueError("Every GPU must have probe samples")
        act_rows, losses = [], []
        for mode in MODES:
            model = new_model()
            apply_artifacts(model, output, mode, config_fingerprint(cfg), require_probe=False)
            if other_hashes(model) != reference_other:
                raise ValueError("SVD initialization changed a non-target parameter")
            model.to(dtype=torch.bfloat16).eval().requires_grad_(False)
            for precision in ("bf16", "fp8"):
                if precision == "fp8":
                    model.apply_fp8(config.train_module.float8_config)
                    converted = [(n, str(m)) for n, m in model.named_modules() if type(m).__name__ == "Float8Linear"]
                    if not converted:
                        raise ValueError("FP8 conversion silently skipped")
                    write_json(output / f"fp8-{mode}-rank{rank}.json", converted)
                acts, ces = forward_probe(model, samples, mode, precision)
                act_rows.extend(acts)
                losses.extend(ces)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        write_csv(output / f"activations-rank{rank}.csv", act_rows)
        write_csv(output / f"ce-rank{rank}.csv", losses)
        # Exercise the real conversion / compile wrapping / FSDP / init order.
        # No optimizer steps: these models are discarded after shard loading.
        from olmo_core.distributed.parallel import build_world_mesh
        from olmo_core.train.train_module.transformer.common import parallelize_model
        mesh = build_world_mesh(dp=config.train_module.dp_config, device_type="cuda")
        for mode in ("independent", "svd_scale_matched"):
            model = parallelize_model(
                cfg.build(init_device="meta"), world_mesh=mesh, device=device,
                max_sequence_length=8192, rank_microbatch_size=32768,
                compile_model=config.train_module.compile_model,
                float8_config=config.train_module.float8_config,
                dp_config=config.train_module.dp_config,
                ac_config=config.train_module.ac_config)
            apply_artifacts(model, output, mode, config_fingerprint(cfg), require_probe=False)
            for name, (enc, dec) in projection_pairs(model).items():
                if mode == "svd_scale_matched":
                    entry = json.loads((output / "manifest.json").read_text())["pairs"][name]
                    saved = torch.load(output / entry["file"], weights_only=True)[mode]
                    if tensor_hash(enc.weight) != tensor_hash(saved["E"]) or tensor_hash(dec.weight) != tensor_hash(saved["D"]):
                        raise ValueError(f"FSDP artifact round-trip mismatch: {name}")
            LOG.info("FSDP INITIALIZATION SMOKE PASSED %s", mode)
            del model, enc, dec
            gc.collect()
            torch.cuda.empty_cache()
        write_json(output / f"rank-{rank}.complete.json", dict(samples=[s["meta"]["sample"] for s in samples],
                                                              ce_rows=len(losses), activation_rows=len(act_rows),
                                                              fsdp_initialization_smoke="passed"))
        dist.barrier()
        if rank == 0:
            from probe_report import report
            report(output)
            write_json(output / "probe_complete.json", dict(status="passed", manifest_sha256=file_hash(output / "manifest.json")))
            LOG.info("STEP0 PROBE COMPLETE; artifacts ready for paired training")
    except Exception as error:
        if output.exists():
            write_json(output / f"rank-{rank}.error.json", dict(error=repr(error), traceback=traceback.format_exc()))
        raise
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
