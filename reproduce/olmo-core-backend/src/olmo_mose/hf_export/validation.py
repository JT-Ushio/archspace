"""Opt-in numerical validation. Importing/exporting does not execute these checks."""

import dataclasses
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from .checkpoint import CheckpointReader, parse_model_config, sha256, write_json
from .exporter import DTYPES, target_dtype
from .registry import native_to_hf_key


def difference(actual, expected, *, atol, rtol):
    if actual.shape != expected.shape:
        return {
            "passed": False,
            "reason": "shape mismatch",
            "actual_shape": list(actual.shape),
            "expected_shape": list(expected.shape),
        }
    a, b = actual.float(), expected.float()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    delta = (a - b).abs()
    return {
        "passed": finite and bool(torch.allclose(a, b, atol=atol, rtol=rtol)),
        "finite": finite,
        "max_absolute_error": float(delta.max()) if finite else None,
        "mean_absolute_error": float(delta.mean()) if finite else None,
    }


def use_torch_attention(config):
    from olmo_core.nn.attention import AttentionConfig, AttentionBackendName

    if isinstance(config, AttentionConfig):
        config.backend = AttentionBackendName.torch
        config.return_max_logits = False
    if dataclasses.is_dataclass(config):
        for field in dataclasses.fields(config):
            use_torch_attention(getattr(config, field.name))
    elif isinstance(config, (list, tuple)):
        for child in config:
            use_torch_attention(child)
    elif isinstance(config, dict):
        for child in config.values():
            use_torch_attention(child)


def load_native(reader, device, export_dtype, validation_dtype="auto"):
    config = parse_model_config(reader.raw)
    use_torch_attention(config)
    model = config.build(init_device="meta")
    reader.check_native(model)
    model.to_empty(device=device)
    # Every state entry is filled. No random initialization, no optimizer.
    for name, tensor in model.state_dict(keep_vars=True).items():
        saved = reader.read([name])[name]
        converted = target_dtype(native_to_hf_key(name), saved.dtype, export_dtype)
        value = saved.to(dtype=converted)
        if validation_dtype != "auto":
            value = value.to(
                dtype=target_dtype(native_to_hf_key(name), value.dtype, validation_dtype)
            )
        tensor.data = value.to(device=device)
    return model.eval().requires_grad_(False)


def load_hf(path, device, dtype):
    from transformers import AutoModelForCausalLM

    kwargs = {"trust_remote_code": True, "local_files_only": True}
    kwargs["dtype"] = "auto" if dtype == "auto" else DTYPES[dtype]
    # Load the copied runtime through the public AutoClass interface.
    model, loading = AutoModelForCausalLM.from_pretrained(path, output_loading_info=True, **kwargs)
    for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        if loading.get(key):
            raise ValueError(f"HF reload {key}: {loading[key]}")
    for group in model.config.architecture["parameter_aliases"]:
        if len({id(model.get_parameter(key)) for key in group}) != 1:
            raise ValueError(f"HF reload lost parameter sharing: {group}")
    return model.to(device).eval().requires_grad_(False)


def capture(model, names):
    results, handles = {}, []
    for name in names:
        module = model.get_submodule(name)

        def hook(_module, _inputs, output, key=name):
            value = output[0] if isinstance(output, tuple) else output
            if isinstance(value, torch.Tensor):
                results[key] = value.detach().float().cpu()

        handles.append(module.register_forward_hook(hook))
    return results, handles


def compare_models(native, hf, input_ids, *, atol=1e-5, rtol=1e-4, intermediates=True):
    """Compare identical unpadded inputs, then test HF-specific inference inputs."""
    report = {}
    native_results, hf_results, handles = {}, {}, []
    if intermediates:
        names = ["embeddings", "lm_head"]
        for index in native.blocks:
            prefix = f"blocks.{index}"
            names += [prefix, prefix + ".attention", prefix + ".feed_forward"]
            names += [prefix + ".attention." + n for n in ("w_q", "w_k", "w_v", "w_out")]
        native_results, nh = capture(native, names)
        hf_results, hh = capture(
            hf, [native_to_hf_key(n + ".weight").removesuffix(".weight") for n in names]
        )
        handles = nh + hh
    try:
        with torch.inference_mode():
            native_logits = native(input_ids)
            hf_logits = hf(input_ids, use_cache=False).logits
    finally:
        for handle in handles:
            handle.remove()
    report["logits"] = difference(hf_logits, native_logits, atol=atol, rtol=rtol)
    report["intermediates"] = {
        name: difference(
            hf_results[native_to_hf_key(name + ".weight").removesuffix(".weight")],
            value,
            atol=atol,
            rtol=rtol,
        )
        for name, value in native_results.items()
    }
    with torch.inference_mode():
        labels = input_ids[:, 1:]
        native_ce = F.cross_entropy(
            native_logits[:, :-1].float().reshape(-1, native_logits.shape[-1]), labels.reshape(-1)
        )
        hf_ce = F.cross_entropy(
            hf_logits[:, :-1].float().reshape(-1, hf_logits.shape[-1]), labels.reshape(-1)
        )
        report["same_target_ce"] = {
            "native": float(native_ce),
            "hf": float(hf_ce),
            "absolute_error": float((native_ce - hf_ce).abs()),
        }
        report["hf_label_loss"] = difference(
            hf(input_ids, labels=input_ids, use_cache=False).loss, hf_ce, atol=atol, rtol=rtol
        )
        # External embeddings must skip the native embedding scale and norm.
        embeds = native.embeddings(input_ids)
        report["external_embeddings"] = difference(
            hf(inputs_embeds=embeds, use_cache=False).logits,
            native(input_ids, input_embeddings=embeds),
            atol=atol,
            rtol=rtol,
        )
        cached, cache = [], None
        for i in range(input_ids.shape[1]):
            out = hf(input_ids[:, i : i + 1], past_key_values=cache, use_cache=True)
            cached.append(out.logits)
            cache = out.past_key_values
        report["cached_decode"] = difference(
            torch.cat(cached, dim=1), hf_logits, atol=atol, rtol=rtol
        )
        midpoint = max(1, input_ids.shape[1] // 2)
        prefill = hf(input_ids[:, :midpoint], use_cache=True)
        suffix = hf(
            input_ids[:, midpoint:], past_key_values=prefill.past_key_values, use_cache=True
        )
        report["prefill_then_decode"] = difference(
            suffix.logits, hf_logits[:, midpoint:], atol=atol, rtol=rtol
        )
        # Padding is an HF feature; compare only valid tokens with the unpadded reference.
        padded = F.pad(input_ids, (2, 0), value=hf.config.pad_token_id or 0)
        mask = F.pad(torch.ones_like(input_ids), (2, 0), value=0)
        report["left_padding"] = difference(
            hf(padded, attention_mask=mask, use_cache=False).logits[:, 2:],
            hf_logits,
            atol=atol,
            rtol=rtol,
        )
        padded_cached = hf(padded[:, :-1], attention_mask=mask[:, :-1], use_cache=True)
        report["left_padding_cached_decode"] = difference(
            hf(
                padded[:, -1:],
                attention_mask=mask,
                past_key_values=padded_cached.past_key_values,
                use_cache=True,
            ).logits,
            hf_logits[:, -1:],
            atol=atol,
            rtol=rtol,
        )
        report["logits_to_keep"] = difference(
            hf(input_ids, logits_to_keep=2, use_cache=False).logits,
            hf_logits[:, -2:],
            atol=atol,
            rtol=rtol,
        )
        generated = hf.generate(
            input_ids[:, :midpoint],
            min_new_tokens=2,
            max_new_tokens=2,
            pad_token_id=hf.config.pad_token_id or 0,
            do_sample=False,
            use_cache=True,
        )
        report["generate"] = {
            "passed": generated.shape == (input_ids.shape[0], midpoint + 2),
            "shape": list(generated.shape),
        }
    checks = [v["passed"] for k, v in report.items() if isinstance(v, dict) and "passed" in v]
    checks += [v["passed"] for v in report["intermediates"].values()]
    report["passed"] = all(checks)
    return report


def validate_export(
    checkpoint,
    output,
    *,
    config_path=None,
    device="cpu",
    dtype="auto",
    input_ids_path=None,
    sequence_length=16,
    seed=12536,
    atol=1e-5,
    rtol=1e-4,
    intermediates=True,
    report_path=None,
):
    if sequence_length < 2:
        raise ValueError("Validation requires at least two tokens")
    reader = CheckpointReader(checkpoint, config_path)
    folder = Path(output).resolve()
    manifest = json.loads((folder / "conversion_manifest.json").read_text())
    if manifest["source_config_sha256"] != reader.provenance()["source_config_sha256"]:
        raise ValueError("Validation source config differs from export source")
    if manifest.get("dcp_metadata_sha256") != reader.provenance()["dcp_metadata_sha256"]:
        raise ValueError("Validation DCP metadata differs from export source")
    # Cast the native reference like the export before an optional validation dtype.
    native = load_native(reader, device, manifest["dtype"], dtype)
    hf = load_hf(folder, device, dtype)
    weights_checked = 0
    for name, tensor in native.state_dict().items():
        restored = hf.state_dict()[native_to_hf_key(name)]
        if tensor.dtype != restored.dtype or not torch.equal(tensor, restored):
            raise ValueError(f"HF reloaded weight differs from reference: {name}")
        weights_checked += 1
    if input_ids_path:
        ids = torch.tensor(
            json.loads(Path(input_ids_path).read_text()), dtype=torch.long, device=device
        )
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        if ids.ndim != 2 or ids.shape[1] < 2:
            raise ValueError("input IDs JSON must be a token sequence or batch with >=2 tokens")
    else:
        ids = torch.randint(
            2,
            min(hf.config.vocab_size, 50000),
            (1, sequence_length),
            generator=torch.Generator().manual_seed(seed),
        ).to(device)
    if bool((ids < 0).any() or (ids >= hf.config.vocab_size).any()):
        raise ValueError("Validation token ID outside embedding vocabulary")
    report = {
        "checkpoint": str(reader.path),
        "hf_output": str(folder),
        "device": str(device),
        "dtype": dtype,
        "atol": atol,
        "rtol": rtol,
        "input_ids": ids.cpu().tolist(),
        "exact_weight_keys_checked": weights_checked,
        "reference": "Non-FP8 eager native inference using checkpoint master weights",
    }
    report.update(
        compare_models(native, hf, ids, atol=atol, rtol=rtol, intermediates=intermediates)
    )
    saved_report = Path(report_path or folder / "validation_report.json")
    write_json(saved_report, report)
    manifest["numerical_validation"] = {
        "status": "passed" if report["passed"] else "failed",
        "report": str(saved_report.resolve()),
        "report_sha256": sha256(saved_report),
        "dtype": dtype,
        "atol": atol,
        "rtol": rtol,
    }
    write_json(folder / "conversion_manifest.json", manifest)
    if not report["passed"]:
        raise ValueError("Numerical validation failed; inspect validation_report.json")
    return report
