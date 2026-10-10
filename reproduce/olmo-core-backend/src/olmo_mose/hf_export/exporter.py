"""Sharded, atomic HF export independent of experiment names and rank choices."""

import json
import shutil
import tempfile
from pathlib import Path

import torch

from .audit import build_hf_meta, make_hf_config, structural_audit
from .checkpoint import CheckpointReader, build_native_meta, tokenizer_fields, write_json
from .registry import native_to_hf_key

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def target_dtype(name, source_dtype, dtype):
    if not source_dtype.is_floating_point or dtype == "auto":
        return source_dtype
    if ".output_subspace." in name or name.endswith(".alpha"):
        return torch.float32
    return DTYPES[dtype]


def parse_size(value):
    if isinstance(value, int):
        result = value
    else:
        text = str(value).strip().upper()
        units = {
            "GIB": 1024**3,
            "MIB": 1024**2,
            "KIB": 1024,
            "GB": 1000**3,
            "MB": 1000**2,
            "KB": 1000,
            "B": 1,
        }
        result = None
        for unit, multiplier in units.items():
            if text.endswith(unit):
                result = int(float(text[: -len(unit)]) * multiplier)
                break
        if result is None:
            result = int(text)
    if result <= 0:
        raise ValueError("max_shard_size must be positive")
    return result


def plan_shards(reader, aliases, dtype, limit):
    groups = {group[0]: group for group in aliases}
    omitted = {name for group in aliases for name in group[1:]}
    mapping = {native_to_hf_key(name): name for name in reader.tensors}
    shards, current, size = [], [], 0
    for hf_name, source_name in sorted(mapping.items()):
        if hf_name in omitted:
            continue
        shape, saved_dtype = reader.tensors[source_name]
        converted_dtype = target_dtype(hf_name, saved_dtype, dtype)
        count = 1
        for dim in shape:
            count *= dim
        nbytes = count * torch.empty((), dtype=converted_dtype).element_size()
        if current and size + nbytes > limit:
            shards.append(current)
            current, size = [], 0
        current.append((hf_name, [mapping[n] for n in groups.get(hf_name, [hf_name])]))
        size += nbytes
    if current:
        shards.append(current)
    return shards


def save_tokenizer(
    folder, config, raw, tokenizer=None, skip_tokenizer=False, local_files_only=False
):
    if skip_tokenizer:
        return {"saved": False, "reason": "explicit --skip-tokenizer"}
    from transformers import AutoTokenizer

    identifier = tokenizer or tokenizer_fields(raw).get("identifier")
    if not identifier:
        raise ValueError(
            "Saved config has no tokenizer identifier; provide --tokenizer or --skip-tokenizer"
        )
    tok = AutoTokenizer.from_pretrained(identifier, local_files_only=local_files_only)
    for name in ("eos_token_id", "pad_token_id", "bos_token_id"):
        expected = getattr(config, name, None)
        if expected is None:
            expected = getattr(tok, name, None)
            setattr(config, name, expected)
        if expected is None:
            continue
        if not 0 <= expected < len(tok):
            raise ValueError(f"{name}={expected} is absent from tokenizer {identifier}")
        if getattr(tok, name) != expected:
            token = tok.convert_ids_to_tokens(expected)
            setattr(tok, name.removesuffix("_id"), token)
        if getattr(tok, name) != expected:
            raise ValueError(f"Tokenizer/config mismatch for {name}")
    if len(tok) > config.vocab_size:
        raise ValueError("Tokenizer vocabulary exceeds model embedding rows")
    tok.save_pretrained(folder)
    return {"saved": True, "identifier": str(identifier), "vocab_size": len(tok)}


def export_checkpoint(
    checkpoint,
    output,
    *,
    config_path=None,
    dtype="auto",
    max_shard_size="2GB",
    tokenizer=None,
    skip_tokenizer=False,
    local_files_only=False,
):
    from safetensors.torch import save_file
    from transformers import GenerationConfig
    from transformers.dynamic_module_utils import custom_object_save

    if dtype not in ("auto", *DTYPES):
        raise ValueError(f"Unsupported output dtype: {dtype}")
    reader = CheckpointReader(checkpoint, config_path)
    native = build_native_meta(reader.raw)
    reader.check_native(native)
    config = make_hf_config(native, reader.raw)
    audit = structural_audit(native, config)
    config.architectures = ["MoSEForCausalLM"]
    shards = plan_shards(
        reader, config.architecture["parameter_aliases"], dtype, parse_size(max_shard_size)
    )
    # The load dtype follows ordinary projection tensors. Small FP32 controllers
    # are separately retained by the runtime's _keep_in_fp32_modules.
    source_dtype = reader.tensors["embeddings.weight"][1]
    config.torch_dtype = target_dtype("model.embeddings.weight", source_dtype, dtype)
    output = Path(output).expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Refusing to overwrite HF output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.export-", dir=output.parent))
    try:
        skeleton = build_hf_meta(config)
        # Copies only standalone runtime files, never the native training package.
        custom_object_save(skeleton, staging, config=config)
        custom_object_save(config, staging, config=config)
        token_info = save_tokenizer(
            staging, config, reader.raw, tokenizer, skip_tokenizer, local_files_only
        )
        config.save_pretrained(staging)
        GenerationConfig.from_model_config(config).save_pretrained(staging)
        weight_map, total_size = {}, 0
        dtype_map = {}
        for index, shard in enumerate(shards, 1):
            tensors = {}
            for name, source_names in shard:
                # Read aliases separately, so peak memory remains a shard plus
                # one tensor rather than a full model or full optimizer state.
                tensor = reader.read(source_names[:1])[source_names[0]]
                if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                    raise ValueError(f"Nonfinite checkpoint tensor: {source_names[0]}")
                for alias in source_names[1:]:
                    other = reader.read([alias])[alias]
                    if other.dtype != tensor.dtype or not torch.equal(tensor, other):
                        raise ValueError(
                            f"Saved shared parameters differ: {source_names[0]} vs {alias}"
                        )
                    del other
                tensor = (
                    tensor.detach().to(dtype=target_dtype(name, tensor.dtype, dtype)).contiguous()
                )
                if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                    raise ValueError(f"Dtype conversion created nonfinite values: {name}")
                tensors[name] = tensor
                total_size += tensor.numel() * tensor.element_size()
                dtype_map[name] = str(tensor.dtype).removeprefix("torch.")
            filename = (
                "model.safetensors"
                if len(shards) == 1
                else f"model-{index:05d}-of-{len(shards):05d}.safetensors"
            )
            save_file(tensors, staging / filename, metadata={"format": "pt"})
            weight_map.update({name: filename for name in tensors})
            del tensors
        if len(shards) > 1:
            write_json(
                staging / "model.safetensors.index.json",
                {"metadata": {"total_size": total_size}, "weight_map": weight_map},
            )
        manifest = {
            "schema_version": 1,
            **reader.provenance(),
            "output": str(output),
            "dtype": dtype,
            "tensor_dtypes": dtype_map,
            "weight_files": sorted(set(weight_map.values())),
            "stored_tensors": len(weight_map),
            "total_size": total_size,
            "parameter_aliases": config.architecture["parameter_aliases"],
            "tokenizer": token_info,
            "audit": audit,
            "numerical_validation": {"status": "not_run"},
            "export_complete": True,
        }
        write_json(staging / "conversion_manifest.json", manifest)
        # Scheduled jobs can write as root on shared storage. mkdtemp uses
        # 0700, which would make the completed model unreadable to its owner
        # on the development machine after the atomic rename.
        staging.chmod(0o755)
        if output.exists():
            output.rmdir()  # Only a still-empty user-provided directory is replaceable.
        staging.rename(output)
        return manifest
    except BaseException:
        shutil.rmtree(staging)
        raise


def export_batch(manifest_path, **options):
    """A JSON list of {checkpoint, output, config?}; keep a result for every item."""
    path = Path(manifest_path).resolve()
    entries = json.loads(path.read_text())
    if not isinstance(entries, list) or not entries:
        raise ValueError("Batch manifest must be a nonempty JSON list")
    outputs = []
    for entry in entries:
        for key in ("checkpoint", "output"):
            if key not in entry:
                raise ValueError(f"Batch entry missing {key}: {entry}")
            p = Path(entry[key]).expanduser()
            entry[key] = str((p if p.is_absolute() else path.parent / p).resolve())
        if entry.get("config"):
            p = Path(entry["config"]).expanduser()
            entry["config"] = str((p if p.is_absolute() else path.parent / p).resolve())
        outputs.append(entry["output"])
    if len(outputs) != len(set(outputs)):
        raise ValueError("Duplicate output directories in batch manifest")
    results = []
    for entry in entries:
        try:
            result = export_checkpoint(
                entry["checkpoint"], entry["output"], config_path=entry.get("config"), **options
            )
            results.append(
                {**entry, "status": "exported", "parameters": result["audit"]["unique_parameters"]}
            )
        except Exception as exc:
            results.append({**entry, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        write_json(path.with_name(path.stem + ".results.json"), results)
    return results
