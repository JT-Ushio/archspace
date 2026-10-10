"""One checkpoint reader for every registered architecture (model tensors only)."""

import copy
import hashlib
import json
import re
from pathlib import Path

import torch
from olmo_core.nn.transformer import TransformerConfig


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_checkpoint(path, config_path=None):
    """Exact step, stage1/run directory, or a model-only .pt/.safetensors file.

    An incomplete latest step is an error; never quietly export an older step.
    """
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_dir():
        if path.name == "model_and_optim":
            path = path.parent
        if not re.fullmatch(r"step\d+", path.name) and not (path / "config.json").exists():
            root = path / "stage1" if (path / "stage1").is_dir() else path
            steps = [p for p in root.iterdir() if p.is_dir() and re.fullmatch(r"step\d+", p.name)]
            if not steps:
                raise ValueError(f"No step directories under {root}")
            path = max(steps, key=lambda p: int(p.name[4:]))
        if (path / "model_and_optim" / ".metadata").is_file():
            state = path / "model_and_optim"
        elif (path / "model.safetensors").is_file():
            state = path / "model.safetensors"
        elif (path / "model.pt").is_file():
            state = path / "model.pt"
        else:
            raise FileNotFoundError(f"Incomplete/unsupported checkpoint: {path}")
        config = Path(config_path) if config_path else path / "config.json"
    else:
        if path.suffix not in (".pt", ".pth", ".safetensors"):
            raise ValueError("Expected a DCP directory or model-only .pt/.safetensors file")
        state = path
        config = Path(config_path) if config_path else path.parent / "config.json"
    config = config.expanduser().resolve()
    if not config.is_file():
        raise FileNotFoundError(f"Saved model configuration required: {config}")
    return path, state, config


def parse_model_config(raw):
    # Import once to resolve historical serialized olmo_core aliases as well.
    import olmo_mose  # noqa: F401
    from .registry import prepare_config

    model_dict = raw.get("model", raw)
    return prepare_config(TransformerConfig.from_dict(copy.deepcopy(model_dict)))


def build_native_meta(raw):
    return parse_model_config(raw).build(init_device="meta")


def tokenizer_fields(raw):
    tokenizer = raw.get("dataset", raw.get("data", {})).get("tokenizer", {}) or {}
    return {
        name: tokenizer[name]
        for name in ("identifier", "eos_token_id", "pad_token_id", "bos_token_id")
        if name in tokenizer and tokenizer[name] is not None
    }


class CheckpointReader:
    def __init__(self, path, config_path=None):
        self.path, self.state_path, self.config_path = resolve_checkpoint(path, config_path)
        self.raw = json.loads(self.config_path.read_text())
        self.flat = None
        self.metadata = None
        if self.state_path.is_dir():
            from olmo_core.distributed.checkpoint import get_checkpoint_metadata

            self.metadata = get_checkpoint_metadata(self.state_path)
            self.tensors = {}
            for name, meta in self.metadata.state_dict_metadata.items():
                if name.startswith("model."):
                    if not hasattr(meta, "size"):
                        raise TypeError(f"Non-tensor model state is unsupported: {name}")
                    self.tensors[name[6:]] = (tuple(meta.size), meta.properties.dtype)
            if not self.tensors:
                raise ValueError("No model.* tensors in DCP checkpoint")
        elif self.state_path.suffix == ".safetensors":
            from safetensors import safe_open

            # Headers give shapes/dtypes without reading the tensor payloads.
            self.tensors = {}
            dtype_names = {
                "F64": torch.float64,
                "F32": torch.float32,
                "F16": torch.float16,
                "BF16": torch.bfloat16,
                "I64": torch.int64,
                "I32": torch.int32,
                "I16": torch.int16,
                "I8": torch.int8,
                "U8": torch.uint8,
                "BOOL": torch.bool,
            }
            with safe_open(self.state_path, framework="pt", device="cpu") as handle:
                for name in handle.keys():
                    view = handle.get_slice(name)
                    self.tensors[name] = (tuple(view.get_shape()), dtype_names[view.get_dtype()])
        else:
            # Flat model-only unsharded files. mmap avoids a second full copy.
            self.flat = torch.load(
                self.state_path, map_location="cpu", weights_only=True, mmap=True
            )
            if isinstance(self.flat, dict) and set(self.flat) == {"model"}:
                self.flat = self.flat["model"]
            if not isinstance(self.flat, dict) or not all(
                isinstance(t, torch.Tensor) for t in self.flat.values()
            ):
                raise ValueError(
                    "Unsharded input must be a model-only state_dict, not trainer/optimizer state"
                )
            self.tensors = {k: (tuple(v.shape), v.dtype) for k, v in self.flat.items()}

    def check_native(self, model):
        state = model.state_dict()
        if set(state) != set(self.tensors):
            raise ValueError(
                f"Checkpoint keys mismatch: missing={sorted(set(state) - set(self.tensors))}, "
                f"extra={sorted(set(self.tensors) - set(state))}"
            )
        for name, value in state.items():
            if tuple(value.shape) != self.tensors[name][0]:
                raise ValueError(
                    f"Checkpoint shape mismatch for {name}: {self.tensors[name][0]} != {tuple(value.shape)}"
                )

    def read(self, names):
        """Load requested model keys; DCP optimizer tensors are never requested."""
        if self.metadata is not None:
            from olmo_core.distributed.checkpoint import load_keys

            return dict(zip(names, load_keys(self.state_path, ["model." + name for name in names])))
        if self.flat is not None:
            return {name: self.flat[name] for name in names}
        from safetensors import safe_open

        with safe_open(self.state_path, framework="pt", device="cpu") as handle:
            return {name: handle.get_tensor(name) for name in names}

    def provenance(self):
        return {
            "checkpoint": str(self.path),
            "state_path": str(self.state_path),
            "source_config_sha256": sha256(self.config_path),
            "dcp_metadata_sha256": sha256(self.state_path / ".metadata") if self.metadata else None,
            "step": int(self.path.name[4:]) if re.fullmatch(r"step\d+", self.path.name) else None,
        }
