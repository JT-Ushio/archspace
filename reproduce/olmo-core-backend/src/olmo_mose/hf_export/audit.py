"""Meta-device structural and forward-contract audits; no numerical execution."""

import inspect
from collections import Counter

import torch

from .registry import describe_model, native_to_hf_key


def make_hf_config(native, raw):
    from .checkpoint import tokenizer_fields
    from .configuration_mose import MoSEConfig

    tokens = tokenizer_fields(raw)
    max_length = raw.get("train_module", {}).get("max_sequence_length", 8192)
    return MoSEConfig(
        architecture=describe_model(native),
        hidden_size=native.d_model,
        vocab_size=native.embeddings.num_embeddings,
        max_position_embeddings=max_length,
        **{k: v for k, v in tokens.items() if k != "identifier"},
    )


def build_hf_meta(config):
    from .modeling_mose import MoSEForCausalLM
    from transformers.modeling_utils import no_init_weights

    with no_init_weights(), torch.device("meta"):
        hf = MoSEForCausalLM(config)
        hf.tie_weights()
    return hf


def structural_audit(native, config):
    hf = build_hf_meta(config)
    source = {native_to_hf_key(k): tuple(v.shape) for k, v in native.state_dict().items()}
    target = {k: tuple(v.shape) for k, v in hf.state_dict().items()}
    if source != target:
        raise ValueError(
            f"HF structure mismatch: missing={sorted(set(source) - set(target))}, "
            f"extra={sorted(set(target) - set(source))}, "
            f"shapes={[k for k in source.keys() & target.keys() if source[k] != target[k]]}"
        )
    for group in config.architecture["parameter_aliases"]:
        if len({id(hf.get_parameter(k)) for k in group}) != 1:
            raise ValueError(f"HF parameter sharing lost: {group}")
    native_count = sum(p.numel() for p in native.parameters())
    hf_count = sum(p.numel() for p in hf.parameters())
    if native_count != hf_count:
        raise ValueError(f"Unique parameter counts differ: {native_count} != {hf_count}")
    classes = Counter()
    signatures = {}

    def visit(value):
        if not isinstance(value, dict):
            return
        if "native_class" in value:
            classes[value["native_class"]] += 1
            signatures[value["native_class"]] = value["native_forward"]
        for child in value.get("children", {}).values():
            visit(child)

    for block in config.architecture["blocks"]:
        visit(block)
    return {
        "state_keys": len(source),
        "unique_parameters": native_count,
        "sharing_groups": len(config.architecture["parameter_aliases"]),
        "native_forward": str(inspect.signature(native.forward)),
        "hf_forward": str(inspect.signature(hf.forward)),
        "module_classes": dict(classes),
        "module_forward_signatures": signatures,
        "input_mapping": {
            "input_ids": "input_ids",
            "input_embeddings": "inputs_embeds",
            "logits_to_keep": "logits_to_keep",
        },
        "label_contract": "Native labels are shifted; HF labels are unshifted (-100 ignored)",
        "unsupported_native_inputs": [
            "doc_lens",
            "cu_doc_lens",
            "max_doc_len",
            "local_k_slice",
            "pos_sin",
            "pos_cos",
            "freqs_cis",
            "cache_leftpad",
            "context_parallelism",
            "z_loss_multiplier",
            "loss_div_factor",
        ],
        "structural_check": "passed",
        "numerical_check": "not_run",
    }
