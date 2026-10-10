"""Architecture descriptions shared by conversion, inspection and validation."""

import inspect
from collections import defaultdict

_ADAPTERS = {}
_CONFIG_ADAPTERS = {}


def register_config(native_class):
    """Explicitly unwrap checkpoint-only initialization wrappers, never their weights."""

    def decorate(adapter):
        if native_class in _CONFIG_ADAPTERS:
            raise ValueError(f"Duplicate checkpoint config adapter for {native_class}")
        _CONFIG_ADAPTERS[native_class] = adapter
        return adapter

    return decorate


def prepare_config(config):
    from . import architectures  # noqa: F401

    adapter = _CONFIG_ADAPTERS.get(type(config))
    return config if adapter is None else adapter(config)


def register(native_class):
    """Register an exact type: a new subclass must declare its own export support."""

    def decorate(describe):
        if native_class in _ADAPTERS:
            raise ValueError(f"Duplicate HF adapter for {native_class}")
        _ADAPTERS[native_class] = describe
        return describe

    return decorate


def describe(module):
    if module is None:
        return None
    adapter = _ADAPTERS.get(type(module))
    if adapter is None:
        name = f"{type(module).__module__}.{type(module).__name__}"
        raise ValueError(
            f"No HF architecture adapter for {name}; add one under hf_export/architectures"
        )
    spec = adapter(module)
    spec["native_class"] = f"{type(module).__module__}.{type(module).__name__}"
    spec["native_forward"] = str(inspect.signature(type(module).forward))
    return spec


def children(module):
    return {name: describe(child) for name, child in module._modules.items() if child is not None}


def native_to_hf_key(name):
    if name.startswith(("embeddings.", "embedding_norm.", "blocks.")):
        return "model." + name
    if name.startswith("lm_head."):
        return name
    raise ValueError(f"Unmapped model state key: {name}")


def parameter_aliases(model):
    groups = defaultdict(list)
    for name, parameter in model.named_parameters(remove_duplicate=False):
        groups[id(parameter)].append(native_to_hf_key(name))
    return [names for names in groups.values() if len(names) > 1]


def describe_model(model):
    # These imports register supported exact types. There is no fallback to Dense.
    from . import architectures  # noqa: F401
    from olmo_core.nn.transformer.model import Transformer
    from olmo_core.nn.lm_head import LMHead

    if type(model) is not Transformer or type(model.lm_head) is not LMHead:
        raise ValueError("Only the registered Transformer / LMHead family is currently supported")
    blocks = [describe(block) for block in model.blocks.values()]
    if len(blocks) != model.n_layers:
        raise ValueError("Export requires all layers, not a pipeline fragment")
    return {
        "embeddings": describe(model.embeddings),
        "embedding_norm": describe(model.embedding_norm),
        "embed_scale": model.embed_scale,
        "blocks": blocks,
        "lm_head": {"norm": describe(model.lm_head.norm), "w_out": describe(model.lm_head.w_out)},
        "parameter_aliases": parameter_aliases(model),
    }


def registered_architectures():
    from . import architectures  # noqa: F401

    return sorted(f"{cls.__module__}.{cls.__name__}" for cls in (*_ADAPTERS, *_CONFIG_ADAPTERS))
