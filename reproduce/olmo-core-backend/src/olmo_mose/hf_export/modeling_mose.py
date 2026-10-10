"""Standalone Transformers CausalLM for the registered MoSE architecture families."""

import re
import torch
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel, GenerationMixin
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_mose import MoSEConfig
from .runtime_ops import build_op


class MoSEBackbone(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.embeddings = build_op(spec["embeddings"])
        self.embedding_norm = build_op(spec["embedding_norm"])
        self.embed_scale = spec["embed_scale"]
        self.blocks = nn.ModuleDict({str(i): build_op(b) for i, b in enumerate(spec["blocks"])})


class MoSELMHead(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.norm = build_op(spec["norm"])
        self.w_out = build_op(spec["w_out"])

    def forward(self, hidden):
        return self.w_out(hidden if self.norm is None else self.norm(hidden))


class MoSEForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MoSEConfig
    base_model_prefix = "model"
    main_input_name = "input_ids"
    _no_split_modules = ["ExportBlock"]
    _supports_cache_class = True
    _supports_static_cache = False
    _supports_sdpa = False  # Runtime selects SDPA itself; no HF backend replacement.
    _keep_in_fp32_modules = []
    _keep_in_fp32_modules_strict = []

    def __init__(self, config):
        super().__init__(config)
        if not config.architecture:
            raise ValueError("A resolved architecture description is required")
        self.model = MoSEBackbone(config.architecture)
        self.lm_head = MoSELMHead(config.architecture["lm_head"])
        parts = {part for name, _ in self.named_parameters() for part in name.split(".")}
        self._keep_in_fp32_modules = [
            name for name in ("alpha", "output_subspace") if name in parts
        ]
        self._keep_in_fp32_modules_strict = self._keep_in_fp32_modules
        self._tied_weights_keys = [
            re.escape(alias)
            for group in config.architecture["parameter_aliases"]
            for alias in group[1:]
        ]
        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def tie_weights(self):
        super().tie_weights()
        for group in self.config.architecture.get("parameter_aliases", []):
            parameter = self.get_parameter(group[0])
            for alias in group[1:]:
                parent, leaf = alias.rsplit(".", 1)
                setattr(self.get_submodule(parent), leaf, parameter)

    def get_input_embeddings(self):
        return self.model.embeddings

    def set_input_embeddings(self, value):
        self.model.embeddings = value

    def get_output_embeddings(self):
        return self.lm_head.w_out

    def set_output_embeddings(self, value):
        self.lm_head.w_out = value

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
        logits_to_keep=0,
        **kwargs,
    ):
        if kwargs:
            raise TypeError(f"Unsupported HF forward arguments: {sorted(kwargs)}")
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Provide exactly one of input_ids and inputs_embeds")
        use_cache = self.config.use_cache if use_cache is None else use_cache
        output_attentions = (
            self.config.output_attentions if output_attentions is None else output_attentions
        )
        output_hidden_states = (
            self.config.output_hidden_states
            if output_hidden_states is None
            else output_hidden_states
        )
        return_dict = self.config.use_return_dict if return_dict is None else return_dict
        if past_key_values is not None and not isinstance(past_key_values, Cache):
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        if past_key_values is not None and type(past_key_values) is not DynamicCache:
            raise TypeError("This runtime supports DynamicCache and legacy tuples only")
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()
        if not use_cache and past_key_values is not None:
            raise ValueError("past_key_values requires use_cache=True")
        hidden = self.model.embeddings(input_ids) if inputs_embeds is None else inputs_embeds
        # Native OLMo skips embed scaling / norm when external embeddings are supplied.
        if inputs_embeds is None:
            if self.model.embed_scale is not None:
                hidden = hidden * self.model.embed_scale
            if self.model.embedding_norm is not None:
                hidden = self.model.embedding_norm(hidden)
        past_length = 0 if past_key_values is None else past_key_values.get_seq_length()
        length = hidden.shape[1]
        expected_positions = torch.arange(past_length, past_length + length, device=hidden.device)
        if cache_position is None:
            cache_position = expected_positions
        if cache_position.ndim != 1 or not torch.equal(cache_position, expected_positions):
            raise ValueError(
                "DynamicCache requires contiguous cache_position starting at past length"
            )
        if position_ids is None:
            if attention_mask is not None and attention_mask.ndim == 2:
                position_ids = (attention_mask.long().cumsum(-1) - 1).clamp_min(0)[:, -length:]
            else:
                position_ids = cache_position.unsqueeze(0)
        states, attentions = (), ()
        for index, block in enumerate(self.model.blocks.values()):
            if output_hidden_states:
                states += (hidden,)
            hidden, weights = block(
                hidden,
                position_ids=position_ids,
                cache_position=cache_position,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                layer_idx=index,
                output_attentions=output_attentions,
            )
            if output_attentions:
                attentions += (weights,)
        if output_hidden_states:
            states += (hidden,)
        if isinstance(logits_to_keep, int) and logits_to_keep < 0:
            raise ValueError("logits_to_keep cannot be negative")
        keep = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        if labels is not None and (not isinstance(logits_to_keep, int) or logits_to_keep != 0):
            raise ValueError("labels requires full logits (logits_to_keep=0)")
        logits = self.lm_head(hidden[:, keep, :])
        loss = None
        if labels is not None:
            # HF labels are unshifted input token IDs; native trainer labels are already shifted.
            loss = F.cross_entropy(
                logits[:, :-1].float().reshape(-1, self.config.vocab_size),
                labels[:, 1:].reshape(-1).to(logits.device),
                ignore_index=-100,
            )
        result = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=states if output_hidden_states else None,
            attentions=attentions if output_attentions else None,
        )
        return result if return_dict else result.to_tuple()


MoSEForCausalLM.register_for_auto_class("AutoModelForCausalLM")
