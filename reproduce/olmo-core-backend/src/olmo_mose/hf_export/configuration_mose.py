"""Self-contained HF config; copied alongside weights, independent of OLMo-core."""

from transformers import PretrainedConfig


class MoSEConfig(PretrainedConfig):
    model_type = "olmo_mose"

    def __init__(
        self,
        architecture=None,
        hidden_size=2048,
        vocab_size=50304,
        max_position_embeddings=8192,
        schema_version=1,
        **kwargs,
    ):
        if schema_version != 1:
            raise ValueError(f"Unsupported MoSE HF schema version: {schema_version}")
        kwargs.setdefault("tie_word_embeddings", False)
        kwargs.setdefault("use_cache", True)
        super().__init__(**kwargs)
        self.schema_version = schema_version
        self.architecture = architecture or {}
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.num_hidden_layers = len(self.architecture.get("blocks", []))
        first = self.architecture.get("blocks", [{}])[0] if self.num_hidden_layers else {}
        att = first.get("children", {}).get("attention", {})
        self.num_attention_heads = att.get("n_heads", 1)
        self.num_key_value_heads = att.get("n_kv_heads", 1)
        self.head_dim = att.get("head_dim", hidden_size)


MoSEConfig.register_for_auto_class()
