from .attention import MLAAttention, MLAAttentionConfig, MLANormType
from .hooks import install_runtime_hooks
from .max_logits import MaxLogitsAttentionConfig, MaxLogitsFlashAttention3Backend
from .patch import patch_attention_for_max_logits, patch_transformer_config

install_runtime_hooks()

__all__ = [
    "MLAAttention",
    "MLAAttentionConfig",
    "MLANormType",
    "MaxLogitsAttentionConfig",
    "MaxLogitsFlashAttention3Backend",
    "patch_attention_for_max_logits",
    "patch_transformer_config",
]
