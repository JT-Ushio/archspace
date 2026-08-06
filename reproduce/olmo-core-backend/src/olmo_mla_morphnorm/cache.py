import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


class MLACacheManager(nn.Module):
    """Inference cache for absorbed, single-KV-head MLA decoding."""

    def __init__(
        self,
        *,
        batch_size: int,
        max_seq_len: int,
        latent_dim: int,
        rope_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.rope_dim = rope_dim
        self._batch_size = batch_size
        self._max_seq_len = max_seq_len
        self._has_data = False
        self._allocate(batch_size, max_seq_len, device=device, dtype=dtype)

    def _allocate(
        self,
        batch_size: int,
        max_seq_len: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        for name in ("latent_cache", "rope_key_cache", "cache_leftpad", "cache_seqlens"):
            if name in self._buffers:
                delattr(self, name)
        latent_shape = (batch_size, max_seq_len, self.latent_dim)
        rope_shape = (batch_size, max_seq_len, self.rope_dim)
        self.register_buffer(
            "latent_cache",
            torch.zeros(latent_shape, device=device, dtype=dtype),
            persistent=False,
        )
        self.register_buffer(
            "rope_key_cache",
            torch.zeros(rope_shape, device=device, dtype=dtype),
            persistent=False,
        )
        self.register_buffer(
            "cache_leftpad",
            torch.zeros(batch_size, device=device, dtype=torch.int32),
            persistent=False,
        )
        self.register_buffer(
            "cache_seqlens",
            torch.zeros((), device=device, dtype=torch.int32),
            persistent=False,
        )

    def current_position(self) -> torch.Tensor:
        """Return the physical write position in the cache."""
        return self.cache_seqlens

    @property
    def has_data(self) -> bool:
        """Return whether at least one prompt/decode chunk has been appended."""
        return self._has_data

    def ensure_compatible(self, value: torch.Tensor) -> None:
        """Adopt the first projected latent's runtime device and compute dtype."""
        if self.latent_cache.device == value.device and self.latent_cache.dtype == value.dtype:
            return
        if self._has_data:
            raise RuntimeError(
                "MLA cache device/dtype changed after prefill: "
                f"cache={self.latent_cache.device}/{self.latent_cache.dtype}, "
                f"value={value.device}/{value.dtype}"
            )
        self._allocate(
            self._batch_size,
            self._max_seq_len,
            device=value.device,
            dtype=value.dtype,
        )

    def record_leftpad(self, leftpad: Optional[torch.Tensor]) -> None:
        """Record prompt left-padding widths for subsequent decode masks."""
        if leftpad is not None:
            if leftpad.shape != self.cache_leftpad.shape:
                raise ValueError(
                    f"Expected cache_leftpad shape {tuple(self.cache_leftpad.shape)}, "
                    f"got {tuple(leftpad.shape)}"
                )
            self.cache_leftpad.copy_(leftpad)

    @torch.compiler.disable
    def append(
        self,
        latent: torch.Tensor,
        rope_key: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Append a chunk and return active cache views.

        Inputs have shapes ``[B, T, latent_dim]`` and ``[B, T, rope_dim]``.
        """
        batch_size, seq_len, latent_dim = latent.shape
        if batch_size != self._batch_size:
            raise ValueError(f"Expected batch size {self._batch_size}, got {batch_size}")
        if latent_dim != self.latent_dim:
            raise ValueError("Invalid MLA latent cache input shape")
        if rope_key.shape != (batch_size, seq_len, self.rope_dim):
            raise ValueError(
                f"Expected RoPE key shape {(batch_size, seq_len, self.rope_dim)}, "
                f"got {tuple(rope_key.shape)}"
            )

        start = int(self.cache_seqlens.item())
        end = start + seq_len
        if end > self._max_seq_len:
            raise RuntimeError(
                f"MLA cache capacity exceeded: need position {end}, max is {self._max_seq_len}"
            )

        self.latent_cache[:, start:end].copy_(latent)
        self.rope_key_cache[:, start:end].copy_(rope_key)
        self.cache_seqlens.add_(seq_len)
        self._has_data = True
        return (
            self.latent_cache[:, :end],
            self.rope_key_cache[:, :end],
        )

    def zero_cache(self) -> None:
        """Clear all cache contents and positions."""
        self.latent_cache.zero_()
        self.rope_key_cache.zero_()
        self.cache_leftpad.zero_()
        self.cache_seqlens.zero_()
        self._has_data = False

    def is_reusable(self, batch_size: int, max_seq_len: int) -> bool:
        """Return whether the current allocation can serve a generation request."""
        return self._batch_size == batch_size and self._max_seq_len >= max_seq_len

    def reallocate(self, batch_size: int, max_seq_len: int) -> None:
        """Reallocate cache buffers for a new batch or a longer sequence."""
        self._batch_size = batch_size
        self._max_seq_len = max_seq_len
        self._has_data = False
        self._allocate(
            batch_size,
            max_seq_len,
            device=self.latent_cache.device,
            dtype=self.latent_cache.dtype,
        )

    def reset(self, batch_size: int, max_seq_len: int) -> None:
        """Reset or resize the cache for a new generation request."""
        if self.is_reusable(batch_size, max_seq_len):
            self.zero_cache()
        else:
            log.debug("Unreusable MLA cache, reallocating")
            self.reallocate(batch_size, max_seq_len)
