"""Record all initial parameter bytes once, without changing training state."""

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.distributed.tensor import DTensor

from olmo_core.distributed.utils import get_rank
from olmo_core.train.callbacks import Callback, CallbackConfig

log = logging.getLogger(__name__)


class InitializationFingerprintCallback(Callback):
    @torch.no_grad()
    def pre_train(self):
        if self.step != 0:
            log.info("Skipping initialization fingerprint for resumed step %s", self.step)
            return
        main_rank = get_rank() == 0
        aggregate = hashlib.sha256()
        rows = []
        for name, parameter in sorted(self.trainer.train_module.model.named_parameters()):
            value = parameter.detach()
            # Every rank participates, including empty parameter shards.
            if isinstance(value, DTensor):
                value = value.full_tensor()
            if main_rank:
                value = value.cpu().contiguous()
                row = {"name": name, "shape": list(value.shape), "dtype": str(value.dtype),
                       "numel": value.numel()}
                metadata = json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
                aggregate.update(metadata)
                parameter_hash = hashlib.sha256()
                # View as bytes also works for BF16; do not convert values or round.
                for chunk in value.reshape(-1).view(torch.uint8).split(4_194_304):
                    data = memoryview(chunk.numpy())
                    aggregate.update(data)
                    parameter_hash.update(data)
                row["sha256"] = parameter_hash.hexdigest()
                rows.append(row)
            del value
        if not main_rank:
            return
        summary = {"parameter_sha256": aggregate.hexdigest(), "parameter_tensors": len(rows),
                   "parameter_elements": sum(row["numel"] for row in rows),
                   "step": self.step, "scope": "all parameter bytes before dry-run and optimizer updates"}
        path = Path(self.trainer.save_folder) / "initialization_fingerprint.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({**summary, "parameters": rows}, indent=2) + "\n")
        log.info("INITIALIZATION_FINGERPRINT %s", json.dumps(summary, sort_keys=True))


@dataclass
class InitializationFingerprintCallbackConfig(CallbackConfig):
    def build(self, trainer):
        del trainer
        return InitializationFingerprintCallback()
