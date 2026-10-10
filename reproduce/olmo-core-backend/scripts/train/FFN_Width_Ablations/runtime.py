"""Local orchestration using the established, authorized submission gateway."""

import importlib.util
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_helper(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


status_helper = load_helper("width_shared_status", HERE.parent / "UG_Partial_Sharing/monitor.py")
status_helper.OUT = Path("/tmp/mose_ffn_width_monitor")


def sanitize(raw):
    raw = re.sub(r"wandb_v1_[A-Za-z0-9_-]+", "[REDACTED]", raw)
    return re.sub(r"(WANDB_(?:API_KEY|SJY)=)[^\s]+", r"\1[REDACTED]", raw)


def remote(command):
    code, raw = status_helper.remote(command)
    return code, sanitize(raw)
