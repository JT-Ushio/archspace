"""Qwen3.5-407M with 4K SWA in all layers; 256K-token batches."""

from functools import partial

from olmo_core.script_utils import main

from _qwen35_407m_entry import build_entry_config
from _qwen35_407m_models import build_qwen35_407m_all_swa

build_config = partial(
    build_entry_config,
    model_builder=build_qwen35_407m_all_swa,
    variant="all-swa",
    profile_name="256K",
)

if __name__ == "__main__":
    main(build_config)
