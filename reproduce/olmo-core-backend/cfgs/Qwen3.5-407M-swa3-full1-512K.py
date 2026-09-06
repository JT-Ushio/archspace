"""Qwen3.5-407M with 3 x 4K SWA + 1 Full; 512K-token batches."""

from functools import partial

from olmo_core.script_utils import main

from _qwen35_407m_entry import build_entry_config
from _qwen35_407m_models import build_qwen35_407m_swa3_full1

build_config = partial(
    build_entry_config,
    model_builder=build_qwen35_407m_swa3_full1,
    variant="swa3-full1",
    profile_name="512K",
)

if __name__ == "__main__":
    main(build_config)
