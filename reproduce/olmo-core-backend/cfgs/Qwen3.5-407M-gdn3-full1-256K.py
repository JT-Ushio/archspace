"""Qwen3.5-407M with 3 GDN + 1 Full; 256K-token batches for 3,915 steps."""

from functools import partial

from olmo_core.script_utils import main

from _qwen35_407m_entry import build_entry_config
from _qwen35_407m_models import build_qwen35_407m_gdn3_full1

build_config = partial(
    build_entry_config,
    model_builder=build_qwen35_407m_gdn3_full1,
    variant="gdn3-full1",
    profile_name="256K",
)

if __name__ == "__main__":
    main(build_config)
