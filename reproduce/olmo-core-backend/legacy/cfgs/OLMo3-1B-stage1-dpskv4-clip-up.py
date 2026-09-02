"""OLMo3-1B stage-1 pretraining with DPSKV4 clipping on Up only."""

import argparse
from typing import List

from olmo_core.script_utils import ExperimentConfig, main

from _models import build_olmo3_1b
from _pretrain_common import build_pretrain_config
from olmo_mose import SwiGLUChannelControl, SwiGLUChannelControlScope


def build_config(opts: argparse.Namespace, overrides: List[str]) -> ExperimentConfig:
    return build_pretrain_config(
        opts,
        overrides,
        lambda tokenizer: build_olmo3_1b(
            tokenizer,
            SwiGLUChannelControl.dpskv4_clip,
            control_scope=SwiGLUChannelControlScope.up,
        ),
        variant="dpskv4-clip-up",
    )


if __name__ == "__main__":
    main(build_config)
