"""OLMo3-1B stage-1 pretraining with Asymmetric RationalClip SwiGLU."""

import argparse
from typing import List

from olmo_core.script_utils import ExperimentConfig, main

from _models import build_olmo3_1b
from _pretrain_common import build_pretrain_config
from olmo_mose import SwiGLUChannelControl


def build_config(opts: argparse.Namespace, overrides: List[str]) -> ExperimentConfig:
    return build_pretrain_config(
        opts,
        overrides,
        lambda tokenizer: build_olmo3_1b(
            tokenizer, SwiGLUChannelControl.asymmetric_rational_clip
        ),
        variant="asymmetric-rational-clip",
    )


if __name__ == "__main__":
    main(build_config)
