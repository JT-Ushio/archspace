"""OLMo3-1B stage-1 pretraining with MoSE-SwiGLU and SiTU control."""

import argparse
from typing import List

from olmo_core.script_utils import ExperimentConfig, main

from _models import build_mose_olmo3_1b
from _pretrain_common import build_pretrain_config, get_mose_cli_parser
from olmo_mose import SwiGLUChannelControl


def build_config(opts: argparse.Namespace, overrides: List[str]) -> ExperimentConfig:
    return build_pretrain_config(
        opts,
        overrides,
        lambda tokenizer: build_mose_olmo3_1b(
            tokenizer,
            SwiGLUChannelControl.situ,
            mose_start_layer=getattr(opts, "mose_start_layer", 0),
            mose_end_layer=getattr(opts, "mose_end_layer", None),
        ),
        variant="mose-situ",
    )


if __name__ == "__main__":
    main(build_config, parser=get_mose_cli_parser())
