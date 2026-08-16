"""OLMo3-1B stage-1 pretraining with Cubit KRR token mixers."""

import argparse
from typing import List

from olmo_core.script_utils import ExperimentConfig, main

from _models import build_cubit_olmo3_1b
from _pretrain_common import build_pretrain_config, get_cubit_cli_parser


def build_config(opts: argparse.Namespace, overrides: List[str]) -> ExperimentConfig:
    return build_pretrain_config(
        opts,
        overrides,
        lambda tokenizer: build_cubit_olmo3_1b(
            tokenizer,
            cubit_start_layer=getattr(opts, "cubit_start_layer", 0),
            cubit_end_layer=getattr(opts, "cubit_end_layer", None),
        ),
        variant="cubit-fa3",
    )


if __name__ == "__main__":
    main(build_config, parser=get_cubit_cli_parser())
