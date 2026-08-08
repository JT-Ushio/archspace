"""Part-1 OLMo3-7B pretraining with MLA, MorphNorm, and max-logit FA3."""

import argparse
from typing import List

from olmo_core.script_utils import ExperimentConfig, main

from _models import build_mla
from _pretrain_common import build_pretrain_config
from olmo_mla_morphnorm import MLANormType


def build_config(opts: argparse.Namespace, overrides: List[str]) -> ExperimentConfig:
    return build_pretrain_config(
        opts,
        overrides,
        lambda tokenizer: build_mla(tokenizer, MLANormType.morphnorm),
    )


if __name__ == "__main__":
    main(build_config)
