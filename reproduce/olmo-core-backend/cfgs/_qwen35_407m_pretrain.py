"""Shared pretraining recipe for the Qwen3.5-style 407M comparison grid."""

import argparse
from dataclasses import dataclass
from typing import Callable, List

from olmo_core.config import DType
from olmo_core.data import (
    DataMix,
    NumpyDataLoaderConfig,
    NumpyFSLDatasetConfig,
    NumpyPaddedFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.float8 import Float8Config
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import SchedulerUnits, WSDSqrtDecay
from olmo_core.script_utils import ExperimentConfig
from olmo_core.train import Duration, TrainerConfig
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    CometCallback,
    ConfigSaverCallback,
    DownstreamEvaluatorCallbackConfig,
    LMEvaluatorCallbackConfig,
    MonkeyPatcherCallback,
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)

from olmo_ahn import SerializableMuonConfig

SEQUENCE_LENGTH = 8_192
LR_MAX = 2e-3
WARMUP_TOKENS = 33_554_432
DECAY_FRACTION = 0.1
LR_FINAL_FACTOR = 0.1
RANK_MICROBATCH_SIZE = 4 * SEQUENCE_LENGTH

EVAL_LM_STEPS = 500
EVAL_DOWN_STEPS = 12_500
SAVE_STEPS = 1_000


@dataclass(frozen=True)
class TrainingProfile:
    name: str
    global_batch_size: int
    max_steps: int

    @property
    def max_tokens(self) -> int:
        return self.global_batch_size * self.max_steps


TRAINING_PROFILES = {
    "256K": TrainingProfile(
        name="256K",
        global_batch_size=256 * 1_024,
        max_steps=3_915,
    ),
    "512K": TrainingProfile(
        name="512K",
        global_batch_size=512 * 1_024,
        max_steps=6_526,
    ),
}

ModelBuilder = Callable[[TokenizerConfig], TransformerConfig]


def build_qwen35_407m_pretrain_config(
    opts: argparse.Namespace,
    overrides: List[str],
    model_builder: ModelBuilder,
    *,
    variant: str,
    profile_name: str,
) -> ExperimentConfig:
    """Build one exact model/layout and batch/step comparison point."""
    profile = TRAINING_PROFILES[profile_name]
    sequence_length = SEQUENCE_LENGTH if opts.sequence_length is None else opts.sequence_length
    if sequence_length != SEQUENCE_LENGTH:
        raise OLMoConfigurationError(
            f"Qwen3.5-407M recipes require sequence_length={SEQUENCE_LENGTH}"
        )
    if profile.global_batch_size % RANK_MICROBATCH_SIZE != 0:
        raise OLMoConfigurationError("global batch size must be divisible by rank microbatch size")

    tokenizer_config = TokenizerConfig.dolma2()
    dataset_config = NumpyFSLDatasetConfig.from_data_mix(
        DataMix.OLMo_mix_0625_150Bsample,
        tokenizer=tokenizer_config,
        mix_base_dir=opts.data_root,
        sequence_length=SEQUENCE_LENGTH,
        max_target_sequence_length=SEQUENCE_LENGTH,
        work_dir=opts.work_dir,
    )
    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=profile.global_batch_size,
        seed=34521,
        num_workers=4,
    )
    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=RANK_MICROBATCH_SIZE,
        max_sequence_length=SEQUENCE_LENGTH,
        optim=SerializableMuonConfig(
            lr=LR_MAX,
        ),
        scheduler=WSDSqrtDecay(
            units=SchedulerUnits.tokens,
            warmup=WARMUP_TOKENS,
            decay_fraction=DECAY_FRACTION,
            decay_min_lr_ratio=LR_FINAL_FACTOR,
        ),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.blocks,
        ),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=None,
        max_grad_norm=1.0,
    )
    trainer_config = (
        TrainerConfig(
            save_folder=opts.save_folder,
            save_overwrite=True,
            metrics_collect_interval=10,
            cancel_check_interval=10,
            max_duration=Duration.steps(profile.max_steps),
        )
        .with_callback("monkey_patcher", MonkeyPatcherCallback())
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=SAVE_STEPS,
                ephemeral_save_interval=None,
                max_checkpoints=1,
                save_async=True,
            ),
        )
        .with_callback(
            "comet",
            CometCallback(
                name=opts.name,
                cancel_check_interval=10,
                enabled=False,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=opts.name,
                entity="archspace",
                project="AHN",
                group=f"qwen35-407m-{variant}-{profile.name.lower()}",
                cancel_tags=[],
                enabled=True,
            ),
        )
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback(
            "lm_evaluator",
            LMEvaluatorCallbackConfig(
                eval_dataset=NumpyPaddedFSLDatasetConfig.from_data_mix(
                    DataMix.v3_small_ppl_validation,
                    mix_base_dir=opts.data_root,
                    sequence_length=SEQUENCE_LENGTH,
                    tokenizer=tokenizer_config,
                    work_dir=opts.work_dir,
                ),
                eval_interval=EVAL_LM_STEPS,
            ),
        )
        .with_callback(
            "downstream_evaluator",
            DownstreamEvaluatorCallbackConfig(
                tasks=sorted(
                    [
                        "arc_challenge_test_bpb_5shot",
                        "arc_challenge_test_mc_5shot_fast",
                        "arc_easy_test_bpb_5shot",
                        "arc_easy_test_mc_5shot_fast",
                        "hellaswag_bpb_5shot",
                        "mmlu_humanities_test_bpb_5shot",
                        "mmlu_humanities_test_mc_5shot_fast",
                        "mmlu_other_test_bpb_5shot",
                        "mmlu_other_test_mc_5shot_fast",
                        "mmlu_social_sciences_test_bpb_5shot",
                        "mmlu_social_sciences_test_mc_5shot_fast",
                        "mmlu_stem_test_bpb_5shot",
                        "mmlu_stem_test_mc_5shot_fast",
                    ]
                ),
                tokenizer=tokenizer_config,
                eval_interval=EVAL_DOWN_STEPS,
            ),
        )
    )

    config = ExperimentConfig(
        model=model_builder(tokenizer_config),
        dataset=dataset_config,
        data_loader=data_loader_config,
        train_module=train_module_config,
        trainer=trainer_config,
    )
    config.train_module.optim = config.train_module.optim.merge(
        [
            "weight_decay=0.1",
            "mu=0.95",
            "adjust_lr=rms_norm",
            "use_polar_express=false",
        ]
    )
    return config.merge(overrides)
