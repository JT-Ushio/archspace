import argparse
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
from olmo_core.float8 import Float8Config
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import CosWithWarmup, MuonConfig
from olmo_core.script_utils import ExperimentConfig
from olmo_core.train import Duration, TrainerConfig
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    CometCallback,
    ConfigSaverCallback,
    DownstreamEvaluatorCallbackConfig,
    LMEvaluatorCallbackConfig,
    MonkeyPatcherCallback,
)
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)
from olmo_mla_morphnorm import MaxLogitsWandBCallback

DEFAULT_SEQUENCE_LENGTH = 4096
GLOBAL_BATCH_SIZE = 4096 * 512  # ~4M tokens
LR = 3e-2
EVAL_LM_STEPS = 500
EVAL_DOWN_STEPS = 12500
N_BATCH_PER_GPU = 4


ModelBuilder = Callable[[TokenizerConfig], TransformerConfig]


def build_pretrain_config(
    opts: argparse.Namespace,
    overrides: List[str],
    model_builder: ModelBuilder,
) -> ExperimentConfig:
    """Build the official OLMo3-1B stage-1 recipe around a model variant."""
    sequence_length = opts.sequence_length or DEFAULT_SEQUENCE_LENGTH
    tokenizer_config = TokenizerConfig.dolma2()

    dataset_config = NumpyFSLDatasetConfig.from_data_mix(
        DataMix.OLMo_mix_0625_150Bsample,
        tokenizer=tokenizer_config,
        mix_base_dir=opts.data_root,
        sequence_length=sequence_length,
        max_target_sequence_length=max(DEFAULT_SEQUENCE_LENGTH, sequence_length),
        work_dir=opts.work_dir,
    )
    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=GLOBAL_BATCH_SIZE,
        seed=34521,
        num_workers=8,
    )
    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=N_BATCH_PER_GPU * DEFAULT_SEQUENCE_LENGTH,
        max_sequence_length=sequence_length,
        optim=MuonConfig(
            lr=LR,
            weight_decay=0.033,
            betas=(0.9, 0.95),
        ),
        scheduler=CosWithWarmup(warmup_steps=2000),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.blocks,
        ),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=1e-5,
        max_grad_norm=None,
    )
    trainer_config = (
        TrainerConfig(
            save_folder=opts.save_folder,
            save_overwrite=True,
            metrics_collect_interval=10,
            cancel_check_interval=10,
            max_duration=Duration.epochs(1),
        )
        .with_callback("monkey_patcher", MonkeyPatcherCallback())
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=None,
                ephemeral_save_interval=2500,
                max_checkpoints=1,
                save_async=False,
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
            MaxLogitsWandBCallback(
                name=opts.name,
                entity="archspace",
                project="MorphNorm",
                group="olmo3-1b-mla-baseline",
                cancel_tags=[],
                enabled=True,
                max_logits_log_interval=100,
            ),
        )
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback(
            "lm_evaluator",
            LMEvaluatorCallbackConfig(
                eval_dataset=NumpyPaddedFSLDatasetConfig.from_data_mix(
                    DataMix.v3_small_ppl_validation,
                    mix_base_dir=opts.data_root,
                    sequence_length=sequence_length,
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
                        "basic_skills_arithmetic_rc_5shot",
                        "basic_skills_coding_rc_5shot",
                        "basic_skills_common_knowledge_rc_5shot",
                        "basic_skills_logical_reasoning_rc_5shot",
                        "basic_skills_pattern_rc_5shot",
                        "basic_skills_string_operations_rc_5shot",
                        "codex_humaneval_gold_bpb_3shot",
                        "codex_mbpp_gold_bpb_3shot",
                        "minerva_math_500_gold_bpb_0shot",
                        "mt_mbpp_cpp_gold_bpb_3shot",
                        "mt_mbpp_java_gold_bpb_3shot",
                        "mt_mbpp_rust_gold_bpb_3shot",
                        "copycolors_10way_fast",
                    ]
                ),
                tokenizer=tokenizer_config,
                eval_interval=EVAL_DOWN_STEPS,
            ),
        )
    )

    return ExperimentConfig(
        model=model_builder(tokenizer_config),
        dataset=dataset_config,
        data_loader=data_loader_config,
        train_module=train_module_config,
        trainer=trainer_config,
    ).merge(overrides)
