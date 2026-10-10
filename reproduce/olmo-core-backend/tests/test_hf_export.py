"""Opt-in HF regression suite. Written now; numerical tests have not been run.

Run explicitly with the [hf,dev] extras installed. Fixtures use tiny models,
never production checkpoints or the scheduler.
"""

import json
import subprocess
import sys

import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("safetensors")

from olmo_core.nn.attention import AttentionBackendName, SlidingWindowAttentionConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_mose import (
    LowRankAttentionSharingScope,
    MoSENonlinearity,
    SwiGLUChannelControl,
    SwiGLUChannelControlScope,
    patch_low_rank_attention,
    patch_mose_swiglu,
    patch_swiglu_channel_control,
)
from olmo_mose.adaptive_qk_attention import AdaptiveQKAttentionConfig
from olmo_mose.dense_shared_attention import DenseSharedKVConfig
from olmo_mose.dense_single_attention import DenseSingleSiLUAttentionConfig
from olmo_mose.dense_single_ffn import DenseSingleSiLUFFNConfig
from olmo_mose.dense_ffn_factorization import DenseFFNFactorizationConfig
from olmo_mose.residual_feed_forward import ResidualMoSESwiGLUConfig
from olmo_mose.hf_export.audit import make_hf_config, structural_audit
from olmo_mose.hf_export.checkpoint import CheckpointReader, write_json
from olmo_mose.hf_export.exporter import export_batch, export_checkpoint
from olmo_mose.hf_export.modeling_mose import MoSEForCausalLM
from olmo_mose.hf_export.registry import describe, native_to_hf_key
from olmo_mose.hf_export.validation import compare_models, validate_export

FAMILIES = [
    "dense",
    "lowrank",
    "qk",
    "kv",
    "qkv",
    "partial",
    "adaptive",
    "dense_q",
    "dense_k",
    "dense_up",
    "dense_gate",
    "dense_up_linear_down_rms",
    "dense_gate_linear_down_rms",
    "dense_up_gate_linear_down_rms",
    "dense_kv",
    "static",
    "dynamic",
    "dynamic_residual",
    "dense_down",
    "mixed_bias",
    "residual_l1",
    "residual_r0",
    "residual_r1",
    "residual_r2",
    "residual_r3",
]


def config_for(family):
    cfg = TransformerConfig.olmo3_1M(
        vocab_size=128,
        attn_backend=AttentionBackendName.torch,
        sliding_window=SlidingWindowAttentionConfig(
            pattern=[3, -1],
            force_full_attention_on_first_layer=False,
            force_full_attention_on_last_layer=False,
        ),
    )
    if family == "dense":
        return cfg
    if family.startswith("residual_"):
        target = family.removeprefix("residual_")
        cfg = patch_low_rank_attention(
            cfg, enabled=True, rank=8, share_scope=LowRankAttentionSharingScope.none,
            q_nonlinearity=MoSENonlinearity.identity,
            k_nonlinearity=MoSENonlinearity.identity,
            v_nonlinearity=MoSENonlinearity.identity,
            o_nonlinearity=MoSENonlinearity.identity,
        )
        cfg.block.feed_forward = ResidualMoSESwiGLUConfig(
            hidden_size=cfg.d_model * (4 if target == "l1" else 1),
            r2=8, down_r2=8,
            wide_residual=target in ("r1", "r3"),
            latent_residual=target in ("l1", "r2", "r3"),
        )
        return cfg
    if family.endswith("_linear_down_rms"):
        cfg.block.feed_forward = DenseFFNFactorizationConfig(
            **cfg.block.feed_forward.as_dict(exclude_none=True, recurse=False),
            target=family.removeprefix("dense_").removesuffix("_linear_down_rms"),
            input_rank=8, down_rank=8,
        )
        return cfg
    if family in ("dense_q", "dense_k", "dense_kv"):
        cls = DenseSharedKVConfig if family == "dense_kv" else DenseSingleSiLUAttentionConfig
        kwargs = {"rank": 8} if family == "dense_kv" else {"target": family[-1]}
        cfg.block.sequence_mixer = cls(
            **cfg.block.sequence_mixer.as_dict(exclude_none=True, recurse=False), **kwargs
        )
        return cfg
    if family in ("dense_up", "dense_gate"):
        cfg.block.feed_forward = DenseSingleSiLUFFNConfig(
            **cfg.block.feed_forward.as_dict(exclude_none=True, recurse=False),
            target=family[6:],
            rank=8,
        )
        return cfg
    cfg = patch_low_rank_attention(
        cfg,
        enabled=True,
        rank=8,
        share_scope=LowRankAttentionSharingScope(family)
        if family in ("qk", "kv", "qkv")
        else LowRankAttentionSharingScope.none,
        q_nonlinearity=MoSENonlinearity.identity,
        k_nonlinearity=MoSENonlinearity.silu,
        v_nonlinearity=MoSENonlinearity.rms_norm,
        o_nonlinearity=MoSENonlinearity.silu_rms_norm,
    )
    cfg = patch_mose_swiglu(
        cfg,
        control=SwiGLUChannelControl.standard,
        control_scope=SwiGLUChannelControlScope.none,
        r1=0,
        r2=8,
        down_r1=0,
        down_r2=8,
        gate_nonlinearity=MoSENonlinearity.identity,
        up_nonlinearity=MoSENonlinearity.identity,
        down_nonlinearity=MoSENonlinearity.rms_norm,
    )
    if family == "partial":
        cfg.block.feed_forward.r1 = 4
        cfg.block.feed_forward.share_linear_gate_up_subspace = True
        cfg.block.feed_forward.share_nonlinear_gate_up_subspace = False
    elif family == "adaptive":
        cfg.block.sequence_mixer = AdaptiveQKAttentionConfig(
            **cfg.block.sequence_mixer.as_dict(exclude_none=True, recurse=False)
        )
    elif family in ("static", "dynamic", "dynamic_residual"):
        cfg.block.feed_forward.down_nonlinearity = MoSENonlinearity.identity
        cfg.block.feed_forward.down_output_rotation = family
        cfg.block.feed_forward.down_rotation_groups = 2
    elif family == "dense_down":
        cfg.block.feed_forward.down_r2 = 0
    elif family == "mixed_bias":
        cfg.block.feed_forward.bias = True
        cfg.block.feed_forward.r1 = cfg.block.feed_forward.down_r1 = 4
        cfg.block.feed_forward.share_gate_up_subspace = False
        cfg.block.feed_forward.rms_norm_learnable_weight = True
    return cfg


def native_for(family):
    model = config_for(family).build(init_device="cpu")
    model.init_weights(device=torch.device("cpu"))
    # Exercise learned controllers, rather than only identity initializations.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith(".alpha"):
                parameter.fill_(0.37)
            elif name.endswith(".residual_gate"):
                parameter.fill_(0.23)
            elif ".output_subspace.weight" in name:
                parameter.fill_(0.1)
    return model.eval()


@pytest.mark.parametrize("family", FAMILIES)
def test_forward_parameters_sharing_and_cache(family):
    native = native_for(family)
    config = make_hf_config(native, {})
    structural_audit(native, config)
    hf = MoSEForCausalLM(config).eval()
    hf.load_state_dict(
        {native_to_hf_key(k): v for k, v in native.state_dict().items()}, strict=True
    )
    report = compare_models(native, hf, torch.tensor([[3, 11, 4, 7, 13, 8]]))
    assert report["passed"], report


@pytest.mark.parametrize("control", list(SwiGLUChannelControl))
@pytest.mark.parametrize("scope", list(SwiGLUChannelControlScope))
def test_dense_channel_controls(control, scope):
    cfg = patch_swiglu_channel_control(config_for("dense"), control=control, control_scope=scope)
    native = cfg.build(init_device="cpu")
    native.init_weights(device=torch.device("cpu"))
    hf = MoSEForCausalLM(make_hf_config(native, {})).eval()
    hf.load_state_dict({native_to_hf_key(k): v for k, v in native.state_dict().items()})
    tokens = torch.tensor([[3, 7, 9, 11]])
    with torch.inference_mode():
        torch.testing.assert_close(
            hf(tokens, use_cache=False).logits, native.eval()(tokens), atol=1e-5, rtol=1e-4
        )


def saved_checkpoint(tmp_path, family="kv", dcp=True):
    native = native_for(family)
    step = tmp_path / "stage1" / "step7"
    step.mkdir(parents=True)
    write_json(step / "config.json", {"model": config_for(family).as_config_dict()})
    if dcp:
        from olmo_core.distributed.checkpoint import save_model_and_optim_state

        save_model_and_optim_state(step / "model_and_optim", native)
    else:
        torch.save(native.state_dict(), step / "model.pt")
    return step


@pytest.mark.parametrize("dcp", [True, False])
@pytest.mark.parametrize("family", ["kv", "adaptive", "dynamic_residual", "partial",
                                   "residual_l1", "residual_r0", "residual_r1",
                                   "residual_r2", "residual_r3"])
def test_sharded_export_auto_reload_and_validate(tmp_path, dcp, family):
    step = saved_checkpoint(tmp_path, family, dcp)
    output = tmp_path / "hf"
    manifest = export_checkpoint(step, output, skip_tokenizer=True, max_shard_size="10KB")
    assert manifest["numerical_validation"]["status"] == "not_run"
    assert len(manifest["weight_files"]) > 1
    assert (output / "runtime_ops.py").is_file()
    assert not (output / "registry.py").exists()
    assert validate_export(step, output)["passed"]
    # Copied runtime must load without the project's PYTHONPATH / olmo_mose imports.
    script = "from transformers import AutoModelForCausalLM; m=AutoModelForCausalLM.from_pretrained(__import__('sys').argv[1], trust_remote_code=True, local_files_only=True); assert m.config.model_type=='olmo_mose'"
    import os

    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    subprocess.run([sys.executable, "-c", script, str(output)], cwd=tmp_path, env=env, check=True)


@pytest.mark.parametrize("family", ["kv", "adaptive", "dynamic_residual", "partial"])
def test_bf16_export_preserves_fp32_controllers(tmp_path, family):
    step = saved_checkpoint(tmp_path, family, dcp=False)
    output = tmp_path / "hf-bf16"
    export_checkpoint(step, output, dtype="bfloat16", skip_tokenizer=True)
    report = validate_export(step, output, atol=0.03, rtol=0.02)
    assert report["passed"], report


def test_svd_config_does_not_need_initialization_artifacts():
    from olmo_mose.svd_initialization import SVDInitTransformerConfig
    from olmo_mose.hf_export.checkpoint import build_native_meta
    from dataclasses import fields

    cfg = config_for("lowrank")
    svd = SVDInitTransformerConfig(
        **{field.name: getattr(cfg, field.name) for field in fields(TransformerConfig)},
        factor_init="svd_scale_matched",
        factor_init_artifacts="/nonexistent/init-artifacts",
    )
    native = build_native_meta({"model": svd.as_config_dict()})
    assert structural_audit(native, make_hf_config(native, {}))["structural_check"] == "passed"


def test_forward_rejects_noninterchangeable_arguments():
    native = native_for("dense")
    hf = MoSEForCausalLM(make_hf_config(native, {}))
    ids = torch.tensor([[1, 2, 3]])
    with pytest.raises(TypeError, match="Unsupported HF forward"):
        hf(ids, cu_doc_lens=torch.tensor([0, 3]))
    with pytest.raises(ValueError, match="exactly one"):
        hf(ids, inputs_embeds=native.embeddings(ids))
    with pytest.raises(ValueError, match="full logits"):
        hf(ids, labels=ids, logits_to_keep=1)
    with pytest.raises(ValueError, match="contiguous"):
        hf(ids, cache_position=torch.tensor([3, 4, 5]))


def test_batch_and_incomplete_checkpoint_fail_explicitly(tmp_path):
    step = saved_checkpoint(tmp_path, dcp=False)
    latest = step.parent / "step8"
    latest.mkdir()
    write_json(latest / "config.json", json.loads((step / "config.json").read_text()))
    with pytest.raises(FileNotFoundError):
        CheckpointReader(step.parent)
    manifest = tmp_path / "batch.json"
    write_json(
        manifest,
        [
            {"checkpoint": str(step), "output": "hf-good"},
            {"checkpoint": str(latest), "output": "hf-incomplete"},
        ],
    )
    results = export_batch(manifest, skip_tokenizer=True)
    assert [r["status"] for r in results] == ["exported", "failed"]
    assert not (tmp_path / "hf-incomplete").exists()
    with pytest.raises(FileExistsError):
        export_checkpoint(step, tmp_path / "hf-good", skip_tokenizer=True)


def test_unknown_architecture_has_no_dense_fallback():
    class NewArchitecture(torch.nn.Linear):
        pass

    with pytest.raises(ValueError, match="No HF architecture adapter"):
        describe(NewArchitecture(4, 4))
