"""Archived Dense evaluation requires the exact resolved source and completion."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def evidence_module():
    path = Path(__file__).resolve().parents[1] / "scripts/analysis/Corrected_LM_Evaluation/case_evidence.py"
    spec = importlib.util.spec_from_file_location("lm_checkpoint_evidence_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolved_case(tmp_path):
    resolved = {"checkpoint": "/ckpts/dense/stage1/step14306", "source_run_id": "dense",
                "source_segments": ["/wandb/old", "/wandb/resumed"],
                "source_job_id": None, "num_params": 1484916736,
                "config_path": "dense.py", "checkpoint_step": 14306, "training_steps": 14306}
    payload = json.dumps(resolved)
    proof = tmp_path / "resolved.json"
    proof.write_text(payload + "\n")
    ledger = tmp_path / "jobs.json"
    ledger.write_text(json.dumps([{"variant": "original-dense-path-resolve", "completion_verified": {
        "state": "Succeeded", "counters": {"active": 0, "succeeded": 1},
        "sha256": hashlib.sha256(payload.encode()).hexdigest()}}]))
    return {**resolved, "completion_proof": str(proof), "proof_kind": "resolved_wandb_metadata",
            "resolver_completion_ledger": str(ledger)}


def test_wandb_resolution_accepts_unknown_original_scheduler_id(tmp_path):
    entry = resolved_case(tmp_path)
    assert evidence_module().validate_case(entry) == entry["checkpoint"]


@pytest.mark.parametrize("changed", ["digest", "active_resolver", "wrong_run", "missing_training_step"])
def test_wandb_resolution_rejects_unproven_or_different_source(tmp_path, changed):
    entry = resolved_case(tmp_path)
    if changed == "digest":
        Path(entry["completion_proof"]).write_text(Path(entry["completion_proof"]).read_text() + " ")
    elif changed == "active_resolver":
        path = Path(entry["resolver_completion_ledger"])
        data = json.loads(path.read_text())
        data[0]["completion_verified"]["counters"]["active"] = 1
        path.write_text(json.dumps(data))
    elif changed == "wrong_run":
        entry["source_run_id"] = "different"
    else:
        path = Path(entry["completion_proof"])
        data = json.loads(path.read_text())
        data["training_steps"] = 14305
        payload = json.dumps(data)
        path.write_text(payload + "\n")
        ledger = Path(entry["resolver_completion_ledger"])
        jobs = json.loads(ledger.read_text())
        jobs[0]["completion_verified"]["sha256"] = hashlib.sha256(payload.encode()).hexdigest()
        ledger.write_text(json.dumps(jobs))
    with pytest.raises(ValueError):
        evidence_module().validate_case(entry)
