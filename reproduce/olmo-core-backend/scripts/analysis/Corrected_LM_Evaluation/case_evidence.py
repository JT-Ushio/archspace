"""Validate raw completion logs or checksum-verified archived checkpoint metadata."""

import hashlib
import json
from pathlib import Path


def validate_case(entry):
    checkpoint = entry["checkpoint"]
    if not checkpoint.endswith("/step14306"):
        raise ValueError("Require an exact final14306 checkpoint")
    proof = Path(entry["completion_proof"]).read_text()
    kind = entry.get("proof_kind", "raw_completion_log")
    if kind == "raw_completion_log":
        if (entry["source_job_id"] not in proof or "Training complete" not in proof
                or checkpoint not in proof):
            raise ValueError("Checkpoint differs from verified completion evidence")
    elif kind == "resolved_reference_metadata":
        payload = proof.removesuffix("\n")
        digest = hashlib.sha256(payload.encode()).hexdigest()
        jobs = json.loads(Path(entry["resolver_completion_ledger"]).read_text())
        matches = [job for job in jobs if job["variant"] == "reference-path-resolve"]
        if len(matches) != 1:
            raise ValueError("Require one completed reference resolution")
        completion = matches[0].get("completion_verified", {})
        if (completion.get("state") != "Succeeded"
                or completion.get("counters", {}).get("active") != 0
                or completion.get("counters", {}).get("succeeded") != 1
                or completion.get("sha256") != digest):
            raise ValueError("Reference metadata lacks verified terminal state and checksum")
        resolved = json.loads(payload)[entry["reference_key"]]
        if (resolved["checkpoint"] != checkpoint
                or resolved["source"]["job_id"] != entry["source_job_id"]
                or resolved["source"]["full_parameter_count"] != entry["num_params"]
                or resolved["checkpoint_step"] != 14306
                or resolved["training_steps"] != 14306):
            raise ValueError("Resolved checkpoint differs from the selected archived source")
    elif kind == "resolved_wandb_metadata":
        payload = proof.removesuffix("\n")
        digest = hashlib.sha256(payload.encode()).hexdigest()
        jobs = json.loads(Path(entry["resolver_completion_ledger"]).read_text())
        matches = [job for job in jobs if job["variant"] == "original-dense-path-resolve"]
        if len(matches) != 1:
            raise ValueError("Require one completed original Dense resolution")
        completion = matches[0].get("completion_verified", {})
        if (completion.get("state") != "Succeeded"
                or completion.get("counters", {}).get("active") != 0
                or completion.get("counters", {}).get("succeeded") != 1
                or completion.get("sha256") != digest):
            raise ValueError("Dense metadata lacks verified terminal state and checksum")
        resolved = json.loads(payload)
        if (resolved["checkpoint"] != checkpoint
                or resolved["source_run_id"] != entry["source_run_id"]
                or resolved["source_segments"] != entry["source_segments"]
                or resolved["num_params"] != entry["num_params"]
                or resolved["config_path"] != entry["config_path"]
                or resolved["checkpoint_step"] != 14306
                or resolved["training_steps"] != 14306
                or resolved["source_job_id"] is not None
                or entry["source_job_id"] is not None):
            raise ValueError("Resolved Dense checkpoint differs from the original W&B source")
    else:
        raise ValueError("Unknown checkpoint evidence type")
    return checkpoint
