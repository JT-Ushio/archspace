"""Completion metadata cannot substitute for complete, aligned CE histories."""

import importlib.util
import json
from pathlib import Path

import pytest


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def cohort(tmp_path, monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts/train/UG_Partial_Sharing"
    monkeypatch.syspath_prepend(str(scripts))
    exporter = load(scripts / "export_completed_batch.py", "cohort_export_test")
    import compare_offline

    root = tmp_path / "UG_Partial_Sharing"
    root.mkdir()
    seeds = tmp_path / "Rank512_Seed_Replication"
    seeds.mkdir()
    monkeypatch.setattr(exporter, "HERE", root)
    labels = ("c0s2", "c6s2", "p1")
    rows, manifest, histories = {}, {}, {}
    for label, loss in zip(labels, (3.0, 2.99, 2.98)):
        segment = tmp_path / ("offline-" + label)
        segment.mkdir()
        path = segment / ("run-" + label + ".wandb")
        path.touch()
        rows[label] = {
            "variant": label, "job_id": "job-" + label, "wandb_segment": str(segment),
            "init_seed": 12537 if label.startswith("c") else 12536, "data_seed": 34521,
            "partition": "pool", "nodes": 1, "gpus_per_node": 8, "priority": 1,
            "completion_verified": {"state": "Succeeded", "counters": {"active": 0},
                                    "final_step": 14306, "final_checkpoint_saved": True},
        }
        manifest[label] = {"job_id": "job-" + label, "segments": [str(segment)]}
        histories[str(path)] = {"train": dict.fromkeys(range(1, 14307), loss),
                                "validation": {}, "duplicate_steps": []}
    histories[str(tmp_path / "offline-c0s2/run-c0s2.wandb")]["train"][13900] = 0.0
    (root / "jobs.json").write_text(json.dumps([rows["p1"]]))
    (seeds / "jobs.json").write_text(json.dumps([rows["c0s2"], rows["c6s2"]]))
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(compare_offline, "read", lambda paths: histories[paths[0]])
    return exporter, path, rows, manifest, histories


def test_cohort_preserves_anomalies_and_checks_matched_seed_pairs(cohort):
    exporter, path, rows, manifest, histories = cohort
    result = exporter.export(path, required_labels=None)
    assert result["full_30b_histories_complete"] is True
    assert result["common_excluded_steps"] == [13900]
    assert result["runs"]["c0s2"]["raw_last1000_mean"] == pytest.approx(2.997)
    assert result["runs"]["c0s2"]["common_clean_last1000_mean"] == 3.0
    assert result["runs"]["c6s2"]["clean_tail_n"] == 999
    assert result["strict_budget_ranking"] == ["c6s2", "c0s2"]
    assert result["all_config_ranking"] == ["p1", "c6s2", "c0s2"]
    assert result["seed_confirmation"]["n_pairs"] == 1
    assert result["seed_confirmation"]["mean_c6_minus_c0"] == pytest.approx(-0.01)
    assert result["seed_confirmation"]["stdev_c6_minus_c0"] is None
    assert result["qk_interaction"] is None
    with pytest.raises(ValueError, match="frozen batch"):
        exporter.export(path)


def test_cohort_rejects_missing_ce_even_with_successful_job_metadata(cohort):
    exporter, path, rows, manifest, histories = cohort
    history = next(value for key, value in histories.items() if "run-c6s2" in key)
    del history["train"][14306]
    with pytest.raises(ValueError, match="Cannot rank incomplete histories"):
        exporter.export(path, required_labels=None)


def test_shared_ffn_qk_controls_use_aligned_clean_final_windows(cohort):
    exporter, path, rows, manifest, histories = cohort
    ff_root = path.parent.parent / "FFN_Rank_Reallocation"
    ff_root.mkdir()
    extra_rows = []
    for label, loss in (("f4", 3.04), ("f4qk", 3.03), ("f5", 3.02), ("f5qk", 3.0)):
        segment = path.parent.parent / ("offline-" + label)
        segment.mkdir()
        history_path = segment / ("run-" + label + ".wandb")
        history_path.touch()
        row = {**rows["p1"], "variant": label, "job_id": "job-" + label,
               "wandb_segment": str(segment), "num_params": 797050880}
        extra_rows.append(row)
        manifest[label] = {"job_id": row["job_id"], "segments": [str(segment)]}
        history = dict.fromkeys(range(1, 14307), loss)
        if label == "f4qk":
            history[13910] = 0.0
        histories[str(history_path)] = {"train": history, "validation": {}, "duplicate_steps": []}
    (ff_root / "jobs.json").write_text(json.dumps(extra_rows))
    path.write_text(json.dumps(manifest))
    result = exporter.export(path, required_labels=None)
    assert result["common_excluded_steps"] == [13900, 13910]
    assert result["runs"]["f4qk"]["clean_tail_n"] == 998
    assert result["qk_effects"]["f4"] == pytest.approx(-0.01)
    assert result["qk_effects"]["f5"] == pytest.approx(-0.02)


@pytest.mark.parametrize("mismatch", [None, "seed", "allocation", "full_parameter_count"])
def test_c2_control_separates_shared_rank_recipe_and_q_activation_effects(cohort, mismatch):
    exporter, path, rows, manifest, histories = cohort
    segment = path.parent.parent / "offline-c2s2"
    segment.mkdir()
    history_path = segment / "run-c2s2.wandb"
    history_path.touch()
    row = {**rows["c0s2"], "variant": "c2s2", "job_id": "job-c2s2", "wandb_segment": str(segment)}
    if mismatch == "seed":
        row["init_seed"] = 12538
    elif mismatch == "allocation":
        row["nodes"] = 2
    elif mismatch == "full_parameter_count":
        row["num_params"] = 794429440
    rows["c2s2"] = row
    manifest["c2s2"] = {"job_id": row["job_id"], "segments": [str(segment)]}
    path.write_text(json.dumps(manifest))
    (exporter.HERE.parent / "Rank512_Seed_Replication/jobs.json").write_text(
        json.dumps([rows[label] for label in ("c0s2", "c2s2", "c6s2")]))
    histories[str(history_path)] = {"train": dict.fromkeys(range(1, 14307), 2.992),
                                   "validation": {}, "duplicate_steps": []}
    if mismatch:
        with pytest.raises(ValueError, match="paired replications must match"):
            exporter.export(path, required_labels=None)
        return
    result = exporter.export(path, required_labels=None)
    contrasts = result["seed_confirmation"]["contrasts"]
    assert contrasts["c2_minus_c0"]["mean_difference"] == pytest.approx(-0.008)
    assert contrasts["c6_minus_c2"]["mean_difference"] == pytest.approx(-0.002)
    assert contrasts["c6_minus_c0"]["mean_difference"] == pytest.approx(-0.01)
    assert contrasts["c6_minus_c2"]["n_pairs"] == 1
    assert result["common_excluded_steps"] == [13900]
    assert result["runs"]["c2s2"]["clean_tail_n"] == 999


@pytest.mark.parametrize("mismatch", ["job_id", "final_checkpoint", "seed", "allocation"])
def test_cohort_rejects_stale_unfinished_or_unmatched_pairs(cohort, mismatch):
    exporter, path, rows, manifest, histories = cohort
    if mismatch == "job_id":
        manifest["c6s2"]["job_id"] = "retired-attempt"
        path.write_text(json.dumps(manifest))
    else:
        if mismatch == "final_checkpoint":
            rows["c6s2"]["completion_verified"]["final_checkpoint_saved"] = False
        elif mismatch == "seed":
            rows["c6s2"]["init_seed"] = 12538
        else:
            rows["c6s2"]["nodes"] = 2
        (exporter.HERE.parent / "Rank512_Seed_Replication/jobs.json").write_text(
            json.dumps([rows["c0s2"], rows["c6s2"]]))
    with pytest.raises(ValueError):
        exporter.export(path, required_labels=None)


@pytest.mark.parametrize("failure", [None, "missing_history", "wrong_ledger"])
def test_archived_references_share_the_current_anomaly_mask_without_faking_fresh_status(cohort, monkeypatch, failure):
    exporter, path, rows, manifest, histories = cohort
    script = Path(__file__).resolve().parents[1] / "scripts/train/UG_Partial_Sharing/export_with_references.py"
    reference_exporter = load(script, "reference_export_test")
    monkeypatch.setattr(reference_exporter, "export", lambda p, required_labels: exporter.export(p, required_labels))
    references = {}
    ledger = path.parent / "archived.md"
    evidence = []
    for label, loss in zip(("c0-original", "c2-original", "c6-original"), (3.01, 3.005, 3.0)):
        segment = path.parent / label
        segment.mkdir()
        history_path = segment / ("run-" + label + ".wandb")
        history_path.touch()
        references[label] = {"job_id": "job-" + label, "segment": str(segment),
                             "completion_ledger": str(ledger), "full_parameter_count": 797050880}
        evidence.extend([references[label]["job_id"], str(segment)])
        histories[str(history_path)] = {"train": dict.fromkeys(range(1, 14307), loss),
                                       "validation": {}, "duplicate_steps": []}
    historical = histories[str(path.parent / "c6-original/run-c6-original.wandb")]["train"]
    historical[12516] = 0.0
    if failure == "missing_history":
        del historical[14306]
    elif failure == "wrong_ledger":
        references["c6-original"]["job_id"] = "unrecorded-job"
    ledger.write_text("\n".join(evidence))
    reference_path = path.parent / "references.json"
    reference_path.write_text(json.dumps(references))
    if failure:
        with pytest.raises(ValueError):
            reference_exporter.export_references(path, reference_path)
        return
    result = reference_exporter.export_references(path, reference_path)
    assert result["common_excluded_steps"] == [12516, 13900]
    assert result["delta_vs_original_c6"]["c6s2"] == pytest.approx(-0.01)
    assert result["provenance"]["c6-original"]["current_scheduler_state_revalidated"] is False
    assert "seed_confirmation" not in result
