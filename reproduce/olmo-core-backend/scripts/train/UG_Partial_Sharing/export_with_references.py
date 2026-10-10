"""Recompute completed candidates and archived C0/C2/C6 with one anomaly mask.

Current jobs must pass the existing live-registry completion gates. Historical
references use the recorded completion ledger and complete W&B histories;
their scheduler state is not claimed to have been revalidated today.
"""

import argparse
import hashlib
import json
from pathlib import Path

from compare_offline import compare
from export_completed_batch import export

HERE = Path(__file__).resolve().parent


def export_references(current_manifest, reference_manifest):
    current = export(current_manifest, required_labels=None)
    references = json.loads(reference_manifest.read_text())
    if set(references) != {"c0-original", "c2-original", "c6-original"}:
        raise ValueError("Provide exactly the archived C0/C2/C6 references")
    paths = {label: row["paths"] for label, row in current["runs"].items()}
    provenance = current["provenance"].copy()
    for label, row in references.items():
        if label in paths:
            raise ValueError("Reference labels must not replace current candidates")
        ledger = Path(row["completion_ledger"]).read_text()
        if row["job_id"] not in ledger or row["segment"] not in ledger:
            raise ValueError(f"{label}: reference does not match the archived ledger")
        if row["full_parameter_count"] != 797050880:
            raise ValueError(f"{label}: not the rank512 reference parameter budget")
        files = list(Path(row["segment"]).glob("run-*.wandb"))
        if len(files) != 1:
            raise ValueError(f"{label}: expected one exact archived W&B history")
        paths[label] = [str(files[0])]
        provenance[label] = {**row, "strictly_parameter_matched": True,
                             "completion_basis": "Archived completed-run ledger and full history",
                             "current_scheduler_state_revalidated": False}
    result = compare(paths, 14306)
    if not result["full_30b_histories_complete"]:
        raise ValueError("Cannot compare incomplete current or archived histories")
    losses = {label: row["common_clean_last1000_mean"] for label, row in result["runs"].items()}
    if any(value is None for value in losses.values()):
        raise ValueError("The matched final-window metric is unavailable")
    result["provenance"] = provenance
    result["primary_metric"] = "common_clean_last1000_mean"
    result["strict_budget_ranking"] = sorted(
        (label for label in losses if provenance[label]["strictly_parameter_matched"]), key=losses.get)
    result["delta_vs_original_c6"] = {label: loss - losses["c6-original"] for label, loss in losses.items()}
    result["qk_effects"] = {label: losses[label + "qk"] - losses[label]
                             for label in ("p0", "p2", "p3", "p4", "f1", "f2", "f3", "f4", "f5")
                             if label in losses and label + "qk" in losses}
    result["scope"] = "Matched quality screening; archived references are not fresh scheduler observations or seed statistics"
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--references", type=Path, default=HERE / "historical_reference_segments.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = export_references(args.manifest, args.references)
    encoded = json.dumps(result, ensure_ascii=False, allow_nan=False)
    args.output.write_text(encoded)
    print("MOSE_REFERENCE_COMPARISON_SHA256 " + hashlib.sha256(encoded.encode()).hexdigest(), flush=True)
    print("MOSE_REFERENCE_COMPARISON_JSON " + encoded, flush=True)
