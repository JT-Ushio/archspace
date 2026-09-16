"""Read-only A/C W&B snapshot; use after terminal-job offline sync finishes.

Reuses the existing sanitized exporter and literal credential-file parser.
An explicit fresh output directory is required; no existing snapshot is replaced.
"""
import sys
from pathlib import Path


if __name__ == "__main__":
    if not any(arg == "--output-dir" or arg.startswith("--output-dir=") for arg in sys.argv[1:]):
        raise SystemExit("Specify --output-dir with a fresh snapshot directory")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Down_Identity_Ablations"))
    import export_runs
    export_runs.RUNS = {
        "A": "olmo3-1b-init-independent-down-rms-fp8-30b",
        "C": "olmo3-1b-init-svd-scale-matched-down-rms-fp8-30b",
    }
    export_runs.main()
