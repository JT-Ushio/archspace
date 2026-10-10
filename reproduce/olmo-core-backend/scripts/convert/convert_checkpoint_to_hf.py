#!/usr/bin/env python3
"""Use this entry point for every supported MoSE and Dense checkpoint."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from olmo_mose.hf_export.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
