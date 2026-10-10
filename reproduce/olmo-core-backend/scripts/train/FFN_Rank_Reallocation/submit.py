"""Reuse the locally validated submission handling for the three FFN jobs."""

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHARING_DIR = HERE.parent / "UG_Partial_Sharing"
sys.path.insert(0, str(SHARING_DIR))
spec = importlib.util.spec_from_file_location("ug_submission_helper", SHARING_DIR / "submit.py")
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
helper.HERE = HERE

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("f1", "f2", "f3"):
        raise SystemExit("usage: submit.py f1|f2|f3")
    helper.submit(sys.argv[1])
