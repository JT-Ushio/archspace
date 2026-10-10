#!/usr/bin/env python3
"""Audit every current CFP on meta. Does not load checkpoints or run forward."""

import argparse
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

BACKEND = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND / "src"))

from olmo_mose.hf_export.audit import make_hf_config, structural_audit  # noqa: E402
from olmo_mose.hf_export.checkpoint import build_native_meta, write_json  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=BACKEND / "cfps")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    original_path = list(sys.path)
    results = []
    for index, path in enumerate(sorted(root.rglob("*.py"))):
        if path.name.startswith("_"):
            continue
        # CFP directories reuse the name _common. Drop only CFP-local imports.
        for name, module in list(sys.modules.items()):
            location = getattr(module, "__file__", None)
            if location and str(location).startswith(str(root) + "/"):
                del sys.modules[name]
        sys.path[:] = [str(path.parent), *original_path]
        try:
            spec = importlib.util.spec_from_file_location(f"_hf_audit_{index}", path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            if not hasattr(module, "build_config"):
                results.append(
                    {
                        "config": str(path.relative_to(root)),
                        "status": "skipped",
                        "reason": "No build_config",
                    }
                )
                continue
            opts = SimpleNamespace(
                name="hf-structural-audit",
                data_root="/unused/data",
                work_dir="/tmp/mose-hf-meta-audit",
                save_folder="/unused/ckpt",
                sequence_length=None,
            )
            experiment = module.build_config(opts, [])
            raw = experiment.as_config_dict()
            native = build_native_meta(raw)
            result = structural_audit(native, make_hf_config(native, raw))
            results.append({"config": str(path.relative_to(root)), "status": "passed", **result})
        except Exception as exc:
            results.append(
                {
                    "config": str(path.relative_to(root)),
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        finally:
            sys.path[:] = original_path
        write_json(args.report, {"numerical_validation": "not_run", "configs": results})
    counts = {
        status: sum(r["status"] == status for r in results)
        for status in ("passed", "failed", "skipped")
    }
    write_json(
        args.report, {"summary": counts, "numerical_validation": "not_run", "configs": results}
    )
    print(counts)
    if counts["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
