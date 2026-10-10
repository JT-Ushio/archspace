"""Unified CLI: inspect/audit-config, convert, batch, and opt-in validate."""

import argparse
import json
from pathlib import Path

from .checkpoint import write_json


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    convert = commands.add_parser("convert", help="Export one native checkpoint")
    convert.add_argument("checkpoint")
    convert.add_argument("output")
    convert.add_argument("--config", help="Override saved config.json location")
    batch = commands.add_parser("batch", help="Export entries in a JSON manifest")
    batch.add_argument("manifest")
    for command in (convert, batch):
        command.add_argument(
            "--dtype", choices=("auto", "float32", "bfloat16", "float16"), default="auto"
        )
        command.add_argument("--max-shard-size", default="2GB")
        command.add_argument(
            "--tokenizer", help="Override saved tokenizer identifier or local directory"
        )
        command.add_argument(
            "--skip-tokenizer", action="store_true", help="Export weights/runtime only"
        )
        command.add_argument(
            "--local-files-only", action="store_true", help="Do not download tokenizer files"
        )
    inspect = commands.add_parser(
        "inspect", help="Audit config/checkpoint shapes without reading weights"
    )
    inspect.add_argument("checkpoint")
    inspect.add_argument("--config")
    inspect.add_argument("--report")
    audit = commands.add_parser(
        "audit-config", help="Meta-device audit of a saved model config; no checkpoint required"
    )
    audit.add_argument("config")
    audit.add_argument("--report")
    validate = commands.add_parser("validate", help="Explicitly run native/HF numerical checks")
    validate.add_argument("checkpoint")
    validate.add_argument("output")
    validate.add_argument("--config")
    validate.add_argument("--device", default="cpu")
    validate.add_argument(
        "--dtype", choices=("auto", "float32", "bfloat16", "float16"), default="auto"
    )
    validate.add_argument("--input-ids", help="JSON token IDs [T] or [B,T]")
    validate.add_argument("--sequence-length", type=int, default=16)
    validate.add_argument("--seed", type=int, default=12536)
    validate.add_argument("--atol", type=float, default=1e-5)
    validate.add_argument("--rtol", type=float, default=1e-4)
    validate.add_argument("--no-intermediates", action="store_true")
    validate.add_argument("--report")
    commands.add_parser("architectures", help="List registered native classes")
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command in ("convert", "batch"):
        from .exporter import export_batch, export_checkpoint

        options = dict(
            dtype=args.dtype,
            max_shard_size=args.max_shard_size,
            tokenizer=args.tokenizer,
            skip_tokenizer=args.skip_tokenizer,
            local_files_only=args.local_files_only,
        )
        if args.command == "convert":
            result = export_checkpoint(
                args.checkpoint, args.output, config_path=args.config, **options
            )
            print(
                json.dumps(
                    {
                        "output": result["output"],
                        "parameters": result["audit"]["unique_parameters"],
                        "numerical_validation": "not_run",
                    },
                    ensure_ascii=False,
                )
            )
        else:
            result = export_batch(args.manifest, **options)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            if any(item["status"] == "failed" for item in result):
                raise SystemExit(1)
    elif args.command in ("inspect", "audit-config"):
        from .audit import make_hf_config, structural_audit
        from .checkpoint import CheckpointReader, build_native_meta

        if args.command == "inspect":
            reader = CheckpointReader(args.checkpoint, args.config)
            native = build_native_meta(reader.raw)
            reader.check_native(native)
            raw, provenance = reader.raw, reader.provenance()
        else:
            raw = json.loads(Path(args.config).read_text())
            native, provenance = build_native_meta(raw), {"config": args.config}
        result = {**provenance, **structural_audit(native, make_hf_config(native, raw))}
        if args.report:
            write_json(args.report, result)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif args.command == "validate":
        from .validation import validate_export

        result = validate_export(
            args.checkpoint,
            args.output,
            config_path=args.config,
            device=args.device,
            dtype=args.dtype,
            input_ids_path=args.input_ids,
            sequence_length=args.sequence_length,
            seed=args.seed,
            atol=args.atol,
            rtol=args.rtol,
            intermediates=not args.no_intermediates,
            report_path=args.report,
        )
        print(json.dumps({"passed": result["passed"], "logits": result["logits"]}))
    else:
        from .registry import registered_architectures

        print(json.dumps(registered_architectures(), indent=2))


if __name__ == "__main__":
    main()
