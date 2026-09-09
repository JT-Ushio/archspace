"""Read-only W&B export for the four precision ablations. No training imports."""
import json
from datetime import datetime, timezone
from pathlib import Path

import wandb


def sanitize(value):
    if isinstance(value, dict):
        return {k: sanitize(v) for k, v in value.items()
                if not any(s in k.lower() for s in ("api_key", "password", "secret", "credential", "access_token", "environ"))}
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    return value


def main():
    output = Path(__file__).parent / "results_four_runs"
    output.mkdir(exist_ok=True)
    api = wandb.Api(timeout=60)
    for precision in ("bf16", "fp8"):
        for activation in ("silu", "rms"):
            label = f"{precision}_{activation}"
            path = f"archspace/MoSE/olmo3-1b-precision-down-{activation}-{precision}"
            run = api.run(path)
            metadata = dict(label=label, path=path, url=run.url, name=run.name,
                            state=run.state, created_at=run.created_at,
                            exported_at=datetime.now(timezone.utc).isoformat(),
                            config=sanitize(run.config), summary=sanitize(dict(run.summary)))
            (output / f"{label}_metadata.json").write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False, default=str))
            count = 0
            keys = set()
            with (output / f"{label}_history.jsonl").open("w") as handle:
                for row in run.scan_history(page_size=1000):
                    handle.write(json.dumps(sanitize(row), ensure_ascii=False, default=str) + "\n")
                    count += 1
                    keys.update(row)
            print(label, run.state, count, "rows", flush=True)
            print("keys:", sorted(keys), flush=True)


if __name__ == "__main__":
    main()
