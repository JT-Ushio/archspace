import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / "scripts/train/UG_Partial_Sharing/recover_report_payload.py"
spec = importlib.util.spec_from_file_location("report_payload_recovery_test", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

JSON_MARKER = "MOSE_REFERENCE_COMPARISON_JSON "
SHA_MARKER = "MOSE_REFERENCE_COMPARISON_SHA256 "
PREFIX = "[2026-10-06T23:00:19+08:00] example-job-cfcd2 >> "


def wire_report():
    value = {"路径": "包含 Unicode 与特殊字符\\\"", "rows": list(range(6000)), "loss": 2.4747629666328432}
    payload = json.dumps(value, ensure_ascii=False, allow_nan=False)
    digest = hashlib.sha256(payload.encode()).hexdigest()
    chunks = [payload[i:i+16384] for i in range(0, len(payload), 16384)]
    log = PREFIX + SHA_MARKER + digest + "\n" + PREFIX + JSON_MARKER + chunks[0] + "\n"
    log += "".join(PREFIX + chunk + "\n" for chunk in chunks[1:])
    log += PREFIX + '{"ExitCode":0}\n'
    return value, payload, digest, log


def test_recovers_transport_chunks_without_inserting_newlines():
    value, payload, digest, log = wire_report()
    recovered = module.recover_json(log, JSON_MARKER, SHA_MARKER)
    assert recovered == {"data": value, "payload": payload, "sha256": digest}


@pytest.mark.parametrize("failure", ["missing_tail", "damaged_chunk", "missing_hash", "duplicate_payload"])
def test_refuses_incomplete_damaged_or_ambiguous_reports(failure):
    _, _, digest, log = wire_report()
    if failure == "missing_tail":
        log = "\n".join(log.splitlines()[:2])
    elif failure == "damaged_chunk":
        log = log.replace('"loss": 2.4747629666328432', '"loss": 2.4747629666328433')
    elif failure == "missing_hash":
        log = log.replace(SHA_MARKER + digest, "")
    else:
        log += PREFIX + JSON_MARKER + "{}\n"
    with pytest.raises(ValueError):
        module.recover_json(log, JSON_MARKER, SHA_MARKER)
