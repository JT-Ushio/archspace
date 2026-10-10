"""Recover a report split into scheduler log chunks, requiring its SHA256."""

import hashlib
import json
import re


def recover_json(raw, json_marker, sha_marker):
    digests = re.findall(re.escape(sha_marker) + r"([a-f0-9]{64})", raw)
    lines = raw.splitlines()
    starts = [index for index, line in enumerate(lines) if json_marker in line]
    if len(digests) != 1 or len(starts) != 1:
        raise ValueError("Require exactly one report payload and one checksum")
    index = starts[0]
    payload = lines[index].split(json_marker, 1)[1]
    while hashlib.sha256(payload.encode()).hexdigest() != digests[0]:
        index += 1
        if index >= len(lines) or " >> " not in lines[index]:
            raise ValueError("Incomplete or damaged scheduler report payload")
        payload += lines[index].split(" >> ", 1)[1]
    return {"payload": payload, "sha256": digests[0], "data": json.loads(payload)}
