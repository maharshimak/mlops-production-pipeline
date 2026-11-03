"""Canonical JSON serialization and SHA-256 helpers.

Every fingerprint produced by the experiment workflow hashes the same canonical
form: JSON with sorted keys, ``(",", ":")`` separators, UTF-8 bytes and no
NaN/Infinity tokens. Numeric values that are semantically numbers are converted
to ``float`` before hashing by the callers, so ``1`` and ``1.0`` hash equally.
"""

from __future__ import annotations

import json
from hashlib import sha256
from math import isfinite


def canonical_float(value: float) -> float:
    """Return a finite float with negative zero normalised to positive zero."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("expected an int or float")
    result = float(value)
    if not isfinite(result):
        raise ValueError("value must be finite")
    return result + 0.0


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return sha256(data).hexdigest()


def canonical_sha256(value: object) -> str:
    return sha256_hex(canonical_json_bytes(value))
