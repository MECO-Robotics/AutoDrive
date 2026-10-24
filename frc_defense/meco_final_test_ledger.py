"""Shared guard for sealed final-test consumption across evaluation and training."""
from __future__ import annotations

import json
from pathlib import Path


def assert_training_not_released_after_final_test(protocol_path: str | Path,
        protocol_id: str, protocol_sha256: str, source_hash: str) -> None:
    """Reject all training after the protocol's one sealed test event.

    This is called at the trainer API boundary after runtime initialization so
    the current source identity includes the prepared native extension.
    """
    ledger_path = Path(protocol_path).with_name(
        f"{protocol_id}.sealed-final-test-consumption.json")
    if not ledger_path.is_file():
        return
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    if (ledger.get("protocol_id") != protocol_id or
            ledger.get("protocol_sha256") != protocol_sha256):
        raise ValueError("sealed final-test ledger protocol identity differs from training protocol")
    if ledger.get("source_identity_sha256") != source_hash:
        raise ValueError("source identity changed after sealed final-test consumption")
    raise ValueError("training is forbidden after final-test seeds have been consumed")
