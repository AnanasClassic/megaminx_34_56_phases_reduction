#!/usr/bin/env python3
"""Rebind untrusted checkpoint provenance after metadata-only checkpoint sanitation."""

from __future__ import annotations

import argparse
import hashlib
import sqlite3
import struct
from pathlib import Path


PREFIX = {"pair34": b"MDRP34CERT1\x00", "pair56": b"MDRP56CERT1\x00"}


def verification_digest(
    pair: str, state_id: int, state: bytes, solution: bytes, beam: int,
    checkpoint: bytes, epoch: int,
) -> bytes:
    digest = hashlib.sha256()
    digest.update(PREFIX[pair])
    digest.update(struct.pack("<III", state_id, beam, epoch))
    digest.update(checkpoint)
    digest.update(state)
    digest.update(solution)
    return digest.digest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pair", choices=sorted(PREFIX))
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--states", required=True, type=Path)
    parser.add_argument(
        "--mapping", action="append", required=True,
        help="OLD_SHA256:NEW_SHA256; may be repeated",
    )
    args = parser.parse_args()
    mapping = {
        bytes.fromhex(item.split(":", 1)[0]): bytes.fromhex(item.split(":", 1)[1])
        for item in args.mapping
    }
    states = args.states.read_bytes()
    if len(states) % 108:
        raise SystemExit("error: malformed FullStateV1 array")

    connection = sqlite3.connect(args.database)
    changed = 0
    try:
        connection.execute("PRAGMA journal_mode=DELETE")
        rows = connection.execute(
            "SELECT state_id, solution, beam_width, checkpoint_sha256, "
            "checkpoint_epoch FROM certificates ORDER BY state_id"
        )
        updates = []
        for state_id, solution, beam, old_checkpoint, epoch in rows:
            old_checkpoint = bytes(old_checkpoint)
            if old_checkpoint not in mapping:
                continue
            start = state_id * 108
            state = states[start : start + 108]
            if len(state) != 108:
                raise ValueError(f"state ID outside physical-state array: {state_id}")
            new_checkpoint = mapping[old_checkpoint]
            updates.append((
                new_checkpoint,
                verification_digest(
                    args.pair, state_id, state, bytes(solution), beam,
                    new_checkpoint, epoch,
                ),
                state_id,
            ))
            if len(updates) == 10_000:
                connection.executemany(
                    "UPDATE certificates SET checkpoint_sha256=?, "
                    "verification_sha256=? WHERE state_id=?", updates,
                )
                changed += len(updates)
                updates.clear()
        if updates:
            connection.executemany(
                "UPDATE certificates SET checkpoint_sha256=?, "
                "verification_sha256=? WHERE state_id=?", updates,
            )
            changed += len(updates)
        connection.commit()
        if changed != connection.execute("SELECT COUNT(*) FROM certificates").fetchone()[0]:
            raise ValueError("mapping did not cover every certificate")
        connection.execute("VACUUM")
    finally:
        connection.close()
    print(f"{args.pair}: rebound {changed} certificate provenance records")


if __name__ == "__main__":
    main()
