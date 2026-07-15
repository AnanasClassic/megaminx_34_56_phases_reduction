#!/usr/bin/env python3
"""Remove producing-machine paths from a PyTorch checkpoint without changing weights."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import torch


PUBLIC_PATHS = {
    "pair34": {
        "problem_file": "training/pair34/problem.json",
        "best_checkpoint": "phase_3_4/models/pair34-qmlp-epoch1120.pt",
        "latest_checkpoint": "artifacts/models/pair34/latest.pt",
        "log_file": "phase_3_4/models/training.csv",
    },
    "pair56": {
        "problem_file": "training/pair56/problem.json",
        "best_checkpoint": "phase_5_6/models/pair56-qmlp-epoch3744.pt",
        "latest_checkpoint": "artifacts/models/pair56/latest.pt",
        "log_file": "phase_5_6/models/training.csv",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contains_absolute_path(value: object) -> bool:
    if isinstance(value, str):
        return value.startswith("/home/")
    if isinstance(value, dict):
        return any(contains_absolute_path(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(contains_absolute_path(item) for item in value)
    return False


def sanitize(path: Path) -> tuple[str, str]:
    source_sha = sha256(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    pair = checkpoint.get("pair")
    if pair not in PUBLIC_PATHS:
        raise ValueError(f"{path}: unsupported checkpoint pair {pair!r}")
    metadata = checkpoint.get("training_metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"{path}: missing training_metadata")
    metadata.update(PUBLIC_PATHS[pair])
    if contains_absolute_path(checkpoint):
        raise ValueError(f"{path}: another producing-machine path remains")

    temporary = path.with_name(path.name + ".sanitized")
    torch.save(checkpoint, temporary)
    reloaded = torch.load(temporary, map_location="cpu", weights_only=False)
    if reloaded["epoch"] != checkpoint["epoch"] or reloaded["model_config"] != checkpoint["model_config"]:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"{path}: checkpoint round-trip changed its contract")
    for name, tensor in checkpoint["model_state_dict"].items():
        if not torch.equal(tensor, reloaded["model_state_dict"][name]):
            temporary.unlink(missing_ok=True)
            raise ValueError(f"{path}: model tensor changed: {name}")
    os.replace(temporary, path)
    return source_sha, sha256(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.checkpoint:
        source_sha, distributed_sha = sanitize(path)
        print(f"{path}: source={source_sha} distributed={distributed_sha}")


if __name__ == "__main__":
    main()
