from __future__ import annotations

import argparse
import json
from pathlib import Path

from .combined_training import CombinedTrainingSpec, train
from .config import ROOT
from .pair34_problem import K_MAX, load_problem


DEFAULT_PROBLEM = ROOT / "training" / "pair34" / "problem.json"
DEFAULT_OUTPUT = ROOT / "models" / "pair34"
SPEC = CombinedTrainingSpec(
    pair="pair34", transition="G5->G7", K_max=K_MAX,
    problem_path=DEFAULT_PROBLEM, output_path=DEFAULT_OUTPUT,
    load_problem=load_problem,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Train the pair34 G5/G7 sparse-Q MLP.")
    result.add_argument("--problem", type=Path, default=DEFAULT_PROBLEM)
    result.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--epochs", type=int, default=8_192)
    result.add_argument("--steps-per-epoch", type=int, default=256)
    result.add_argument("--batch-size", type=int, default=1_024)
    result.add_argument("--K-min", dest="K_min", type=int, default=2)
    result.add_argument("--val-size", type=int, default=16_384)
    result.add_argument("--val-batch-size", type=int, default=2_048)
    result.add_argument("--val-every", type=int, default=32)
    result.add_argument("--log-every", type=int, default=1)
    result.add_argument("--save-every", type=int, default=100)
    result.add_argument("--lr", type=float, default=1e-4)
    result.add_argument("--weight-decay", type=float, default=0.003)
    result.add_argument("--grad-clip", type=float, default=1.0)
    result.add_argument("--hd1", type=int, default=64)
    result.add_argument("--hd2", type=int, default=256)
    result.add_argument("--residual-blocks", type=int, default=2)
    result.add_argument("--dropout", type=float, default=0.0)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--device", default="auto")
    result.add_argument("--amp", choices=("bf16", "fp32"), default="bf16")
    result.add_argument("--resume", type=Path)
    result.add_argument("--run-id")
    result.add_argument("--smoke", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if min(
            args.epochs, args.steps_per_epoch, args.batch_size, args.val_size,
            args.val_every, args.log_every, args.save_every,
        ) <= 0:
            raise ValueError("training counts must be positive")
        train(args, SPEC)
        return 0
    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=__import__("sys").stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
