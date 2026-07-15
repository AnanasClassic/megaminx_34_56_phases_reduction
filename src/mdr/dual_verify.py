from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .config import ROOT
from .state import FullState, StateError, in_target, parse_word


def independent_result(state_path: Path, solution_path: Path, target: str, max_length: int) -> dict[str, object]:
    if max_length < 0:
        raise StateError("max-length must be nonnegative")
    state = FullState.read(state_path)
    word = parse_word(solution_path.read_bytes())
    if len(word) > max_length:
        raise StateError(f"solution length {len(word)} exceeds bound {max_length}")
    if not in_target(state.apply(word), target):
        raise StateError(f"solution does not reach target {target}")
    return {"valid": True, "length": len(word), "target": target.lower()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="verify")
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--solution", type=Path, required=True)
    parser.add_argument("--target", default="solved")
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--primary", type=Path, default=ROOT / "build" / "mdr-verify")
    args = parser.parse_args(argv)
    if not args.primary.is_file():
        print("error: primary verifier is absent; run make build-verifier", file=sys.stderr)
        return 2
    command = [
        str(args.primary), "verify",
        "--state", str(args.state),
        "--solution", str(args.solution),
        "--target", args.target,
        "--max-length", str(args.max_length),
    ]
    primary = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    try:
        independent = independent_result(args.state, args.solution, args.target, args.max_length)
        independent_error = None
    except (OSError, StateError) as exc:
        independent = None
        independent_error = str(exc)
    if primary.returncode != 0 or independent_error is not None:
        print("verification rejected", file=sys.stderr)
        if primary.stderr:
            print(f"primary: {primary.stderr.strip()}", file=sys.stderr)
        if independent_error:
            print(f"independent: error: {independent_error}", file=sys.stderr)
        if (primary.returncode == 0) != (independent_error is None):
            print("error: verifier disagreement", file=sys.stderr)
        return 2
    try:
        primary_result = json.loads(primary.stdout)
    except json.JSONDecodeError as exc:
        print(f"error: malformed primary verifier output: {exc}", file=sys.stderr)
        return 2
    if primary_result != independent:
        print(f"error: verifier disagreement: primary={primary_result!r} independent={independent!r}", file=sys.stderr)
        return 2
    print(json.dumps(independent, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
