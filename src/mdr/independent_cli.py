from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .state import FullState, StateError, in_target, parse_word


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mdr-independent-verify")
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--solution", type=Path, required=True)
    parser.add_argument("--target", default="solved")
    parser.add_argument("--max-length", type=int, required=True)
    args = parser.parse_args(argv)
    try:
        if args.max_length < 0:
            raise StateError("max-length must be nonnegative")
        state = FullState.read(args.state)
        word = parse_word(args.solution.read_bytes())
        if len(word) > args.max_length:
            raise StateError(f"solution length {len(word)} exceeds bound {args.max_length}")
        final = state.apply(word)
        if not in_target(final, args.target):
            raise StateError(f"solution does not reach target {args.target}")
        print(json.dumps({"valid": True, "length": len(word), "target": args.target.lower()}, sort_keys=True))
        return 0
    except (OSError, StateError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
