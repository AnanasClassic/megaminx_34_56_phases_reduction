from __future__ import annotations

import argparse
import hashlib
import io
import json
import statistics
import time
from pathlib import Path

import torch

from .pair34_problem import K_MAX, load_problem
from .pair34_training import DEFAULT_PROBLEM
from .pair34_certificates import PROOF_MAX_LENGTH, Pair34CertificateStore
from .config import ROOT
from .pair56_testing import apply_word, batched_beam_solve, load_hard_states, random_walk_states
from .pair56_training import (
    build_allowed_moves,
    evaluate,
    make_generator,
    resolve_device,
    sample_rw_middle_batch,
    sha256,
)
from .qmlp import PairQMLP, count_parameters
from .state import FullState, Move, in_target


DEFAULT_HARD_IDS = ROOT / "reductions" / "pair34" / "remaining_ids.bin"
DEFAULT_HARD_STATES = ROOT / "compositions" / "pair34" / "unique_states.bin"


def load_checkpoint(checkpoint: Path, problem_path: Path, device: torch.device):
    problem = load_problem(problem_path)
    checkpoint_bytes = checkpoint.read_bytes()
    payload = torch.load(io.BytesIO(checkpoint_bytes), map_location=device, weights_only=False)
    if payload.get("pair") != "pair34" or payload.get("schema_version") != 1:
        raise ValueError("not a pair34 QMLP checkpoint")
    if payload.get("training_metadata", {}).get("problem_sha256") != sha256(problem_path):
        raise ValueError("checkpoint problem checksum mismatch")
    model = PairQMLP(**payload["model_config"]).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return problem, payload, model, hashlib.sha256(checkpoint_bytes).hexdigest()


def test(args: argparse.Namespace) -> dict[str, object]:
    device = resolve_device(args.device)
    problem_path = args.problem.resolve()
    problem, checkpoint, model, checkpoint_sha256 = load_checkpoint(
        args.checkpoint.resolve(), problem_path, device
    )
    actions = torch.tensor(problem["actions"], dtype=torch.int64, device=device)
    target = torch.tensor(problem["target"], dtype=torch.int64, device=device)
    inverses = torch.tensor(problem["inverse_actions"], dtype=torch.int64, device=device)
    face_ids = torch.tensor(problem["face_ids"], dtype=torch.int64, device=device)
    allowed = build_allowed_moves(face_ids).to(device)

    validation = sample_rw_middle_batch(
        batch_size=args.val_size, K_min=args.K_min, K_max=K_MAX,
        target=target, actions=actions, inverse_actions=inverses,
        face_ids=face_ids, allowed_moves=allowed,
        generator=make_generator(device, args.seed + 1_000_000), debug=True,
    )
    validation_metrics = evaluate(model, validation, args.val_batch_size, device, args.amp)

    solved = 0
    lengths: list[int] = []
    unsolved_ids: list[int] = []
    elapsed = 0.0
    hard_mode = args.hard_states > 0
    requested = args.hard_states if hard_mode else args.tests
    hard_ids: list[int] = []
    requested_hard_ids: list[int] = []
    physical_states: list[FullState] | None = None
    certificate_store: Pair34CertificateStore | None = None
    already_certified = 0
    solved_new = 0
    pending_certificates = 0
    if requested:
        if hard_mode:
            hard_ids, physical_states, states = load_hard_states(
                count=args.hard_states, offset=args.hard_offset,
                ids_path=args.hard_ids.resolve(), states_path=args.composition_states.resolve(),
                problem=problem, device=device,
            )
            requested_hard_ids = hard_ids.copy()
            if args.solutions_db is not None:
                certificate_store = Pair34CertificateStore(
                    args.solutions_db.resolve(), problem_path=problem_path,
                    hard_ids_path=args.hard_ids.resolve(),
                    composition_states_path=args.composition_states.resolve(),
                )
                certified = certificate_store.certified_ids(hard_ids)
                if certified:
                    retained = [index for index, state_id in enumerate(hard_ids) if state_id not in certified]
                    hard_ids = [hard_ids[index] for index in retained]
                    physical_states = [physical_states[index] for index in retained]
                    states = [states[index] for index in retained]
                already_certified = len(certified)
                solved = already_certified
        else:
            states = random_walk_states(
                count=args.tests, depth=args.scramble_depth, target=target,
                actions=actions, face_ids=face_ids, allowed_moves=allowed,
                generator=make_generator(device, args.seed),
            )
        started = time.perf_counter()
        for start in range(0, len(states), args.search_batch_size):
            state_slice = states[start : start + args.search_batch_size]
            solutions = batched_beam_solve(
                model=model, initials=torch.stack(state_slice), target=target,
                actions=actions, face_ids=face_ids, beam_width=args.beam_width,
                max_steps=args.max_steps, amp=args.amp,
            )
            for local, (state, solution) in enumerate(zip(state_slice, solutions)):
                absolute = start + local
                if solution is None:
                    if hard_mode:
                        unsolved_ids.append(hard_ids[absolute])
                    continue
                if not torch.equal(apply_word(state, actions, solution), target):
                    raise RuntimeError("beam search returned an invalid pair34 word")
                if physical_states is not None:
                    word = tuple(
                        Move(problem["names"][move][:-1], int(problem["names"][move][-1]))
                        for move in solution
                    )
                    if not in_target(physical_states[absolute].apply(word), "g7"):
                        raise RuntimeError("beam word failed independent FullStateV1 G7 replay")
                    if certificate_store is not None:
                        certificate_store.add(
                            state_id=hard_ids[absolute],
                            state_bytes=physical_states[absolute].encode(),
                            solution=solution, beam_width=args.beam_width,
                            checkpoint_sha256_hex=checkpoint_sha256,
                            checkpoint_epoch=int(checkpoint["epoch"]),
                        )
                        pending_certificates += 1
                        if pending_certificates >= args.certificate_commit_every:
                            certificate_store.commit()
                            pending_certificates = 0
                solved += 1
                solved_new += 1
                lengths.append(len(solution))
        elapsed = time.perf_counter() - started
        if certificate_store is not None:
            certificate_store.commit()

    certificate_statistics = certificate_store.statistics() if certificate_store is not None else None
    if certificate_store is not None:
        certificate_store.close()
    searched = len(states) if requested else 0

    result: dict[str, object] = {
        "valid": True,
        "pair": "pair34",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "num_parameters": count_parameters(model),
        "K_max": K_MAX,
        "validation": validation_metrics,
        "evaluation_source": "hard_representatives" if hard_mode else "random_walks",
        "random_walk_tests": 0 if hard_mode else args.tests,
        "hard_states_tested": args.hard_states if hard_mode else 0,
        "states_searched": searched,
        "already_certified": already_certified,
        "solved_new": solved_new,
        "hard_offset": args.hard_offset if hard_mode else None,
        "first_hard_state_id": requested_hard_ids[0] if requested_hard_ids else None,
        "last_hard_state_id": requested_hard_ids[-1] if requested_hard_ids else None,
        "unsolved_hard_state_ids": unsolved_ids,
        "certificate_store": certificate_statistics,
        "scramble_depth": None if hard_mode else args.scramble_depth,
        "beam_width": args.beam_width,
        "search_batch_size": args.search_batch_size,
        "max_steps": args.max_steps,
        "solved": solved,
        "solve_rate": solved / requested if requested else None,
        "median_solution_length": statistics.median(lengths) if lengths else None,
        "maximum_solution_length": max(lengths) if lengths else None,
        "elapsed_seconds": elapsed,
        "states_per_second": searched / elapsed if elapsed else None,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.require_all and solved != requested:
        raise RuntimeError(f"only {solved}/{requested} pair34 states were solved")
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Evaluate a pair34 QMLP checkpoint.")
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--problem", type=Path, default=DEFAULT_PROBLEM)
    result.add_argument("--tests", type=int, default=100)
    result.add_argument("--hard-states", type=int, default=0)
    result.add_argument("--hard-offset", type=int, default=0)
    result.add_argument("--hard-ids", type=Path, default=DEFAULT_HARD_IDS)
    result.add_argument("--composition-states", type=Path, default=DEFAULT_HARD_STATES)
    result.add_argument("--solutions-db", type=Path)
    result.add_argument("--certificate-commit-every", type=int, default=1_024)
    result.add_argument("--scramble-depth", type=int, default=100)
    result.add_argument("--beam-width", type=int, default=32)
    result.add_argument("--search-batch-size", type=int, default=100)
    result.add_argument("--max-steps", type=int, default=K_MAX)
    result.add_argument("--val-size", type=int, default=1_024)
    result.add_argument("--val-batch-size", type=int, default=1_024)
    result.add_argument("--K-min", dest="K_min", type=int, default=2)
    result.add_argument("--seed", type=int, default=43)
    result.add_argument("--device", default="auto")
    result.add_argument("--amp", choices=("bf16", "fp32"), default="bf16")
    result.add_argument("--require-all", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.tests < 0 or args.hard_states < 0 or args.hard_offset < 0 or min(
            args.scramble_depth, args.beam_width, args.search_batch_size,
            args.max_steps, args.val_size, args.val_batch_size,
        ) <= 0:
            raise ValueError("evaluation counts must be positive")
        if args.max_steps > K_MAX:
            raise ValueError(f"pair34 evaluation max_steps cannot exceed fixed K_max={K_MAX}")
        if args.solutions_db is not None and args.hard_states <= 0:
            raise ValueError("--solutions-db requires --hard-states")
        if args.solutions_db is not None and args.max_steps > PROOF_MAX_LENGTH:
            raise ValueError(f"certificate solutions cannot exceed pair34 proof bound {PROOF_MAX_LENGTH}")
        if args.certificate_commit_every <= 0:
            raise ValueError("--certificate-commit-every must be positive")
        test(args)
        return 0
    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=__import__("sys").stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
