from __future__ import annotations

import argparse
import hashlib
import io
import json
import statistics
import struct
import time
from pathlib import Path

import torch

from .config import ROOT
from .pair56_problem import K_MAX, full_state_to_pair56, load_problem
from .pair56_certificates import PROOF_MAX_LENGTH, Pair56CertificateStore
from .pair56_training import (
    DEFAULT_PROBLEM,
    _autocast,
    build_allowed_moves,
    evaluate,
    make_generator,
    resolve_device,
    sample_nontrivial_moves,
    sample_rw_middle_batch,
    sha256,
)
from .qmlp import PairQMLP, count_parameters
from .state import FullState, Move, in_target


DEFAULT_HARD_IDS = ROOT / "reductions" / "pair56" / "remaining_ids.bin"
DEFAULT_HARD_STATES = ROOT / "compositions" / "pair56" / "unique_states.bin"


def load_checkpoint(checkpoint: Path, problem_path: Path, device: torch.device):
    problem = load_problem(problem_path)
    checkpoint_bytes = checkpoint.read_bytes()
    payload = torch.load(io.BytesIO(checkpoint_bytes), map_location=device, weights_only=False)
    if payload.get("pair") != "pair56" or payload.get("schema_version") != 1:
        raise ValueError("not a pair56 QMLP checkpoint")
    metadata = payload.get("training_metadata", {})
    if metadata.get("problem_sha256") != sha256(problem_path):
        raise ValueError("checkpoint problem checksum mismatch")
    model = PairQMLP(**payload["model_config"]).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return problem, payload, model, hashlib.sha256(checkpoint_bytes).hexdigest()


def apply_word(state: torch.Tensor, actions: torch.Tensor, word: tuple[int, ...]) -> torch.Tensor:
    result = state.clone()
    for action in word:
        result = result.index_select(0, actions[action])
    return result


@torch.no_grad()
def beam_solve(
    *, model: PairQMLP, initial: torch.Tensor, target: torch.Tensor,
    actions: torch.Tensor, face_ids: torch.Tensor, beam_width: int,
    max_steps: int,
) -> tuple[int, ...] | None:
    if torch.equal(initial, target):
        return ()
    states = initial.unsqueeze(0)
    last_moves = torch.full((1,), -1, dtype=torch.int64, device=initial.device)
    paths: list[tuple[int, ...]] = [()]
    visited = {(bytes(initial.to(device="cpu", dtype=torch.uint8).tolist()), -1)}
    n_actions = int(actions.size(0))

    for _depth in range(1, max_steps + 1):
        q_values = model(states).float()
        valid = torch.ones_like(q_values, dtype=torch.bool)
        active = last_moves >= 0
        if bool(active.any()):
            valid[active] = face_ids.unsqueeze(0) != face_ids[last_moves[active]].unsqueeze(1)
        order = torch.argsort(q_values.masked_fill(~valid, torch.inf).flatten())
        parent_indices = torch.div(order, n_actions, rounding_mode="floor")
        move_indices = order % n_actions
        finite = valid.flatten().index_select(0, order)
        parent_indices = parent_indices[finite]
        move_indices = move_indices[finite]
        parents = states.index_select(0, parent_indices)
        candidates = torch.gather(parents, 1, actions.index_select(0, move_indices))

        next_states: list[torch.Tensor] = []
        next_last: list[int] = []
        next_paths: list[tuple[int, ...]] = []
        candidates_cpu = candidates.to(device="cpu", dtype=torch.uint8)
        parent_cpu = parent_indices.to(device="cpu").tolist()
        moves_cpu = move_indices.to(device="cpu").tolist()
        for row in range(candidates.size(0)):
            candidate = candidates[row]
            parent = parent_cpu[row]
            move = moves_cpu[row]
            path = paths[parent] + (move,)
            if torch.equal(candidate, target):
                return path
            key = (bytes(candidates_cpu[row].tolist()), int(face_ids[move].item()))
            if key in visited:
                continue
            visited.add(key)
            next_states.append(candidate)
            next_last.append(move)
            next_paths.append(path)
            if len(next_states) == beam_width:
                break
        if not next_states:
            return None
        states = torch.stack(next_states)
        last_moves = torch.tensor(next_last, dtype=torch.int64, device=initial.device)
        paths = next_paths
    return None


@torch.no_grad()
def batched_beam_solve(
    *, model: PairQMLP, initials: torch.Tensor, target: torch.Tensor,
    actions: torch.Tensor, face_ids: torch.Tensor, beam_width: int,
    max_steps: int, amp: str,
) -> list[tuple[int, ...] | None]:
    """Run independent beams with one model call and one CPU transfer per depth."""
    roots, state_size = initials.shape
    device = initials.device
    n_actions = int(actions.size(0))
    candidate_width = min(beam_width * n_actions, max(n_actions, beam_width * 16))
    target_key = target.to(device="cpu", dtype=torch.uint8).numpy().tobytes()
    face_ids_cpu = face_ids.to(device="cpu").tolist()

    states = target.view(1, 1, -1).expand(roots, beam_width, -1).clone()
    states[:, 0] = initials
    last_moves = torch.full((roots, beam_width), -1, dtype=torch.int64, device=device)
    counts = [1] * roots
    paths: list[list[tuple[int, ...]]] = [[()] for _ in range(roots)]
    states_cpu = states.to(device="cpu", dtype=torch.uint8).numpy()
    initial_cpu = states_cpu[:, 0]
    visited = [{(initial_cpu[root].tobytes(), -1)} for root in range(roots)]
    solutions: list[tuple[int, ...] | None] = [None] * roots
    done = [initial_cpu[root].tobytes() == target_key for root in range(roots)]
    for root, is_done in enumerate(done):
        if is_done:
            solutions[root] = ()
            counts[root] = 0

    for _depth in range(1, max_steps + 1):
        if all(done):
            break
        active = torch.zeros((roots, beam_width), dtype=torch.bool, device=device)
        for root, count in enumerate(counts):
            if not done[root] and count:
                active[root, :count] = True

        q_values = torch.full(
            (roots, beam_width, n_actions), torch.inf, dtype=torch.float32, device=device
        )
        with _autocast(device, amp):
            active_q = model(states[active]).float()
        q_values[active] = active_q
        has_last = last_moves >= 0
        last_faces = face_ids[last_moves.clamp_min(0)]
        valid = active.unsqueeze(2) & (
            (~has_last).unsqueeze(2)
            | face_ids.view(1, 1, -1).ne(last_faces.unsqueeze(2))
        )
        scores = q_values.masked_fill(~valid, torch.inf).flatten(1)
        order = torch.topk(
            scores, k=candidate_width, dim=1, largest=False, sorted=True
        ).indices
        first_width = min(candidate_width, max(n_actions, beam_width * 2))
        next_states_cpu = torch.empty(
            (roots, beam_width, state_size), dtype=torch.uint8
        ).numpy()
        next_last_cpu = torch.full((roots, beam_width), -1, dtype=torch.int64).numpy()
        next_counts = [0] * roots
        next_paths: list[list[tuple[int, ...]]] = [[] for _ in range(roots)]

        def process_order_slice(root_ids: list[int], start: int, end: int) -> set[int]:
            if not root_ids or start >= end:
                return set()
            root_index = torch.tensor(root_ids, dtype=torch.int64, device=device)
            selected_order = order.index_select(0, root_index)[:, start:end]
            parents = torch.div(selected_order, n_actions, rounding_mode="floor")
            moves = selected_order % n_actions
            ordered_valid = torch.gather(
                valid.flatten(1).index_select(0, root_index), 1, selected_order
            )
            selected_states = states.index_select(0, root_index)
            parent_states = torch.gather(
                selected_states, 1, parents.unsqueeze(2).expand(-1, -1, state_size)
            )
            candidates = torch.gather(parent_states, 2, actions[moves])
            candidates_cpu = candidates.to(device="cpu", dtype=torch.uint8).numpy()
            parents_cpu = parents.to(device="cpu").numpy()
            moves_cpu = moves.to(device="cpu").numpy()
            valid_cpu = ordered_valid.to(device="cpu").numpy()
            exhausted: set[int] = set()
            width = end - start

            for local_root, root in enumerate(root_ids):
                if done[root] or next_counts[root] == beam_width:
                    continue
                for row in range(width):
                    if not valid_cpu[local_root, row]:
                        exhausted.add(root)
                        break
                    parent = int(parents_cpu[local_root, row])
                    move = int(moves_cpu[local_root, row])
                    state_key = candidates_cpu[local_root, row].tobytes()
                    if state_key == states_cpu[root, parent].tobytes():
                        continue
                    path = paths[root][parent] + (move,)
                    if state_key == target_key:
                        solutions[root] = path
                        done[root] = True
                        next_counts[root] = 0
                        next_paths[root] = []
                        break
                    key = (state_key, face_ids_cpu[move])
                    if key in visited[root]:
                        continue
                    visited[root].add(key)
                    slot = next_counts[root]
                    next_states_cpu[root, slot] = candidates_cpu[local_root, row]
                    next_last_cpu[root, slot] = move
                    next_paths[root].append(path)
                    next_counts[root] += 1
                    if next_counts[root] == beam_width:
                        break
            return exhausted

        active_roots = [root for root in range(roots) if not done[root] and counts[root]]
        exhausted = process_order_slice(active_roots, 0, first_width)
        fallback_roots = [
            root for root in active_roots
            if not done[root] and next_counts[root] < beam_width and root not in exhausted
        ]
        process_order_slice(fallback_roots, first_width, candidate_width)

        states_cpu = next_states_cpu
        states = torch.from_numpy(states_cpu).to(device=device, dtype=torch.int64)
        last_moves = torch.from_numpy(next_last_cpu).to(device=device)
        counts = next_counts
        paths = next_paths
        for root, count in enumerate(counts):
            if not done[root] and count == 0:
                done[root] = True
    return solutions


def random_walk_states(
    *, count: int, depth: int, target: torch.Tensor, actions: torch.Tensor,
    face_ids: torch.Tensor, allowed_moves: torch.Tensor, generator: torch.Generator,
) -> list[torch.Tensor]:
    states = target.unsqueeze(0).expand(count, -1).clone()
    last_moves = torch.full((count,), -1, dtype=torch.int64, device=target.device)
    for _ in range(depth):
        moves, states = sample_nontrivial_moves(
            states=states, last_moves=last_moves, actions=actions, face_ids=face_ids,
            allowed_moves=allowed_moves, generator=generator,
        )
        last_moves = moves
    return list(states.unbind(0))


def load_hard_states(
    *, count: int, offset: int, ids_path: Path, states_path: Path,
    problem: dict[str, object], device: torch.device,
) -> tuple[list[int], list[FullState], list[torch.Tensor]]:
    ids_blob = ids_path.read_bytes()
    states_blob = states_path.read_bytes()
    if len(ids_blob) % 4 or len(states_blob) % 108:
        raise ValueError("malformed pair56 hard-state artifacts")
    total = len(ids_blob) // 4
    if offset < 0 or count <= 0 or offset + count > total:
        raise ValueError(f"hard-state slice {offset}:{offset + count} exceeds {total} representatives")
    ids = [
        struct.unpack_from("<I", ids_blob, 4 * row)[0]
        for row in range(offset, offset + count)
    ]
    physical: list[FullState] = []
    quotient: list[torch.Tensor] = []
    state_count = len(states_blob) // 108
    for state_id in ids:
        if state_id >= state_count:
            raise ValueError(f"hard-state ID {state_id} exceeds composition state count {state_count}")
        state = FullState.decode(states_blob[state_id * 108 : (state_id + 1) * 108])
        physical.append(state)
        quotient.append(torch.tensor(full_state_to_pair56(state, problem), device=device))
    return ids, physical, quotient


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
    validation_generator = make_generator(device, args.seed + 1_000_000)
    validation = sample_rw_middle_batch(
        batch_size=args.val_size, K_min=args.K_min, target=target, actions=actions,
        inverse_actions=inverses, face_ids=face_ids, allowed_moves=allowed,
        generator=validation_generator, debug=True,
    )
    validation_metrics = evaluate(model, validation, args.val_batch_size, device, args.amp)

    solved = 0
    solved_new = 0
    already_certified = 0
    lengths: list[int] = []
    unsolved_ids: list[int] = []
    elapsed = 0.0
    hard_mode = args.hard_states > 0
    hard_ids: list[int] = []
    requested_hard_ids: list[int] = []
    physical_states: list[FullState] | None = None
    requested = args.hard_states if hard_mode else args.tests
    certificate_store: Pair56CertificateStore | None = None
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
                certificate_store = Pair56CertificateStore(
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
            scramble_generator = make_generator(device, args.seed)
            states = random_walk_states(
                count=args.tests, depth=args.scramble_depth, target=target, actions=actions,
                face_ids=face_ids, allowed_moves=allowed, generator=scramble_generator,
            )
        started = time.perf_counter()
        for start in range(0, len(states), args.search_batch_size):
            state_batch = torch.stack(states[start : start + args.search_batch_size])
            solutions = batched_beam_solve(
                model=model, initials=state_batch, target=target, actions=actions,
                face_ids=face_ids, beam_width=args.beam_width, max_steps=args.max_steps,
                amp=args.amp,
            )
            for local, (state, solution) in enumerate(zip(
                states[start : start + args.search_batch_size], solutions
            )):
                absolute = start + local
                if solution is None:
                    if hard_mode:
                        unsolved_ids.append(hard_ids[absolute])
                    continue
                if not torch.equal(apply_word(state, actions, solution), target):
                    raise RuntimeError("beam search returned an invalid pair56 word")
                if physical_states is not None:
                    physical_word = tuple(Move(problem["names"][move][:-1], int(problem["names"][move][-1])) for move in solution)
                    if not in_target(physical_states[absolute].apply(physical_word), "g9"):
                        raise RuntimeError("beam word failed independent FullStateV1 G9 replay")
                    if certificate_store is not None:
                        certificate_store.add(
                            state_id=hard_ids[absolute],
                            state_bytes=physical_states[absolute].encode(),
                            solution=solution,
                            beam_width=args.beam_width,
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
        "pair": "pair56",
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
        raise RuntimeError(f"only {solved}/{requested} test states were solved")
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Evaluate a pair56 QMLP checkpoint.")
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--problem", type=Path, default=DEFAULT_PROBLEM)
    result.add_argument("--tests", type=int, default=100)
    result.add_argument("--hard-states", type=int, default=0)
    result.add_argument("--hard-offset", type=int, default=0)
    result.add_argument("--hard-ids", type=Path, default=DEFAULT_HARD_IDS)
    result.add_argument("--composition-states", type=Path, default=DEFAULT_HARD_STATES)
    result.add_argument("--solutions-db", type=Path)
    result.add_argument("--certificate-commit-every", type=int, default=1_024)
    result.add_argument("--scramble-depth", type=int, default=26)
    result.add_argument("--beam-width", type=int, default=256)
    result.add_argument("--search-batch-size", type=int, default=32)
    result.add_argument("--max-steps", type=int, default=K_MAX)
    result.add_argument("--val-size", type=int, default=16_384)
    result.add_argument("--val-batch-size", type=int, default=2_048)
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
            raise ValueError(f"pair56 evaluation max_steps cannot exceed fixed K_max={K_MAX}")
        if args.solutions_db is not None and args.hard_states <= 0:
            raise ValueError("--solutions-db requires --hard-states")
        if args.solutions_db is not None and args.max_steps > PROOF_MAX_LENGTH:
            raise ValueError(f"certificate solutions cannot exceed pair56 proof bound {PROOF_MAX_LENGTH}")
        if args.certificate_commit_every <= 0:
            raise ValueError("--certificate-commit-every must be positive")
        test(args)
        return 0
    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=__import__("sys").stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
