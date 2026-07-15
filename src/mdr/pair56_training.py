from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch

from .config import ROOT
from .pair56_problem import K_MAX, load_problem
from .qmlp import PairQMLP, count_parameters


DEFAULT_PROBLEM = ROOT / "training" / "pair56" / "problem.json"
DEFAULT_OUTPUT = ROOT / "models" / "pair56"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def make_generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device.type)
    generator.manual_seed(seed)
    return generator


def build_allowed_moves(face_ids: torch.Tensor) -> torch.Tensor:
    rows = []
    for face in torch.unique(face_ids, sorted=True).tolist():
        rows.append(torch.nonzero(face_ids != int(face), as_tuple=False).flatten())
    widths = {int(row.numel()) for row in rows}
    expected = int(face_ids.numel()) - 4
    if widths != {expected}:
        raise ValueError(f"expected {expected} continuations after each FTM face, got {widths}")
    return torch.stack(rows)


def sample_nontrivial_moves(
    *, states: torch.Tensor, last_moves: torch.Tensor, actions: torch.Tensor,
    face_ids: torch.Tensor, allowed_moves: torch.Tensor, generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample uniformly from nontrivial, same-face-reduced Schreier edges.

    Evaluate the complete 20-action row in one fixed-shape CUDA operation.  The
    previous rejection sampler used dynamic boolean indexing, which forced a
    device synchronization at nearly every random-walk step.
    """
    batch_size = states.size(0)
    n_actions = actions.size(0)
    expected_shape = (n_actions // 4, n_actions - 4)
    if tuple(allowed_moves.shape) != expected_shape:
        raise ValueError(
            f"expected a {expected_shape} same-face continuation table, got {tuple(allowed_moves.shape)}"
        )
    expanded_actions = actions.unsqueeze(0).expand(batch_size, -1, -1)
    expanded_states = states.unsqueeze(1).expand(-1, n_actions, -1)
    successors = torch.gather(expanded_states, 2, expanded_actions)
    eligible = (successors != expanded_states).any(dim=2)

    has_last = last_moves >= 0
    last_faces = face_ids.index_select(0, last_moves.clamp_min(0))
    eligible &= (~has_last).unsqueeze(1) | face_ids.unsqueeze(0).ne(last_faces.unsqueeze(1))
    torch._assert_async(
        eligible.any(dim=1).all(),
        "quotient state has no nontrivial continuation outside the previous face",
    )

    # IID continuous scores make argmax a uniform choice over eligible moves.
    scores = torch.rand(
        (batch_size, n_actions), generator=generator, device=states.device
    ).masked_fill(~eligible, -1.0)
    moves = scores.argmax(dim=1)
    rows = torch.arange(batch_size, device=states.device)
    return moves, successors[rows, moves]


def sample_rw_middle_batch(
    *,
    batch_size: int,
    K_min: int,
    target: torch.Tensor,
    actions: torch.Tensor,
    inverse_actions: torch.Tensor,
    face_ids: torch.Tensor,
    allowed_moves: torch.Tensor,
    generator: torch.Generator,
    debug: bool = False,
    K_max: int = K_MAX,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Identity-only form of the sparse-Q sampler used by model 1783930119."""
    if batch_size <= 0 or K_min < 2 or K_min > K_max:
        raise ValueError(f"batch_size must be positive and K_min must be in 2..{K_max}")
    device = target.device
    lengths = torch.randint(K_min, K_max + 1, (batch_size,), generator=generator, device=device)
    pivots = (
        torch.rand((batch_size,), generator=generator, device=device)
        * (lengths - 1).to(torch.float32)
    ).to(torch.int64) + 1
    states = target.unsqueeze(0).expand(batch_size, -1).clone()
    pivot_states = torch.empty_like(states)
    last_moves = torch.full((batch_size,), -1, dtype=torch.int64, device=device)
    previous_actions = torch.full_like(last_moves, -1)
    next_actions = torch.full_like(last_moves, -1)
    captured = torch.zeros((batch_size,), dtype=torch.bool, device=device)
    for step in range(K_max):
        moves, next_states = sample_nontrivial_moves(
            states=states, last_moves=last_moves, actions=actions, face_ids=face_ids,
            allowed_moves=allowed_moves, generator=generator,
        )

        at_pivot = pivots == step
        pivot_states = torch.where(at_pivot.unsqueeze(1), states, pivot_states)
        previous_actions = torch.where(
            at_pivot,
            inverse_actions.index_select(0, last_moves.clamp_min(0)),
            previous_actions,
        )
        next_actions = torch.where(at_pivot, moves, next_actions)
        captured |= at_pivot

        states = next_states
        last_moves = moves

    if debug:
        if not bool(captured.all()):
            raise RuntimeError("random-walk sampler missed a pivot")
        if bool((previous_actions == next_actions).any()):
            raise RuntimeError("sparse-Q actions unexpectedly coincide")
        if bool((face_ids[previous_actions] == face_ids[next_actions]).any()):
            raise RuntimeError("sparse-Q actions use the same face")
        replayed = torch.gather(
            pivot_states, 1, actions.index_select(0, next_actions)
        )
        if bool((replayed == pivot_states).all(dim=1).any()):
            raise RuntimeError("sparse-Q sampler emitted a quotient self-loop")
    return pivot_states, pivots, previous_actions, next_actions


def sparse_q_metrics(
    predictions: torch.Tensor,
    pivots: torch.Tensor,
    previous_actions: torch.Tensor,
    next_actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = torch.arange(predictions.size(0), device=predictions.device)
    previous = predictions[rows, previous_actions].float()
    following = predictions[rows, next_actions].float()
    targets = torch.stack(((pivots - 1).float(), (pivots + 1).float()), dim=1)
    selected = torch.stack((previous, following), dim=1)
    loss = torch.mean((selected - targets) ** 2)
    accuracy = torch.mean((previous < following).to(torch.float32))
    return loss, accuracy


def _autocast(device: torch.device, amp: str):
    if device.type != "cuda" or amp == "fp32":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


@torch.no_grad()
def evaluate(
    model: PairQMLP,
    validation: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    batch_size: int,
    device: torch.device,
    amp: str,
) -> dict[str, float]:
    model.eval()
    states, pivots, previous, following = validation
    loss_sum = accuracy_sum = 0.0
    items = 0
    for start in range(0, states.size(0), batch_size):
        end = min(start + batch_size, states.size(0))
        with _autocast(device, amp):
            predictions = model(states[start:end])
            loss, accuracy = sparse_q_metrics(
                predictions, pivots[start:end], previous[start:end], following[start:end]
            )
        count = end - start
        loss_sum += float(loss.item()) * count
        accuracy_sum += float(accuracy.item()) * count
        items += count
    return {"loss": loss_sum / items, "pair_accuracy": accuracy_sum / items}


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    torch.save(payload, temporary)
    temporary.replace(path)


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def checkpoint_payload(
    *, model: PairQMLP, optimizer: torch.optim.Optimizer, epoch: int,
    best_val: float, metadata: dict[str, Any], train_generator: torch.Generator,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": 1,
        "pair": "pair56",
        "epoch": epoch,
        "best_val_loss": best_val,
        "model_config": model.config,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_generator_state": train_generator.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "training_metadata": metadata,
    }
    if torch.cuda.is_available():
        result["cuda_rng_states"] = torch.cuda.get_rng_state_all()
    return result


def train(args: argparse.Namespace) -> dict[str, Any]:
    if args.smoke:
        args.epochs = 2
        args.steps_per_epoch = 4
        args.batch_size = min(args.batch_size, 32)
        args.val_size = min(args.val_size, 128)
        args.val_every = 1
        args.save_every = 1
    if args.K_min < 2 or args.K_min > K_MAX:
        raise ValueError(f"K_min must be in 2..{K_MAX}; K_max is fixed at {K_MAX}")
    problem_path = args.problem.resolve()
    problem = load_problem(problem_path)
    problem_sha256 = sha256(problem_path)
    device = resolve_device(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True

    actions = torch.tensor(problem["actions"], dtype=torch.int64, device=device)
    target = torch.tensor(problem["target"], dtype=torch.int64, device=device)
    inverse_actions = torch.tensor(problem["inverse_actions"], dtype=torch.int64, device=device)
    face_ids = torch.tensor(problem["face_ids"], dtype=torch.int64, device=device)
    allowed_moves = build_allowed_moves(face_ids).to(device)
    train_generator = make_generator(device, args.seed)
    validation_generator = make_generator(device, args.seed + 1_000_000)

    model = PairQMLP(
        state_size=problem["state_size"], num_classes=problem["num_classes"],
        actions=len(problem["actions"]), hd1=args.hd1, hd2=args.hd2,
        residual_blocks=args.residual_blocks, dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_epoch = 1
    best_val = math.inf
    if args.resume:
        resume = torch.load(args.resume, map_location=device, weights_only=False)
        if resume.get("pair") != "pair56" or resume["model_config"] != model.config:
            raise ValueError("resume checkpoint is incompatible with this pair56 model")
        if resume.get("training_metadata", {}).get("problem_sha256") != problem_sha256:
            raise ValueError("resume checkpoint used a different pair56 problem")
        model.load_state_dict(resume["model_state_dict"])
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        train_generator.set_state(resume["train_generator_state"].to(device="cpu", dtype=torch.uint8))
        torch.set_rng_state(resume["torch_rng_state"].to(device="cpu"))
        if device.type == "cuda" and "cuda_rng_states" in resume:
            torch.cuda.set_rng_state_all([
                state.to(device="cpu", dtype=torch.uint8)
                for state in resume["cuda_rng_states"]
            ])
        start_epoch = int(resume["epoch"]) + 1
        best_val = float(resume["best_val_loss"])

    validation = sample_rw_middle_batch(
        batch_size=args.val_size, K_min=args.K_min, target=target, actions=actions,
        inverse_actions=inverse_actions, face_ids=face_ids, allowed_moves=allowed_moves,
        generator=validation_generator, debug=True,
    )
    run_id = args.run_id or str(int(time.time()))
    output = args.output_dir.resolve()
    stem = f"pair56-qmlp_{run_id}"
    latest_path = output / f"{stem}_latest.pt"
    best_path = output / f"{stem}_best.pt"
    log_path = output / f"train_{stem}.csv"
    metadata_path = output / f"model_{stem}.json"
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "pair": "pair56",
        "training_mode": "identity_sparse_q_random_walk",
        "run_id": run_id,
        "problem_file": str(problem_path),
        "problem_sha256": problem_sha256,
        "K_min": args.K_min,
        "K_max": K_MAX,
        "K_max_fixed": True,
        "move_ban": "same_face",
        "symmetry_policy": "identity_only",
        "model_config": model.config,
        "num_parameters": count_parameters(model),
        "optimizer": "AdamW",
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "batch_size": args.batch_size,
        "steps_per_epoch": args.steps_per_epoch,
        "epochs": args.epochs,
        "val_size": args.val_size,
        "val_every": args.val_every,
        "log_every": args.log_every,
        "seed": args.seed,
        "device": str(device),
        "amp": args.amp,
        "best_checkpoint": str(best_path),
        "latest_checkpoint": str(latest_path),
        "log_file": str(log_path),
        "complete": False,
    }
    atomic_json(metadata, metadata_path)
    print(json.dumps({
        "event": "start", "run_id": run_id, "device": str(device),
        "parameters": metadata["num_parameters"], "K_min": args.K_min, "K_max": K_MAX,
    }, sort_keys=True), flush=True)

    last_validation: dict[str, float] | None = None
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        started = time.perf_counter()
        loss_sum = torch.zeros((), dtype=torch.float32, device=device)
        accuracy_sum = torch.zeros((), dtype=torch.float32, device=device)
        items = 0
        for _ in range(args.steps_per_epoch):
            batch = sample_rw_middle_batch(
                batch_size=args.batch_size, K_min=args.K_min, target=target, actions=actions,
                inverse_actions=inverse_actions, face_ids=face_ids, allowed_moves=allowed_moves,
                generator=train_generator,
            )
            with _autocast(device, args.amp):
                predictions = model(batch[0])
                loss, accuracy = sparse_q_metrics(predictions, batch[1], batch[2], batch[3])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            count = int(batch[0].size(0))
            loss_sum += loss.detach().float() * count
            accuracy_sum += accuracy.detach().float() * count
            items += count

        # One synchronization per epoch instead of two per optimizer step.
        train_loss = float((loss_sum / items).item())
        train_accuracy = float((accuracy_sum / items).item())
        row: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_pair_accuracy": train_accuracy,
            "seconds": time.perf_counter() - started,
            "vertices_seen": items,
            "val_loss": "",
            "val_pair_accuracy": "",
        }
        if epoch == 1 or epoch % args.val_every == 0 or epoch == args.epochs:
            last_validation = evaluate(model, validation, args.val_batch_size, device, args.amp)
            row["val_loss"] = last_validation["loss"]
            row["val_pair_accuracy"] = last_validation["pair_accuracy"]
            if last_validation["loss"] < best_val:
                best_val = last_validation["loss"]
                atomic_torch_save(
                    checkpoint_payload(
                        model=model, optimizer=optimizer, epoch=epoch, best_val=best_val,
                        metadata=metadata, train_generator=train_generator,
                    ),
                    best_path,
                )
        append_csv(log_path, row)
        if epoch % args.save_every == 0 or epoch == args.epochs:
            atomic_torch_save(
                checkpoint_payload(
                    model=model, optimizer=optimizer, epoch=epoch, best_val=best_val,
                    metadata=metadata, train_generator=train_generator,
                ),
                latest_path,
            )
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(json.dumps({"event": "epoch", **row}, sort_keys=True), flush=True)

    metadata.update({
        "best_val_loss": best_val,
        "latest_epoch": args.epochs,
        "last_validation": last_validation,
        "complete": True,
    })
    atomic_json(metadata, metadata_path)
    result = {
        "run_id": run_id,
        "best_checkpoint": str(best_path),
        "latest_checkpoint": str(latest_path),
        "metadata": str(metadata_path),
        "num_parameters": metadata["num_parameters"],
        "K_max": K_MAX,
        "last_validation": last_validation,
    }
    print(json.dumps({"event": "complete", **result}, sort_keys=True), flush=True)
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Train the pair56 identity-symmetry sparse-Q MLP.")
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
        train(args)
        return 0
    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=__import__("sys").stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
