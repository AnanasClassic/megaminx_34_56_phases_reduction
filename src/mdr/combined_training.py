from __future__ import annotations

import csv
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from .pair56_training import (
    _autocast,
    atomic_json,
    atomic_torch_save,
    build_allowed_moves,
    evaluate,
    make_generator,
    resolve_device,
    sample_rw_middle_batch,
    sha256,
    sparse_q_metrics,
)
from .qmlp import PairQMLP, count_parameters


@dataclass(frozen=True)
class CombinedTrainingSpec:
    pair: str
    transition: str
    K_max: int
    problem_path: Path
    output_path: Path
    load_problem: Callable[[Path], dict[str, Any]]


def append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def checkpoint_payload(
    *, spec: CombinedTrainingSpec, model: PairQMLP,
    optimizer: torch.optim.Optimizer, epoch: int, best_val: float,
    metadata: dict[str, Any], train_generator: torch.Generator,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": 1,
        "pair": spec.pair,
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


def train(args: Any, spec: CombinedTrainingSpec) -> dict[str, Any]:
    if args.smoke:
        args.epochs = 2
        args.steps_per_epoch = 4
        args.batch_size = min(args.batch_size, 32)
        args.val_size = min(args.val_size, 128)
        args.val_every = 1
        args.save_every = 1
    if args.K_min < 2 or args.K_min > spec.K_max:
        raise ValueError(f"K_min must be in 2..{spec.K_max}; K_max is fixed")

    problem_path = args.problem.resolve()
    problem = spec.load_problem(problem_path)
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
        if resume.get("pair") != spec.pair or resume["model_config"] != model.config:
            raise ValueError(f"resume checkpoint is incompatible with {spec.pair} model")
        if resume.get("training_metadata", {}).get("problem_sha256") != problem_sha256:
            raise ValueError("resume checkpoint used a different problem")
        model.load_state_dict(resume["model_state_dict"])
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        train_generator.set_state(
            resume["train_generator_state"].to(device="cpu", dtype=torch.uint8)
        )
        torch.set_rng_state(resume["torch_rng_state"].to(device="cpu", dtype=torch.uint8))
        if device.type == "cuda" and "cuda_rng_states" in resume:
            torch.cuda.set_rng_state_all([
                state.to(device="cpu", dtype=torch.uint8)
                for state in resume["cuda_rng_states"]
            ])
        start_epoch = int(resume["epoch"]) + 1
        best_val = float(resume["best_val_loss"])

    validation = sample_rw_middle_batch(
        batch_size=args.val_size, K_min=args.K_min, K_max=spec.K_max,
        target=target, actions=actions, inverse_actions=inverse_actions,
        face_ids=face_ids, allowed_moves=allowed_moves,
        generator=validation_generator, debug=True,
    )
    run_id = args.run_id or str(int(time.time()))
    output = args.output_dir.resolve()
    stem = f"{spec.pair}-qmlp_{run_id}"
    latest_path = output / f"{stem}_latest.pt"
    best_path = output / f"{stem}_best.pt"
    log_path = output / f"train_{stem}.csv"
    metadata_path = output / f"model_{stem}.json"
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "pair": spec.pair,
        "transition": spec.transition,
        "training_mode": "identity_sparse_q_random_walk",
        "run_id": run_id,
        "problem_file": str(problem_path),
        "problem_sha256": problem_sha256,
        "K_min": args.K_min,
        "K_max": spec.K_max,
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
        "event": "start", "pair": spec.pair, "run_id": run_id,
        "device": str(device), "parameters": metadata["num_parameters"],
        "K_min": args.K_min, "K_max": spec.K_max,
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
                batch_size=args.batch_size, K_min=args.K_min, K_max=spec.K_max,
                target=target, actions=actions, inverse_actions=inverse_actions,
                face_ids=face_ids, allowed_moves=allowed_moves,
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
                atomic_torch_save(checkpoint_payload(
                    spec=spec, model=model, optimizer=optimizer, epoch=epoch,
                    best_val=best_val, metadata=metadata, train_generator=train_generator,
                ), best_path)
        append_csv(log_path, row)
        if epoch % args.save_every == 0 or epoch == args.epochs:
            atomic_torch_save(checkpoint_payload(
                spec=spec, model=model, optimizer=optimizer, epoch=epoch,
                best_val=best_val, metadata=metadata, train_generator=train_generator,
            ), latest_path)
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
        "pair": spec.pair,
        "run_id": run_id,
        "best_checkpoint": str(best_path),
        "latest_checkpoint": str(latest_path),
        "metadata": str(metadata_path),
        "num_parameters": metadata["num_parameters"],
        "K_max": spec.K_max,
        "last_validation": last_validation,
    }
    print(json.dumps({"event": "complete", **result}, sort_keys=True), flush=True)
    return result
