from __future__ import annotations

import argparse
import fcntl
import os
import subprocess
import sys
import shutil
from datetime import datetime, timezone
from pathlib import Path

from .artifacts import atomic_write_json
from .config import DEFAULT_CONFIG, config_sha256, load_config
from .resources import check_disk


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one manifested, single-writer job")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--jobs-dir", type=Path, default=Path("artifacts/jobs"))
    parser.add_argument("--projected-gib", type=float, default=0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required")

    config = load_config(DEFAULT_CONFIG)
    args.jobs_dir.mkdir(parents=True, exist_ok=True)
    policy = config["resource_policy"]
    budget = check_disk(
        args.jobs_dir,
        policy["minimum_free_gib"],
        args.projected_gib,
        project_root=DEFAULT_CONFIG.parents[1],
        maximum_project_gib=policy["maximum_project_gib"],
    )
    lock_path = args.jobs_dir / f"{args.job_id}.lock"
    manifest_path = args.jobs_dir / f"{args.job_id}.json"

    with lock_path.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"job already running: {args.job_id}", file=sys.stderr)
            return 73

        manifest = {
            "schema_version": 1,
            "job_id": args.job_id,
            "command": args.command,
            "upstream_commit": config["upstream"]["commit"],
            "config_sha256": config_sha256(DEFAULT_CONFIG),
            "status": "running",
            "started_at": now(),
            "pid": os.getpid(),
            "resource_policy": {
                "maximum_workers": policy["maximum_workers"],
                "cpu_nice": policy["cpu_nice"],
                "io_priority_class": policy["io_priority_class"],
                "io_priority_level": policy["io_priority_level"],
                "project_bytes_at_start": budget.project_bytes,
                "projected_additional_bytes": budget.projected_bytes,
                "maximum_project_bytes": budget.maximum_project_bytes,
            },
        }
        atomic_write_json(manifest_path, manifest)
        try:
            command = args.command
            if shutil.which("ionice") is not None:
                command = ["ionice", "-c", "2", "-n", str(policy["io_priority_level"]), "--", *command]
            worker_cap = str(policy["maximum_workers"])
            child_env = os.environ.copy()
            for variable in (
                "GOMAXPROCS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS",
            ):
                child_env[variable] = worker_cap
            completed = subprocess.run(
                command,
                check=False,
                env=child_env,
                preexec_fn=lambda: os.nice(policy["cpu_nice"]),
            )
        except KeyboardInterrupt:
            manifest["status"] = "interrupted"
            manifest["finished_at"] = now()
            atomic_write_json(manifest_path, manifest)
            return 130
        except OSError as exc:
            manifest["status"] = "failed"
            manifest["exit_code"] = 127
            manifest["error"] = str(exc)
            manifest["finished_at"] = now()
            atomic_write_json(manifest_path, manifest)
            print(f"failed to start job: {exc}", file=sys.stderr)
            return 127
        manifest["exit_code"] = completed.returncode
        manifest["status"] = "complete" if completed.returncode == 0 else "failed"
        manifest["finished_at"] = now()
        atomic_write_json(manifest_path, manifest)
        return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
