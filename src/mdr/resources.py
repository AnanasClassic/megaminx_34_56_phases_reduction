from __future__ import annotations

import shutil
import os
from dataclasses import dataclass
from pathlib import Path


GIB = 1024 ** 3


class ResourceError(RuntimeError):
    pass


@dataclass(frozen=True)
class DiskBudget:
    path: Path
    free_bytes: int
    projected_bytes: int
    reserve_bytes: int
    project_bytes: int = 0
    maximum_project_bytes: int = 0

    @property
    def remaining_bytes(self) -> int:
        return self.free_bytes - self.projected_bytes

    @property
    def projected_project_bytes(self) -> int:
        return self.project_bytes + self.projected_bytes


def directory_size(path: Path | str) -> int:
    """Return allocated file bytes without following symlinks."""
    total = 0
    stack = [Path(path)]
    while stack:
        current = stack.pop()
        with os.scandir(current) as entries:
            for entry in entries:
                try:
                    if entry.is_symlink():
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(Path(entry.path))
                    elif entry.is_file(follow_symlinks=False):
                        total += entry.stat(follow_symlinks=False).st_size
                except FileNotFoundError:
                    continue
    return total


def check_disk(
    path: Path | str,
    minimum_free_gib: float,
    projected_gib: float = 0,
    *,
    project_root: Path | str | None = None,
    maximum_project_gib: float | None = None,
) -> DiskBudget:
    target = Path(path).resolve()
    target.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(target).free
    projected = int(projected_gib * GIB)
    reserve = int(minimum_free_gib * GIB)
    project_bytes = directory_size(project_root) if project_root is not None else 0
    maximum_project_bytes = int(maximum_project_gib * GIB) if maximum_project_gib is not None else 0
    budget = DiskBudget(target, free, projected, reserve, project_bytes, maximum_project_bytes)
    if budget.remaining_bytes < reserve:
        raise ResourceError(
            f"disk guard failed: free={free / GIB:.2f} GiB, "
            f"projected={projected_gib:.2f} GiB, reserve={minimum_free_gib:.2f} GiB"
        )
    if maximum_project_bytes and budget.projected_project_bytes > maximum_project_bytes:
        raise ResourceError(
            f"project size guard failed: current={project_bytes / GIB:.2f} GiB, "
            f"projected additional={projected_gib:.2f} GiB, "
            f"cap={maximum_project_gib:.2f} GiB"
        )
    return budget
