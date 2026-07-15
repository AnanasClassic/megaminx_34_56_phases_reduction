#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_BYTES = 50 * 1024 * 1024


def contains_bytes(path: Path, pattern: bytes) -> bool:
    overlap = b""
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            block = overlap + chunk
            if pattern in block:
                return True
            overlap = block[-len(pattern) + 1 :]
    return False


def main() -> None:
    if not (ROOT / ".git").is_dir():
        raise SystemExit("error: initialize Git before auditing the publication tree")
    result = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, check=True, stdout=subprocess.PIPE,
    )
    files = [ROOT / item.decode() for item in result.stdout.split(b"\0") if item]
    oversized = [str(path.relative_to(ROOT)) for path in files if path.stat().st_size > MAX_FILE_BYTES]
    absolute_paths: list[str] = []
    forbidden: list[str] = []
    for path in files:
        relative = str(path.relative_to(ROOT))
        lowered = relative.lower()
        if "rokicki114" in lowered or "megaminx_phase1" in lowered:
            forbidden.append(relative)
        developer_home = (str(Path("/home") / "ananasclassic") + "/").encode()
        if contains_bytes(path, developer_home):
            absolute_paths.append(relative)
    if oversized or absolute_paths or forbidden:
        raise SystemExit(json.dumps({
            "valid": False,
            "oversized": oversized,
            "absolute_paths": absolute_paths,
            "forbidden_phase1_or_rokicki_files": forbidden,
        }, indent=2, sort_keys=True))
    print(json.dumps({
        "valid": True,
        "tracked_files": len(files),
        "largest_tracked_file_bytes": max(path.stat().st_size for path in files),
        "maximum_allowed_file_bytes": MAX_FILE_BYTES,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
