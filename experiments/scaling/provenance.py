"""Identify training code in a checkout or an installed/source-archive package."""

import hashlib
from pathlib import Path
import subprocess


def source_revision(root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    if (root / ".git").exists():
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    digest = hashlib.sha256()
    suffixes = {".py", ".yaml", ".json", ".cu", ".cuh", ".h", ".cpp"}
    files = [p for directory in ("src", "experiments") for p in (root / directory).rglob("*")
             if p.is_file() and p.suffix in suffixes]
    if not files:
        raise ValueError("Cannot identify training source: no package files found")
    for path in sorted(files):
        digest.update(str(path.relative_to(root)).encode() + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return "archive-sha256:" + digest.hexdigest()
