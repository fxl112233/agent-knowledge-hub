"""Fetch pinned public datasets and record local integrity metadata."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

BENCHMARK_ROOT = Path(__file__).resolve().parent
RAW_ROOT = BENCHMARK_ROOT / "data" / "raw"
PREPARED_ROOT = BENCHMARK_ROOT / "data" / "prepared"
MANIFEST_PATH = BENCHMARK_ROOT / "manifest.json"
LOCK_PATH = BENCHMARK_ROOT / "manifest.lock.json"


def load_manifest() -> dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run_git(*arguments: str, cwd: Path | None = None) -> str:
    process = subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return process.stdout.strip()


def _tree_integrity(root: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or ".git" in path.parts:
            continue
        relative = path.relative_to(root).as_posix()
        file_hash = sha256_file(path)
        size = path.stat().st_size
        digest.update(f"{relative}\0{size}\0{file_hash}\n".encode())
        file_count += 1
        total_bytes += size
    return digest.hexdigest(), file_count, total_bytes


def fetch_datasets(dataset: str = "all", *, force: bool = False) -> dict[str, Any]:
    manifest = load_manifest()
    definitions = manifest["datasets"]
    names = list(definitions) if dataset == "all" else [dataset]
    unknown = set(names) - set(definitions)
    if unknown:
        raise ValueError(f"unknown dataset(s): {', '.join(sorted(unknown))}")
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    lock: dict[str, Any] = {"schema_version": 1, "datasets": {}}
    if LOCK_PATH.exists():
        lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    for name in names:
        definition = definitions[name]
        destination = RAW_ROOT / name
        if force and destination.exists():
            resolved = destination.resolve()
            if resolved.parent != RAW_ROOT.resolve():
                raise RuntimeError(f"refusing to replace unsafe path: {resolved}")
            shutil.rmtree(resolved)
        if not destination.exists():
            _run_git(
                "clone", "--filter=blob:none", "--no-checkout", definition["repository"], str(destination)
            )
        revision = str(definition["revision"])
        _run_git("fetch", "--depth", "1", "origin", revision, cwd=destination)
        _run_git("checkout", "--detach", "--force", revision, cwd=destination)
        commit = _run_git("rev-parse", "HEAD", cwd=destination)
        if commit != revision:
            raise RuntimeError(f"{name} resolved to {commit}, expected {revision}")
        data_commit = ""
        if definition.get("data_repository"):
            data_destination = destination / "dataset"
            if not data_destination.exists():
                _run_git(
                    "clone",
                    "--filter=blob:none",
                    "--no-checkout",
                    str(definition.get("data_fetch_url") or definition["data_repository"]),
                    str(data_destination),
                )
            data_revision = str(definition["data_revision"])
            _run_git("fetch", "--depth", "1", "origin", data_revision, cwd=data_destination)
            _run_git("checkout", "--detach", "--force", data_revision, cwd=data_destination)
            data_commit = _run_git("rev-parse", "HEAD", cwd=data_destination)
            if data_commit != data_revision:
                raise RuntimeError(f"{name} data resolved to {data_commit}, expected {data_revision}")
        tree_hash, file_count, total_bytes = _tree_integrity(destination)
        lock["datasets"][name] = {
            "repository": definition["repository"],
            "revision": commit,
            "data_repository": definition.get("data_repository", ""),
            "data_revision": data_commit,
            "license": definition["license"],
            "sha256": tree_hash,
            "files": file_count,
            "bytes": total_bytes,
        }
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOCK_PATH.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return lock


def verify_datasets(dataset: str = "all") -> dict[str, bool]:
    if not LOCK_PATH.exists():
        raise FileNotFoundError("dataset lock is missing; run `akh-benchmark data fetch`")
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    names = list(lock["datasets"]) if dataset == "all" else [dataset]
    result: dict[str, bool] = {}
    for name in names:
        expected = lock["datasets"][name]
        root = RAW_ROOT / name
        if not root.exists():
            result[name] = False
            continue
        tree_hash, file_count, total_bytes = _tree_integrity(root)
        result[name] = (
            tree_hash == expected["sha256"]
            and file_count == expected["files"]
            and total_bytes == expected["bytes"]
        )
    return result
