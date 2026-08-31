"""Managed binary assets used by multimodal parsing and embedding."""

from __future__ import annotations

import hashlib
import io
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from config import settings


class AssetStore:
    """Store normalized images under a document-scoped, content-addressed path."""

    def __init__(self, root: str | None = None) -> None:
        self.root = Path(root or settings.asset_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def save_image(self, doc_id: str, image: Any) -> str:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="PNG", optimize=True)
        payload = buffer.getvalue()
        digest = hashlib.sha256(payload).hexdigest()
        directory = self._doc_dir(doc_id)
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / f"{digest}.png"
        if not destination.exists():
            destination.write_bytes(payload)
        return str(destination)

    def prune_document(self, doc_id: str, keep_paths: Iterable[str] = ()) -> int:
        directory = self._doc_dir(doc_id)
        if not directory.exists():
            return 0
        keep = {Path(value).resolve() for value in keep_paths if value}
        removed = 0
        for path in directory.iterdir():
            if path.is_file() and path.resolve() not in keep:
                path.unlink()
                removed += 1
        if not any(directory.iterdir()):
            directory.rmdir()
        return removed

    def delete_document(self, doc_id: str) -> int:
        directory = self._doc_dir(doc_id)
        if not directory.exists():
            return 0
        count = sum(1 for path in directory.rglob("*") if path.is_file())
        shutil.rmtree(directory)
        return count

    def is_managed(self, value: str) -> bool:
        try:
            Path(value).resolve().relative_to(self.root)
            return True
        except (OSError, ValueError):
            return False

    def _doc_dir(self, doc_id: str) -> Path:
        value = str(doc_id).strip()
        if not value:
            raise ValueError("invalid document id for asset storage")
        safe = "".join(character for character in value if character.isalnum() or character in "-_")
        directory_name = (
            safe if safe == value else f"external-{hashlib.sha256(value.encode('utf-8')).hexdigest()}"
        )
        directory = (self.root / directory_name).resolve()
        directory.relative_to(self.root)
        return directory
