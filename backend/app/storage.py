"""Document storage.

Everything outside this module talks to `storage` by sha256, never by path, so
the local folder can later be replaced by a Snowflake stage (PUT/GET +
GET_PRESIGNED_URL) without touching the routers or services.

Layout on disk:
    data/docs/<sha256>/original.pdf
    data/docs/<sha256>/meta.json      -> stand-in for the Snowflake `documents` row
    data/docs/<sha256>/pages.json     -> page text index used by search
    data/docs/<sha256>/regions/<n>.json -> per-page detection cache
    data/templates/<id>.json
"""

import hashlib
import json
from pathlib import Path
from typing import Any

from .config import settings


class LocalStorage:
    def __init__(self, root: Path):
        self.root = root
        (root / "docs").mkdir(parents=True, exist_ok=True)
        (root / "templates").mkdir(parents=True, exist_ok=True)

    @staticmethod
    def sha256(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def doc_dir(self, doc_id: str) -> Path:
        # doc_id is a hex digest; reject anything else so it can't escape root.
        if len(doc_id) != 64 or any(c not in "0123456789abcdef" for c in doc_id):
            raise KeyError(doc_id)
        return self.root / "docs" / doc_id

    def exists(self, doc_id: str) -> bool:
        return (self.doc_dir(doc_id) / "meta.json").exists()

    def put_pdf(self, doc_id: str, data: bytes) -> Path:
        d = self.doc_dir(doc_id)
        d.mkdir(parents=True, exist_ok=True)
        path = d / "original.pdf"
        path.write_bytes(data)
        return path

    def pdf_path(self, doc_id: str) -> Path:
        path = self.doc_dir(doc_id) / "original.pdf"
        if not path.exists():
            raise KeyError(doc_id)
        return path

    def read_json(self, doc_id: str, name: str) -> Any | None:
        path = self.doc_dir(doc_id) / name
        return json.loads(path.read_text()) if path.exists() else None

    def write_json(self, doc_id: str, name: str, value: Any) -> None:
        path = self.doc_dir(doc_id) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(value))
        tmp.replace(path)  # atomic, so a concurrent reader never sees half a file

    def list_docs(self) -> list[dict]:
        metas = []
        for meta in (self.root / "docs").glob("*/meta.json"):
            metas.append(json.loads(meta.read_text()))
        return sorted(metas, key=lambda m: m["uploaded_at"], reverse=True)

    def template_path(self, template_id: str) -> Path:
        if not template_id.replace("-", "").isalnum():
            raise KeyError(template_id)
        return self.root / "templates" / f"{template_id}.json"


storage = LocalStorage(settings.data_dir)
