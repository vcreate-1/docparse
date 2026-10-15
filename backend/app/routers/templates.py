import json
import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..services import fields
from ..storage import storage

router = APIRouter(prefix="/api/templates", tags=["templates"])


class Target(BaseModel):
    """One thing to extract. Page and bbox record where it was found in the
    sample; anchor_heading, header and page_keywords are what let the batch
    run find it again when the layout moves."""

    name: str = Field(min_length=1, max_length=120)
    kind: str
    method: str
    page: int
    bbox: list[float] = Field(min_length=4, max_length=4)
    instruction: str | None = None
    anchor_heading: str | None = None
    header: list[str] = []
    page_keywords: list[str] = []
    # Computed on save: what the field *is* (anchor, context, header, labels,
    # look-alikes, learned aliases) - how batch runs find it in other documents.
    fingerprint: dict | None = None
    # Text fields to pull out with the language model (name, type, description).
    fields: list[dict] = []
    rows: bool = False


class TemplateIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    source_doc_id: str
    targets: list[Target]


@router.get("")
def list_templates() -> list[dict]:
    items = [json.loads(p.read_text()) for p in (storage.root / "templates").glob("*.json")]
    return sorted(items, key=lambda t: t["updated_at"], reverse=True)


@router.get("/{template_id}")
def get_template(template_id: str) -> dict:
    try:
        path = storage.template_path(template_id)
    except KeyError:
        raise HTTPException(404, "Template not found")
    if not path.exists():
        raise HTTPException(404, "Template not found")
    return json.loads(path.read_text())


@router.post("")
def create_template(body: TemplateIn) -> dict:
    return _save(str(uuid.uuid4()), body, version=1)


@router.put("/{template_id}")
def update_template(template_id: str, body: TemplateIn) -> dict:
    # Every save bumps the version; batch results record which version
    # produced them, so output changes can be traced to template edits.
    current = get_template(template_id)
    return _save(template_id, body, version=current["version"] + 1)


def _save(template_id: str, body: TemplateIn, version: int) -> dict:
    data = body.model_dump()
    previous = {}
    path = storage.template_path(template_id)
    if path.exists():
        previous = {t["name"]: t.get("fingerprint") for t in json.loads(path.read_text())["targets"]}
    for t in data["targets"]:
        t["fingerprint"] = fields.fingerprint_target(body.source_doc_id, t, previous.get(t["name"]))
    record = {
        "id": template_id,
        "version": version,
        "updated_at": datetime.now(UTC).isoformat(),
        **data,
    }
    storage.template_path(template_id).write_text(json.dumps(record, indent=2))
    return record


class LocateIn(BaseModel):
    doc_id: str


@router.post("/{template_id}/locate")
def locate(template_id: str, body: LocateIn) -> dict:
    """Dry run: find every field of the template in another document."""
    template = get_template(template_id)
    if not storage.exists(body.doc_id):
        raise HTTPException(404, "Document not found")
    results = []
    for t in template["targets"]:
        fp = t.get("fingerprint") or fields.fingerprint_target(template["source_doc_id"], t)
        r = fields.locate_target(fp, body.doc_id, field_specs=t.get("fields") or None,
                                 instruction=t.get("instruction"), rows=t.get("rows", False))
        results.append({"name": t["name"], "kind": t["kind"], **r})
    return {"template_id": template_id, "version": template["version"], "doc_id": body.doc_id, "fields": results}


class ConfirmIn(BaseModel):
    doc_id: str
    page: int
    bbox: list[float] = Field(min_length=4, max_length=4)


@router.post("/{template_id}/fields/{index}/confirm")
def confirm(template_id: str, index: int, body: ConfirmIn) -> dict:
    """A person confirmed where a field is in another document; the template
    learns that document's heading as an alias (saved as a new version)."""
    template = get_template(template_id)
    if not 0 <= index < len(template["targets"]):
        raise HTTPException(404, "Field not found")
    target = template["targets"][index]
    fp = target.get("fingerprint") or fields.fingerprint_target(template["source_doc_id"], target)
    target["fingerprint"] = fields.confirm_target(fp, body.doc_id, body.page, body.bbox)
    template.update(version=template["version"] + 1, updated_at=datetime.now(UTC).isoformat())
    storage.template_path(template_id).write_text(json.dumps(template, indent=2))
    return template
