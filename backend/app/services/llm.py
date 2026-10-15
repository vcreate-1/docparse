"""Language-model clients for turning text into structured fields.

Order of preference (first one configured wins):
  1. FacilityClient - the internal RAG facility (OpenAI behind it). Upload a
     document once for a ref id, then prompt it page-scoped with a JSON schema
     at temperature 0. UNTESTED: written before the facility's API spec was
     available - every assumption about its request/response shape lives in
     this class, so adapting to the real spec is a one-place change.
  2. OpenAIClient   - OpenAI API directly (structured outputs, temperature 0),
     for local development when an API key is available.
Nothing configured -> text is still extracted exactly from the text layer; only
field extraction is unavailable, and the UI says so.

Every call is cached on disk by (client, model, prompt version, schema, text),
so batch reruns are deterministic and free.
"""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Protocol

from ..config import settings


class LlmClient(Protocol):
    name: str
    model: str

    def complete_json(self, system: str, user: str, schema: dict, *,
                      ref_id: str | None = None, pages: list[int] | None = None,
                      images: list[bytes] | None = None) -> dict: ...

    def supports_vision(self) -> bool: ...


class OpenAIClient:
    name = "openai"

    def __init__(self):
        from openai import OpenAI

        self.model = os.environ["DOCPARSE_OPENAI_MODEL"]  # no default: pick the model deliberately
        self._client = OpenAI()  # OPENAI_API_KEY from the environment

    @staticmethod
    def available() -> bool:
        return bool(os.environ.get("OPENAI_API_KEY") and os.environ.get("DOCPARSE_OPENAI_MODEL")) \
            and importlib.util.find_spec("openai") is not None

    def supports_vision(self) -> bool:
        return os.environ.get("DOCPARSE_OPENAI_VISION", "1") == "1"

    def complete_json(self, system, user, schema, *, ref_id=None, pages=None, images=None) -> dict:
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": _content(user, images)}],
            response_format={"type": "json_schema", "json_schema": {"name": "fields", "schema": schema, "strict": True}},
        )
        return json.loads(resp.choices[0].message.content)


class FacilityClient:
    """The internal RAG facility. UNTESTED - see module docstring.

    Assumed (to be confirmed against the facility's spec):
      POST {base}/documents            multipart file          -> {"ref_id": "..."}
      POST {base}/documents/{ref}/query
           {"prompt", "system", "pages": [..], "json_schema": {...}, "temperature": 0}
                                                               -> {"output": {...}} or the object itself
    Auth: "Authorization: Bearer {DOCPARSE_FACILITY_KEY}".
    """

    name = "facility"

    def __init__(self):
        import httpx

        self.base = os.environ["DOCPARSE_FACILITY_URL"].rstrip("/")
        self.model = os.environ.get("DOCPARSE_FACILITY_MODEL", "facility-default")
        self._http = httpx.Client(timeout=120, headers={"Authorization": f"Bearer {os.environ['DOCPARSE_FACILITY_KEY']}"})

    @staticmethod
    def available() -> bool:
        return bool(os.environ.get("DOCPARSE_FACILITY_URL") and os.environ.get("DOCPARSE_FACILITY_KEY"))

    def upload(self, pdf_path: Path, file_name: str) -> str:
        with open(pdf_path, "rb") as f:
            r = self._http.post(f"{self.base}/documents", files={"file": (file_name, f, "application/pdf")})
        r.raise_for_status()
        return r.json()["ref_id"]

    def supports_vision(self) -> bool:
        # Unknown until the spec is seen; opt in with DOCPARSE_FACILITY_VISION=1.
        return os.environ.get("DOCPARSE_FACILITY_VISION") == "1"

    def complete_json(self, system, user, schema, *, ref_id=None, pages=None, images=None) -> dict:
        if images:
            raise NotImplementedError("image input to the facility is not wired yet (needs its API spec)")
        if ref_id is None:
            raise ValueError("facility queries need the document's ref_id")
        r = self._http.post(f"{self.base}/documents/{ref_id}/query", json={
            "system": system, "prompt": user, "pages": pages or [], "json_schema": schema, "temperature": 0,
        })
        r.raise_for_status()
        body = r.json()
        return body.get("output", body)


class LocalOpenAIClient:
    """Any OpenAI-compatible server - LM Studio, Ollama, vLLM - for practice
    and offline development. Same request as OpenAIClient, local base URL:
        DOCPARSE_LLM_BASE_URL=http://localhost:1234/v1   (LM Studio's default)
        DOCPARSE_LLM_MODEL=qwen/qwen3.5-9b
    Last in the preference order, so it never shadows the facility or OpenAI."""

    name = "local"

    def __init__(self):
        import httpx

        self.base = os.environ["DOCPARSE_LLM_BASE_URL"].rstrip("/")
        self.model = os.environ["DOCPARSE_LLM_MODEL"]
        self._http = httpx.Client(timeout=300)

    @staticmethod
    def available() -> bool:
        return bool(os.environ.get("DOCPARSE_LLM_BASE_URL") and os.environ.get("DOCPARSE_LLM_MODEL"))

    def supports_vision(self) -> bool:
        """LM Studio says whether a model takes images (type "vlm"); others: DOCPARSE_LLM_VISION=1."""
        if "DOCPARSE_LLM_VISION" in os.environ:
            return os.environ["DOCPARSE_LLM_VISION"] == "1"
        try:
            root = self.base.rsplit("/v1", 1)[0]
            info = self._http.get(f"{root}/api/v0/models/{self.model}", timeout=5).json()
            return info.get("type") == "vlm"
        except Exception:
            return False

    def complete_json(self, system, user, schema, *, ref_id=None, pages=None, images=None) -> dict:
        r = self._http.post(f"{self.base}/chat/completions", json={
            "model": self.model,
            "temperature": 0,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": _content(user, images)}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "fields", "schema": schema, "strict": True}},
        })
        r.raise_for_status()
        msg = r.json()["choices"][0]["message"]
        # LM Studio's reasoning parser can file a reasoning model's whole answer
        # under reasoning_content and leave content empty; thinking text may
        # also surround the JSON. Take the JSON object from whichever has it.
        for content in (msg.get("content") or "", msg.get("reasoning_content") or ""):
            start, end = content.find("{"), content.rfind("}")
            if start != -1 and end > start:
                return json.loads(content[start:end + 1])
        raise ValueError(f"model returned no JSON (finish_reason={r.json()['choices'][0].get('finish_reason')})")


CLIENTS = (FacilityClient, OpenAIClient, LocalOpenAIClient)


def _content(text: str, images: list[bytes] | None):
    """OpenAI-style message content: plain text, or text + base64 images."""
    if not images:
        return text
    import base64

    parts = [{"type": "text", "text": text}]
    for png in images:
        parts.append({"type": "image_url",
                      "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}})
    return parts


def vision_client() -> LlmClient | None:
    """First configured client that can read images (charts)."""
    for cls in CLIENTS:
        if cls.available():
            c = cls()
            if c.supports_vision():
                return c
    return None


def client() -> LlmClient | None:
    for cls in CLIENTS:
        if cls.available():
            return cls()
    return None


def describe() -> dict:
    c = next((cls for cls in CLIENTS if cls.available()), None)
    model = os.environ.get("DOCPARSE_LLM_MODEL") if c is LocalOpenAIClient else None
    return {"configured": c is not None, "client": c.name if c else None, "model": model}


# ---------------------------------------------------------------- cache

def cached_call(c: LlmClient, prompt_version: str, system: str, user: str, schema: dict, **kw) -> tuple[dict, bool]:
    """(result, came_from_cache). Same inputs -> same answer, no second charge."""
    images = [hashlib.sha256(i).hexdigest() for i in kw.get("images") or []]
    key = hashlib.sha256(json.dumps(
        [c.name, c.model, prompt_version, system, user, schema, kw.get("pages"), images], sort_keys=True
    ).encode()).hexdigest()
    path = settings.data_dir / "llm_cache" / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text()), True
    result = c.complete_json(system, user, schema, **kw)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result))
    return result, False
