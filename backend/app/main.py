from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .routers import documents, templates
import threading

from .services import llm
from .services.detectors import list_detectors, warm_up

app = FastAPI(title="docparse")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Accept-Ranges", "Content-Range", "Content-Length"],  # PDF.js range loading
)
app.include_router(documents.router)


@app.on_event("startup")
def _warm_models() -> None:
    threading.Thread(target=warm_up, name="model-warm-up", daemon=True).start()

app.include_router(templates.router)


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.get("/api/detectors")
def detectors() -> list[dict]:
    return list_detectors()


@app.get("/api/llm")
def llm_status() -> dict:
    """Whether field extraction from text is available (facility or OpenAI)."""
    return llm.describe()
