from __future__ import annotations

import os
from importlib.resources import files
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import (
    APIRouter,
    Body,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from local_rag import __version__
from local_rag.models import (
    ChunkPage,
    DeleteChunkResponse,
    IngestStats,
    KnowledgeBaseManifest,
    LlmProviderUpdateRequest,
    LlmStatusResponse,
    QueryResponse,
)
from local_rag.service import RagService
from local_rag.settings import Settings
from local_rag.store import KnowledgeBaseNotFoundError


os.environ["ANONYMIZED_TELEMETRY"] = "False"

WEB_DIR = Path(str(files("local_rag").joinpath("web")))
ASSETS_DIR = WEB_DIR / "assets"
router = APIRouter()


class IngestRequest(BaseModel):
    source_dir: str | None = None


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=20)


def get_service(request: Request) -> RagService:
    return request.app.state.service


@router.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@router.get("/health")
def health(request: Request) -> dict[str, bool]:
    service = get_service(request)
    return {"status": True, "index_ready": service.store.is_ready()}


@router.get("/llm/status", response_model=LlmStatusResponse)
def get_llm_status(request: Request) -> LlmStatusResponse:
    service = get_service(request)
    return service.get_llm_status()


@router.put("/llm/provider", response_model=LlmStatusResponse)
def update_llm_provider(
    payload: LlmProviderUpdateRequest,
    request: Request,
) -> LlmStatusResponse:
    service = get_service(request)
    try:
        return service.set_generation_provider(payload.provider)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/manifest", response_model=KnowledgeBaseManifest)
def get_manifest(request: Request) -> KnowledgeBaseManifest:
    service = get_service(request)
    try:
        return service.manifest()
    except KnowledgeBaseNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/chunks", response_model=ChunkPage)
def get_chunks(
    request: Request,
    page: int = 1,
    page_size: int = 10,
) -> ChunkPage:
    service = get_service(request)
    try:
        return service.list_chunks(page=page, page_size=page_size)
    except KnowledgeBaseNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/chunks/{chunk_id}", response_model=DeleteChunkResponse)
def delete_chunk(chunk_id: str, request: Request) -> DeleteChunkResponse:
    service = get_service(request)
    try:
        return service.delete_chunk(chunk_id)
    except KnowledgeBaseNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        message = str(exc)
        status_code = 404 if message.startswith("未找到 chunk") else 400
        raise HTTPException(status_code=status_code, detail=message) from exc


@router.post("/ingest", response_model=IngestStats)
def ingest(
    request: Request,
    payload: IngestRequest = Body(default_factory=IngestRequest),
) -> IngestStats:
    service = get_service(request)
    try:
        return service.ingest(payload.source_dir)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/upload", response_model=IngestStats)
async def upload(
    request: Request,
    files: list[UploadFile] = File(..., description="上传的 PDF / 文档文件"),
    source_label: str = Form(default="uploaded://api"),
) -> IngestStats:
    service = get_service(request)
    if not files:
        raise HTTPException(status_code=400, detail="至少上传一个文件。")

    with TemporaryDirectory(prefix="local-rag-upload-") as temp_dir:
        temp_root = Path(temp_dir)
        saved_files: list[Path] = []

        try:
            for index, uploaded in enumerate(files, start=1):
                filename = _build_upload_name(uploaded.filename, index)
                target_path = temp_root / filename
                target_path.write_bytes(await uploaded.read())
                saved_files.append(target_path)

            return service.ingest_files(saved_files, source_label=source_label)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            for uploaded in files:
                await uploaded.close()


@router.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest, request: Request) -> QueryResponse:
    service = get_service(request)
    try:
        return service.query(payload.question, top_k=payload.top_k)
    except KnowledgeBaseNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def create_app(
    settings: Settings | None = None,
    service: RagService | None = None,
) -> FastAPI:
    app = FastAPI(title="Local RAG Knowledge Base", version=__version__)
    app.state.service = service or RagService(settings=settings)
    app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")
    app.include_router(router)

    return app


def _build_upload_name(filename: str | None, index: int) -> str:
    candidate = Path(filename or f"upload-{index}.txt").name
    if not candidate or candidate in {".", ".."}:
        return f"upload-{index}.txt"
    return f"{index:03d}-{candidate}"


app = create_app()
