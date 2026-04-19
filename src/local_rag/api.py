from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from local_rag import __version__
from local_rag.models import (
    ChunkPage,
    IngestStats,
    KnowledgeBaseManifest,
    QueryResponse,
)
from local_rag.service import RagService
from local_rag.settings import Settings
from local_rag.store import KnowledgeBaseNotFoundError


class IngestRequest(BaseModel):
    source_dir: str | None = None


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=20)


def create_app(
    settings: Settings | None = None,
    service: RagService | None = None,
) -> FastAPI:
    app = FastAPI(title="Local RAG Knowledge Base", version=__version__)
    service = service or RagService(settings=settings)
    web_dir = Path(str(files("local_rag").joinpath("web")))
    assets_dir = web_dir / "assets"

    app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(web_dir / "index.html")

    @app.get("/health")
    def health() -> dict[str, bool]:
        return {"status": True, "index_ready": service.store.is_ready()}

    @app.get("/manifest", response_model=KnowledgeBaseManifest)
    def get_manifest() -> KnowledgeBaseManifest:
        try:
            return service.manifest()
        except KnowledgeBaseNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/chunks", response_model=ChunkPage)
    def get_chunks(
        page: int = 1,
        page_size: int = 10,
    ) -> ChunkPage:
        try:
            return service.list_chunks(page=page, page_size=page_size)
        except KnowledgeBaseNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/ingest", response_model=IngestStats)
    def ingest(
        payload: IngestRequest = Body(default_factory=IngestRequest),
    ) -> IngestStats:
        try:
            return service.ingest(payload.source_dir)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/upload", response_model=IngestStats)
    async def upload(
        files: list[UploadFile] = File(..., description="上传的 PDF / 文档文件"),
        source_label: str = Form(default="uploaded://api"),
    ) -> IngestStats:
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

    @app.post("/query", response_model=QueryResponse)
    def query(payload: QueryRequest) -> QueryResponse:
        try:
            return service.query(payload.question, top_k=payload.top_k)
        except KnowledgeBaseNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    return app


def _build_upload_name(filename: str | None, index: int) -> str:
    candidate = Path(filename or f"upload-{index}.txt").name
    if not candidate or candidate in {".", ".."}:
        return f"upload-{index}.txt"
    return f"{index:03d}-{candidate}"


app = create_app()
