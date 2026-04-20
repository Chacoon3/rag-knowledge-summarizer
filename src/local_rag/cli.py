from __future__ import annotations

import asyncio
import json
import logging

import typer
import uvicorn

from local_rag.service import RagService
from local_rag.store import KnowledgeBaseNotFoundError


app = typer.Typer(no_args_is_help=True, help="本地 RAG 知识库")


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )


@app.command()
def ingest(
    source_dir: str | None = typer.Option(
        None, "--source-dir", "-s", help="要入库的文档目录"
    ),
) -> None:
    service = RagService()
    try:
        stats = asyncio.run(service.ingest_async(source_dir))
    except (FileNotFoundError, ValueError) as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    typer.echo(
        f"已完成入库：{stats.manifest.document_count} 个文件，{stats.manifest.chunk_count} 个切片。"
    )
    typer.echo(f"索引目录：{stats.manifest.source_dir}")


@app.command()
def query(
    question: str = typer.Argument(..., help="要查询的问题"),
    top_k: int | None = typer.Option(None, "--top-k", min=1, help="返回的相关片段数量"),
) -> None:
    service = RagService()
    try:
        response = asyncio.run(service.query_async(question, top_k=top_k))
    except KnowledgeBaseNotFoundError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    typer.echo(response.answer)
    if response.matches:
        typer.echo("\n参考来源：")
        for match in response.matches:
            typer.echo(f"- {match.chunk.source_path} ({match.score:.3f})")


@app.command("manifest")
def show_manifest() -> None:
    service = RagService()
    try:
        manifest = asyncio.run(service.manifest_async())
    except KnowledgeBaseNotFoundError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    typer.echo(json.dumps(manifest.model_dump(), ensure_ascii=False, indent=2))


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="监听地址"),
    port: int = typer.Option(8000, "--port", min=1, max=65535, help="监听端口"),
) -> None:
    uvicorn.run("local_rag.api:app", host=host, port=port)


def main() -> None:
    configure_logging()
    app()
