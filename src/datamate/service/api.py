"""API FastAPI cho chế độ online.

    uv run --extra service uvicorn --factory datamate.service.api:create_app --port 8000

- GET  /health
- POST /ask         JSON {data_dir, question, as_of_date?, stream?}: data_dir phải nằm trong DATAMATE_DATA_ROOT
- POST /ask/upload  multipart: files (csv / xlsx / sqlite / README.md…), question, as_of_date?, stream?

Trả về answer, type, code đã chạy, kết quả verifier, lý do nếu abstain / unanswerable (xem service/engine.py).
`stream=true`: Server-Sent Events, mỗi bước một event (load, graph, precheck, understand, mechanism, verifier),
event cuối là `result` (hoặc `error`).

Biến môi trường: LLM_BASE_URL / LLM_MODEL / LLM_API_KEY (như chế độ thi), DATAMATE_DATA_ROOT (mặc định thư mục chạy),
DATAMATE_WORKERS (2), DATAMATE_CACHE_DIR, DATAMATE_RULEBOOK, DATAMATE_CALENDAR, DATAMATE_MAX_UPLOAD_MB (50).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from datamate.service.engine import DATA_SUFFIXES


class AskRequest(BaseModel):
    data_dir: str = Field(description="thư mục dữ liệu của shop, nằm trong DATAMATE_DATA_ROOT")
    question: str
    as_of_date: str | None = Field(None, description="YYYY-MM-DD; mặc định ngày cuối dữ liệu")
    stream: bool = False


def _sse(items):
    for it in items:
        yield f"event: {it['event']}\ndata: {json.dumps(it['data'], ensure_ascii=False)}\n\n"


def _data_root() -> Path:
    return Path(os.environ.get("DATAMATE_DATA_ROOT") or ".").resolve()


def _upload_root() -> Path:
    base = os.environ.get("DATAMATE_CACHE_DIR") or "outputs/service-cache"
    return Path(base).resolve() / "uploads"


def create_app(runner=None) -> FastAPI:
    """`runner`: đối tượng có ask(req) / stream(req) (mặc định ProcessRunner, worker spawn)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if getattr(app.state, "runner", None) is None:
            from datamate.service.workers import ProcessRunner

            app.state.runner = ProcessRunner(int(os.environ.get("DATAMATE_WORKERS", "2")))
            app.state.owns_runner = True
        yield
        if getattr(app.state, "owns_runner", False):
            app.state.runner.close()

    app = FastAPI(title="DataMate Agent", version="0.1.0", lifespan=lifespan)
    app.state.runner = runner

    def run(req: dict, stream: bool):
        if stream:
            return StreamingResponse(
                _sse(app.state.runner.stream(req)), media_type="text/event-stream"
            )
        res = app.state.runner.ask(req)
        if not res["ok"]:
            raise HTTPException(res.get("status", 500), res["error"])
        return res["result"]

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "model": os.environ.get("LLM_MODEL", ""),
            "llm_base_url": bool(os.environ.get("LLM_BASE_URL")),
        }

    @app.post("/ask")
    def ask(body: AskRequest):
        root = _data_root()
        p = (
            (root / body.data_dir).resolve()
            if not Path(body.data_dir).is_absolute()
            else Path(body.data_dir).resolve()
        )
        if not p.is_relative_to(root):
            raise HTTPException(403, f"data_dir phải nằm trong {root}")
        if not p.is_dir():
            raise HTTPException(400, f"không có thư mục {body.data_dir}")
        return run(
            {"data_dir": str(p), "question": body.question, "as_of_date": body.as_of_date},
            body.stream,
        )

    @app.post("/ask/upload")
    def ask_upload(
        files: Annotated[list[UploadFile], File()],
        question: Annotated[str, Form()],
        as_of_date: Annotated[str | None, Form()] = None,
        stream: Annotated[bool, Form()] = False,
        shop_name: Annotated[str, Form()] = "upload",
    ):
        limit = int(os.environ.get("DATAMATE_MAX_UPLOAD_MB", "50")) * 1024 * 1024
        blobs, total = [], 0
        for f in files:
            name = Path(f.filename or "").name
            if Path(name).suffix.lower() not in DATA_SUFFIXES:
                raise HTTPException(
                    400, f"không nhận file {name!r} (chỉ {', '.join(sorted(DATA_SUFFIXES))})"
                )
            data = f.file.read()
            total += len(data)
            if total > limit:
                raise HTTPException(413, f"tổng dung lượng vượt {limit // 2**20} MB")
            blobs.append((name, data))
        if not blobs:
            raise HTTPException(400, "chưa có file")
        h = hashlib.sha256()
        for name, data in sorted(blobs):
            h.update(name.encode() + b"\0" + hashlib.sha256(data).digest())
        # Cùng nội dung → cùng thư mục → dùng lại semantic graph đã cache.
        d = _upload_root() / h.hexdigest()[:16] / (re.sub(r"[^\w.-]+", "_", shop_name) or "upload")
        d.mkdir(parents=True, exist_ok=True)
        for name, data in blobs:
            if not (d / name).exists():
                (d / name).write_bytes(data)
        return run(
            {"data_dir": str(d), "question": question, "as_of_date": as_of_date or None}, stream
        )

    return app
