from __future__ import annotations

import ast
import sys
from pathlib import Path
import re

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kern_compiler import compile_kern
from kern_transpiler import transpile


UNSUPPORTED_STMT = "# UNSUPPORTED:"
UNSUPPORTED_EXPR_RE = re.compile(r"<[A-Za-z_][A-Za-z0-9_]*>")
DATA_ROOT = ROOT / "data"
MAX_FILE_BYTES = 1_000_000
TEXT_FILE_SUFFIXES = {".py", ".kern", ".json", ".jsonl", ".txt", ".md"}


class ConvertRequest(BaseModel):
    code: str = Field(min_length=1)


class ConvertResponse(BaseModel):
    code: str


class DataFileInfo(BaseModel):
    path: str
    size_bytes: int


class DataFilesResponse(BaseModel):
    root: str
    files: list[DataFileInfo]


class FileContentResponse(BaseModel):
    path: str
    code: str


app = FastAPI(title="Kern Converter API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _safe_data_file(rel_path: str) -> Path:
    rel = Path(rel_path)
    if rel.is_absolute():
        raise HTTPException(status_code=400, detail="Absolute paths are not allowed.")

    candidate = (DATA_ROOT / rel).resolve()
    root_resolved = DATA_ROOT.resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Path escapes data directory.") from exc

    if not candidate.exists() or not candidate.is_file():
        raise HTTPException(status_code=404, detail="File not found.")
    if candidate.suffix.lower() not in TEXT_FILE_SUFFIXES:
        raise HTTPException(status_code=400, detail="Unsupported file type.")
    if candidate.stat().st_size > MAX_FILE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File is too large (>{MAX_FILE_BYTES} bytes).",
        )
    return candidate


@app.get("/api/files/list", response_model=DataFilesResponse)
def list_data_files() -> DataFilesResponse:
    if not DATA_ROOT.exists():
        return DataFilesResponse(root="data", files=[])

    files: list[DataFileInfo] = []
    for path in sorted(DATA_ROOT.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in TEXT_FILE_SUFFIXES:
            continue
        rel = path.relative_to(DATA_ROOT).as_posix()
        files.append(DataFileInfo(path=rel, size_bytes=path.stat().st_size))

    return DataFilesResponse(root="data", files=files)


@app.get("/api/files/content", response_model=FileContentResponse)
def read_data_file(path: str = Query(min_length=1)) -> FileContentResponse:
    file_path = _safe_data_file(path)
    try:
        content = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="File is not valid UTF-8 text.") from exc

    return FileContentResponse(path=file_path.relative_to(DATA_ROOT).as_posix(), code=content)


@app.post("/api/convert/python-to-kern", response_model=ConvertResponse)
def python_to_kern(payload: ConvertRequest) -> ConvertResponse:
    try:
        out = transpile(payload.code)
    except SyntaxError as exc:
        raise HTTPException(status_code=400, detail=f"Python syntax error: {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Transpile error: {exc}") from exc

    if UNSUPPORTED_STMT in out or UNSUPPORTED_EXPR_RE.search(out):
        raise HTTPException(
            status_code=400,
            detail="Conversion produced unsupported markers. Source uses syntax not yet supported by Kern transpiler.",
        )

    return ConvertResponse(code=out)


@app.post("/api/convert/kern-to-python", response_model=ConvertResponse)
def kern_to_python(payload: ConvertRequest) -> ConvertResponse:
    try:
        out = compile_kern(payload.code)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Compile error: {exc}") from exc

    try:
        ast.parse(out)
    except SyntaxError as exc:
        raise HTTPException(status_code=400, detail=f"Compiled Python is invalid: {exc}") from exc

    return ConvertResponse(code=out)
