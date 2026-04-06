"""FastAPI + Jinja2 + HTMX web interface for audio-cleaner.

Provides pages for dataset management, model training, and inference.
Training and pipeline scripts run as background subprocesses with live log
streaming via Server-Sent Events (SSE).

Start the server::

    uv run uvicorn web_app:app --host 127.0.0.1 --port 8000 --reload
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

# dotenv MUST load before scripts.config so JINGLE_BASE_DIR is set in the
# environment before config.py resolves its paths at import time.
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, File, Form, Request, UploadFile  # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse  # noqa: E402
from fastapi.templating import Jinja2Templates  # noqa: E402
from loguru import logger  # noqa: E402

from core.file_manager import (  # noqa: E402
    dir_summary,
    list_checkpoints,
    list_dataset_tracks,
    list_inference_results,
    list_jingles_original,
    list_jingles_processed,
    list_music_clips,
)
from core.job_runner import JobRunner  # noqa: E402
from scripts.config import (  # noqa: E402
    BASE_DIR,
    INFERENCE_INPUT_FILE,
    INFERENCE_OUTPUT_DIR,
    INPUT_MUSIC_DIR,
    MODEL_OUTPUTS_DIR,
    ORIGINAL_JINGLES_DIR,
    OUTPUT_DATASET_DIR,
    PROCESSED_JINGLES_DIR,
)

# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialise shared resources on startup."""
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | {level} | {message}")
    logger.info("audio-cleaner web UI starting up.")
    yield
    logger.info("audio-cleaner web UI shutting down.")


app = FastAPI(title="audio-cleaner", lifespan=lifespan)
templates = Jinja2Templates(directory="templates")

# Single shared job runner — one job at a time across the whole app.
_runner = JobRunner()

_PROJECT_ROOT = Path(__file__).parent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def _assert_audio(filename: str) -> bool:
    return Path(filename).suffix.lower() in _AUDIO_EXTS


def _job_status_str() -> str:
    return _runner.status.name  # e.g. "IDLE", "RUNNING", "DONE", "ERROR"


def _uv_cmd() -> list[str]:
    """Return the uv executable path."""
    uv = shutil.which("uv")
    return [uv] if uv else ["uv"]


def _base_dir_ok() -> bool:
    """Return True if BASE_DIR exists and is a directory."""
    return BASE_DIR.is_dir()


def _not_configured_response(request: Request) -> HTMLResponse:
    """Return a page prompting the user to configure their data directory."""
    return templates.TemplateResponse(
        request,
        "not_configured.html",
        {"active_page": "", "base_dir": str(BASE_DIR)},
    )


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request) -> HTMLResponse:
    """Render the dashboard page."""
    if not _base_dir_ok():
        return _not_configured_response(request)
    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "active_page": "dashboard",
            "summary": dir_summary(),
            "checkpoints": list_checkpoints(),
            "job_status": _job_status_str(),
        },
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


@app.get("/dataset", response_class=HTMLResponse)
async def dataset_page(request: Request) -> HTMLResponse:
    """Render the dataset management page."""
    if not _base_dir_ok():
        return _not_configured_response(request)
    return templates.TemplateResponse(
        request,
        "dataset.html",
        {
            "active_page": "dataset",
            "music_clips": list_music_clips(),
            "jingles_original": list_jingles_original(),
            "jingles_processed": list_jingles_processed(),
            "tracks": list_dataset_tracks(),
            "job_status": _job_status_str(),
            "_music_dir": INPUT_MUSIC_DIR,
            "_jingle_dir": ORIGINAL_JINGLES_DIR,
            "_jingle_proc_dir": PROCESSED_JINGLES_DIR,
        },
    )


@app.post("/upload/music-clip", response_class=HTMLResponse)
async def upload_music_clip(request: Request, file: UploadFile = File(...)) -> HTMLResponse:
    """Upload a music clip WAV/FLAC file."""
    return await _save_audio_upload(file, INPUT_MUSIC_DIR, request, "music-list", list_music_clips)


@app.post("/upload/jingle", response_class=HTMLResponse)
async def upload_jingle(request: Request, file: UploadFile = File(...)) -> HTMLResponse:
    """Upload a raw jingle WAV/FLAC file."""
    return await _save_audio_upload(file, ORIGINAL_JINGLES_DIR, request, "jingle-list", list_jingles_original)


async def _save_audio_upload(
    upload: UploadFile,
    dest_dir: Path,
    request: Request,
    list_id: str,
    list_fn: object,
) -> HTMLResponse:
    if not upload.filename or not _assert_audio(upload.filename):
        return HTMLResponse(
            f'<div id="{list_id}"><p class="text-error">Unsupported file type.</p></div>',
            status_code=400,
        )
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / Path(upload.filename).name
    content = await upload.read()
    dest.write_bytes(content)
    logger.info("Uploaded {} ({} bytes) → {}", upload.filename, len(content), dest)
    files = list_fn()  # type: ignore[call-arg]
    return templates.TemplateResponse(
        request,
        "partials/file_list.html",
        {"files": files},
    )


# ---------------------------------------------------------------------------
# Pipeline scripts
# ---------------------------------------------------------------------------


@app.post("/pipeline/create-samples", response_class=HTMLResponse)
async def pipeline_create_samples() -> HTMLResponse:
    """Kick off create_samples.py in the background."""
    cmd = [*_uv_cmd(), "run", "python", "-m", "scripts.create_samples"]
    started = _runner.start(cmd, cwd=_PROJECT_ROOT)
    if not started:
        return HTMLResponse('<p class="text-error">A job is already running.</p>')
    return HTMLResponse('<p class="text-muted">create_samples started — see live log below.</p>')


@app.post("/pipeline/generate-dataset", response_class=HTMLResponse)
async def pipeline_generate_dataset() -> HTMLResponse:
    """Kick off generate_dataset.py in the background."""
    cmd = [*_uv_cmd(), "run", "python", "-m", "scripts.generate_dataset"]
    started = _runner.start(cmd, cwd=_PROJECT_ROOT)
    if not started:
        return HTMLResponse('<p class="text-error">A job is already running.</p>')
    return HTMLResponse('<p class="text-muted">generate_dataset started — see live log below.</p>')


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

_DEFAULT_TRAIN_CONFIG = {
    "epochs": 200,
    "batch_size": 4,
    "lr": "0.0002",
    "group_size": 4,
    "test_every": 999,
    "weights": "0.1,0.1,1.0,5.0",
    "name": "HIFI_HDEMUCS",
}


@app.get("/training", response_class=HTMLResponse)
async def training_page(request: Request) -> HTMLResponse:
    """Render the training page."""
    if not _base_dir_ok():
        return _not_configured_response(request)
    return templates.TemplateResponse(
        request,
        "training.html",
        {
            "active_page": "training",
            "config": _DEFAULT_TRAIN_CONFIG,
            "checkpoints": list_checkpoints(),
            "job_status": _job_status_str(),
        },
    )


@app.post("/training/start", response_class=HTMLResponse)
async def training_start(
    request: Request,
    epochs: int = Form(200),
    batch_size: int = Form(4),
    lr: str = Form("0.0002"),
    group_size: int = Form(4),
    test_every: int = Form(999),
    weights: str = Form("0.1,0.1,1.0,5.0"),
    name: str = Form(""),
) -> HTMLResponse:
    """Start a training run with the given hyperparameters."""
    weights_arg = f"[{weights}]"
    cmd = [
        *_uv_cmd(), "run", "dora", "--package", "demucs", "run",
        "model=hdemucs",
        "dset=musdb44",
        f"epochs={epochs}",
        f"++batch_size={batch_size}",
        "++misc.num_workers=0",
        f"++optim.lr={lr}",
        f'"++weights={weights_arg}"',
        f"++augment.remix.group_size={group_size}",
        "++augment.repitch.proba=0",
        f"++test.every={test_every}",
        "++dset.use_musdb=false",
        f"dset.wav={OUTPUT_DATASET_DIR}",
    ]
    if name:
        cmd.append(f"+name={name}")

    started = _runner.start(cmd, cwd=_PROJECT_ROOT)
    if not started:
        return templates.TemplateResponse(
            request,
            "partials/job_status.html",
            {"job_status": _job_status_str()},
        )
    logger.info("Training started: epochs={} batch={} lr={}", epochs, batch_size, lr)
    return templates.TemplateResponse(
        request,
        "partials/job_status.html",
        {"job_status": _job_status_str()},
    )


@app.post("/training/stop", response_class=HTMLResponse)
async def training_stop(request: Request) -> HTMLResponse:
    """Terminate the running training job."""
    _runner.stop()
    await asyncio.sleep(0.3)
    return templates.TemplateResponse(
        request,
        "partials/job_status.html",
        {"job_status": _job_status_str()},
    )


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


@app.get("/inference", response_class=HTMLResponse)
async def inference_page(request: Request) -> HTMLResponse:
    """Render the inference page."""
    if not _base_dir_ok():
        return _not_configured_response(request)
    return templates.TemplateResponse(
        request,
        "inference.html",
        {
            "active_page": "inference",
            "checkpoints": list_checkpoints(),
            "results": list_inference_results(),
            "job_status": _job_status_str(),
        },
    )


@app.post("/inference/run", response_class=HTMLResponse)
async def inference_run(
    request: Request,
    file: UploadFile = File(...),
    checkpoint: str = Form(""),
) -> HTMLResponse:
    """Upload audio and start a separation job."""
    if not file.filename or not _assert_audio(file.filename):
        return HTMLResponse('<p class="text-error">Unsupported file type.</p>', status_code=400)

    # Save uploaded file as the inference input
    INFERENCE_INPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    content = await file.read()
    INFERENCE_INPUT_FILE.write_bytes(content)
    logger.info("Inference input saved: {} ({} bytes)", INFERENCE_INPUT_FILE, len(content))

    env = {**os.environ}
    cmd_parts = [*_uv_cmd(), "run", "python", "-m", "scripts.separate_audio"]
    if checkpoint:
        env["DEMUCS_MODEL_FOLDER"] = checkpoint

    started = _runner.start(cmd_parts, cwd=_PROJECT_ROOT, env=env)
    if not started:
        return templates.TemplateResponse(
            request,
            "partials/job_status.html",
            {"job_status": _job_status_str()},
        )
    return templates.TemplateResponse(
        request,
        "partials/job_status.html",
        {"job_status": _job_status_str()},
    )


@app.get("/inference/download/{filename}")
async def inference_download(filename: str) -> FileResponse:
    """Download an inference result stem file."""
    safe_name = Path(filename).name  # prevent path traversal
    path = INFERENCE_OUTPUT_DIR / safe_name
    if not path.exists():
        return JSONResponse({"error": "File not found"}, status_code=404)
    return FileResponse(path, media_type="audio/wav", filename=safe_name)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request) -> HTMLResponse:
    """Render the settings page."""
    return templates.TemplateResponse(
        request,
        "settings.html",
        {
            "active_page": "settings",
            "base_dir": str(BASE_DIR),
            "paths": {
                "music_clips":        str(INPUT_MUSIC_DIR),
                "jingles_original":   str(ORIGINAL_JINGLES_DIR),
                "jingles_processed":  str(PROCESSED_JINGLES_DIR),
                "training_dataset":   str(OUTPUT_DATASET_DIR),
                "test_audio":         str(INFERENCE_INPUT_FILE.parent),
                "inference_results":  str(INFERENCE_OUTPUT_DIR),
                "checkpoints":        str(MODEL_OUTPUTS_DIR),
            },
        },
    )


@app.post("/settings/save", response_class=HTMLResponse)
async def settings_save(base_dir: str = Form(...)) -> HTMLResponse:
    """Persist JINGLE_BASE_DIR to .env.

    With ``uvicorn --reload`` the server reloads automatically when .env changes.
    Without --reload the user must restart the server manually.
    """
    env_file = _PROJECT_ROOT / ".env"
    lines: list[str] = []
    if env_file.exists():
        lines = [ln for ln in env_file.read_text().splitlines() if not ln.startswith("JINGLE_BASE_DIR")]
    lines.append(f"JINGLE_BASE_DIR={base_dir}")
    env_file.write_text("\n".join(lines) + "\n")
    os.environ["JINGLE_BASE_DIR"] = base_dir
    logger.info("JINGLE_BASE_DIR saved: {}", base_dir)

    # Touch web_app.py so uvicorn --reload picks up the change immediately.
    _PROJECT_ROOT.joinpath("web_app.py").touch()

    return HTMLResponse(
        '<div id="settings-msg">'
        '<p class="text-success mt-1">✓ Saved. Server is reloading…</p>'
        '<p class="text-muted mt-1" style="font-size:0.8rem;">'
        "Page will refresh in a moment — or reload manually if nothing happens.</p>"
        '<script>setTimeout(() => location.href = "/", 2000);</script>'
        "</div>"
    )


# ---------------------------------------------------------------------------
# Shared SSE log stream
# ---------------------------------------------------------------------------


@app.get("/api/stream-log")
async def stream_log() -> StreamingResponse:
    """SSE endpoint — streams job log lines as they arrive.

    Keeps the connection open permanently so HTMX does not reconnect in a loop.
    When no job is running it sends a keepalive comment every 15 seconds.
    When a job starts it streams lines in real time; after the job ends it
    continues waiting for the next job.
    """

    async def event_generator() -> AsyncGenerator[str, None]:
        yielded = 0
        while True:
            snapshot = _runner.get_log()
            while yielded < len(snapshot):
                line = snapshot[yielded]
                safe = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                yield f"data: <span>{safe}</span>\n\n"
                yielded += 1

            if _runner.is_running:
                # Job in progress — tight poll to stream output quickly.
                await asyncio.sleep(0.2)
            else:
                # Idle — send a keepalive SSE comment every 15 s so the
                # browser does not time out and HTMX does not reconnect.
                await asyncio.sleep(15)
                yield ": keepalive\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/job-status", response_class=HTMLResponse)
async def api_job_status(request: Request) -> HTMLResponse:
    """HTMX polling target — returns the job status badge fragment."""
    return templates.TemplateResponse(
        request,
        "partials/job_status.html",
        {"job_status": _job_status_str()},
    )
