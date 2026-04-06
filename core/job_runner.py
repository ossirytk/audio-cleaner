"""Background subprocess runner with SSE log streaming.

Only one job can run at a time. Callers receive live output via the
async generator :func:`stream_log`.

Usage::

    runner = JobRunner()
    ok = runner.start(["uv", "run", "dora", ...], cwd=Path("."))
    async for line in runner.stream_log():
        ...
"""

from __future__ import annotations

import asyncio
import contextlib
import subprocess
import threading
from collections import deque
from collections.abc import AsyncGenerator
from enum import Enum, auto
from pathlib import Path

from loguru import logger


class JobStatus(Enum):
    """Lifecycle state of the background job."""

    IDLE = auto()
    RUNNING = auto()
    DONE = auto()
    ERROR = auto()


_LOG_BUFFER_SIZE = 2000


class JobRunner:
    """Runs a single subprocess at a time and streams its output."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._status: JobStatus = JobStatus.IDLE
        self._log: deque[str] = deque(maxlen=_LOG_BUFFER_SIZE)
        self._process: subprocess.Popen | None = None  # type: ignore[type-arg]
        self._new_line = asyncio.Event()
        self._loop: asyncio.AbstractEventLoop | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def status(self) -> JobStatus:
        """Current job status."""
        return self._status

    @property
    def is_running(self) -> bool:
        """True while a subprocess is active."""
        return self._status is JobStatus.RUNNING

    def start(self, cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> bool:
        """Spawn the subprocess in a background thread.

        Returns False if a job is already running.
        """
        with self._lock:
            if self._status is JobStatus.RUNNING:
                return False
            self._log.clear()
            self._status = JobStatus.RUNNING
            self._loop = asyncio.get_event_loop()

        thread = threading.Thread(target=self._run, args=(cmd, cwd, env), daemon=True)
        thread.start()
        logger.info("Job started: {}", " ".join(cmd))
        return True

    def stop(self) -> None:
        """Terminate the running subprocess if any."""
        with self._lock:
            proc = self._process
        if proc and proc.poll() is None:
            proc.terminate()
            logger.info("Job terminated by user request.")

    def get_log(self) -> list[str]:
        """Return the full buffered log as a list of lines."""
        return list(self._log)

    async def stream_log(self, from_line: int = 0) -> AsyncGenerator[str, None]:
        """Async generator that yields log lines as they arrive.

        Yields lines already in the buffer starting from *from_line*, then
        waits for new lines until the job finishes.
        """
        yielded = from_line
        while True:
            snapshot = list(self._log)
            while yielded < len(snapshot):
                yield snapshot[yielded]
                yielded += 1

            if self._status is not JobStatus.RUNNING:
                break

            # Wait for the background thread to signal a new line (with timeout
            # so we don't stall if the event is never set after job ends).
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(
                    asyncio.shield(asyncio.get_event_loop().run_in_executor(None, self._new_line.wait)),
                    timeout=1.0,
                )
            self._new_line.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self, cmd: list[str], cwd: Path | None, env: dict[str, str] | None) -> None:
        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=cwd,
                env=env,
            )
            assert self._process.stdout is not None  # noqa: S101
            for raw in self._process.stdout:
                line = raw.rstrip("\n")
                self._log.append(line)
                self._new_line.set()

            self._process.wait()
            rc = self._process.returncode
            with self._lock:
                self._status = JobStatus.DONE if rc == 0 else JobStatus.ERROR
            self._log.append(f"[exit code {rc}]")
            self._new_line.set()
            logger.info("Job finished with exit code {}.", rc)
        except Exception as exc:
            logger.exception("Job runner error: {}", exc)
            with self._lock:
                self._status = JobStatus.ERROR
            self._log.append(f"[runner error: {exc}]")
            self._new_line.set()
