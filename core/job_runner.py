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
        self._total_appended: int = 0  # monotonic counter; never decrements
        self._process: subprocess.Popen | None = None  # type: ignore[type-arg]
        # threading.Event is thread-safe and can be waited on in run_in_executor.
        self._new_line = threading.Event()

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
            self._total_appended = 0
            self._status = JobStatus.RUNNING

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

    def get_log_cursor(self) -> tuple[list[str], int]:
        """Return ``(log_snapshot, total_appended)`` for cursor-based SSE streaming.

        ``total_appended`` is a monotonically increasing counter that is never
        reset (even when the deque evicts old lines), so consumers can detect
        how many new lines have arrived since their last poll regardless of
        eviction.
        """
        with self._lock:
            return list(self._log), self._total_appended

    async def stream_log(self, from_line: int = 0) -> AsyncGenerator[str, None]:
        """Async generator that yields log lines as they arrive.

        Yields lines already in the buffer starting from *from_line*, then
        waits for new lines until the job finishes.
        """
        yielded = from_line
        loop = asyncio.get_event_loop()
        while True:
            snapshot = list(self._log)
            while yielded < len(snapshot):
                yield snapshot[yielded]
                yielded += 1

            if self._status is not JobStatus.RUNNING:
                break

            # Clear before waiting to avoid missing a signal that arrives
            # between the snapshot check and this point; re-check log above.
            self._new_line.clear()
            # threading.Event.wait() is a blocking function — safe to run in
            # executor.  Pass a timeout so we re-check status if no line arrives.
            await loop.run_in_executor(None, self._new_line.wait, 1.0)

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
                with self._lock:
                    self._log.append(line)
                    self._total_appended += 1
                self._new_line.set()

            self._process.wait()
            rc = self._process.returncode
            with self._lock:
                self._status = JobStatus.DONE if rc == 0 else JobStatus.ERROR
                self._log.append(f"[exit code {rc}]")
                self._total_appended += 1
            self._new_line.set()
            logger.info("Job finished with exit code {}.", rc)
        except Exception as exc:
            logger.exception("Job runner error: {}", exc)
            with self._lock:
                self._status = JobStatus.ERROR
                self._log.append(f"[runner error: {exc}]")
                self._total_appended += 1
            self._new_line.set()
