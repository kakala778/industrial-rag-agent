"""External-process boundary for the configured MinerU CLI."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Sequence

from .profiles import OCRMode, ParseProfile


class RunnerErrorType(str, Enum):
    """Failure categories produced before content-level quality checks."""

    INVALID_INPUT = "invalid_input"
    INVALID_OUTPUT = "invalid_output"
    EXECUTABLE_NOT_FOUND = "executable_not_found"
    PROCESS_START_FAILED = "process_start_failed"
    PROCESS_TIMEOUT = "process_timeout"
    PROCESS_FAILED = "process_failed"
    OUTPUT_INCOMPLETE = "output_incomplete"


@dataclass(frozen=True)
class RunResult:
    """Facts about one MinerU process invocation."""

    success: bool
    return_code: int | None
    output_dir: Path
    stdout: str
    stderr: str
    duration_seconds: float
    profile: ParseProfile
    error_type: RunnerErrorType | None = None
    error_message: str | None = None

    def to_dict(self, include_logs: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "success": self.success,
            "return_code": self.return_code,
            "output_dir": str(self.output_dir),
            "duration_seconds": self.duration_seconds,
            "profile": self.profile.to_dict(),
            "error_type": (
                self.error_type.value if self.error_type is not None else None
            ),
            "error_message": self.error_message,
        }
        if include_logs:
            result["stdout"] = self.stdout
            result["stderr"] = self.stderr
        return result


ProcessRunner = Callable[..., subprocess.CompletedProcess[str]]


class MinerURunner:
    """Run one configured MinerU CLI process without reading its content."""

    executable_env_var = "MINERU_KIT_PATH"

    def __init__(
        self,
        mineru_kit_path: str | Path | None = None,
        *,
        process_runner: ProcessRunner = subprocess.run,
    ) -> None:
        self._configured_path = (
            Path(mineru_kit_path) if mineru_kit_path is not None else None
        )
        self._process_runner = process_runner

    def build_command(
        self,
        input_pdf: str | Path,
        page_selector: str | int | None,
        output_dir: str | Path,
        profile: ParseProfile,
    ) -> list[str]:
        """Build a deterministic CLI command for one profile attempt."""

        executable = str(self._configured_path or "mineru-kit")
        command = [
            executable,
            "parse",
            str(Path(input_pdf)),
            "--output",
            str(Path(output_dir)),
            "--format",
            "zip",
            "--tier",
            profile.tier.value,
        ]
        if page_selector is not None:
            command.extend(["--pages", str(page_selector)])
        command.extend(["--ocr-mode", profile.ocr_mode.value])
        return command

    def run(
        self,
        input_pdf: str | Path,
        page_selector: str | int | None,
        output_dir: str | Path,
        profile: ParseProfile,
        timeout: float | None = 600.0,
    ) -> RunResult:
        """Execute MinerU and return explicit process/output status."""

        started = time.monotonic()
        input_path = Path(input_pdf)
        output_path = Path(output_dir)

        if not input_path.is_file():
            return self._failure(
                output_path,
                profile,
                RunnerErrorType.INVALID_INPUT,
                f"input PDF does not exist: {input_path}",
                started,
            )

        try:
            output_path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return self._failure(
                output_path,
                profile,
                RunnerErrorType.INVALID_OUTPUT,
                f"cannot create output directory: {exc}",
                started,
            )

        executable = self._resolve_executable()
        if executable is None:
            return self._failure(
                output_path,
                profile,
                RunnerErrorType.EXECUTABLE_NOT_FOUND,
                "MinerU CLI was not found; configure MINERU_KIT_PATH",
                started,
            )

        command = self.build_command(
            input_path, page_selector, output_path, profile
        )
        command[0] = executable
        previous_artifacts = _output_artifact_snapshot(output_path)
        try:
            completed = self._process_runner(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return self._failure(
                output_path,
                profile,
                RunnerErrorType.PROCESS_TIMEOUT,
                f"MinerU process exceeded timeout: {timeout}",
                started,
                stdout=_as_text(exc.stdout),
                stderr=_as_text(exc.stderr),
            )
        except FileNotFoundError as exc:
            return self._failure(
                output_path,
                profile,
                RunnerErrorType.EXECUTABLE_NOT_FOUND,
                str(exc),
                started,
            )
        except OSError as exc:
            return self._failure(
                output_path,
                profile,
                RunnerErrorType.PROCESS_START_FAILED,
                str(exc),
                started,
            )

        stdout = _as_text(completed.stdout)
        stderr = _as_text(completed.stderr)
        duration = time.monotonic() - started
        if completed.returncode != 0:
            return RunResult(
                success=False,
                return_code=completed.returncode,
                output_dir=output_path,
                stdout=stdout,
                stderr=stderr,
                duration_seconds=duration,
                profile=profile,
                error_type=RunnerErrorType.PROCESS_FAILED,
                error_message=(
                    f"MinerU exited with return code {completed.returncode}"
                ),
            )

        if not _has_new_output_artifacts(output_path, previous_artifacts):
            return RunResult(
                success=False,
                return_code=completed.returncode,
                output_dir=output_path,
                stdout=stdout,
                stderr=stderr,
                duration_seconds=duration,
                profile=profile,
                error_type=RunnerErrorType.OUTPUT_INCOMPLETE,
                error_message=(
                    "MinerU exited successfully but produced no new output artifact"
                ),
            )

        return RunResult(
            success=True,
            return_code=completed.returncode,
            output_dir=output_path,
            stdout=stdout,
            stderr=stderr,
            duration_seconds=duration,
            profile=profile,
        )

    def _resolve_executable(self) -> str | None:
        configured = self._configured_path
        if configured is None:
            environment_value = os.getenv(self.executable_env_var)
            configured = Path(environment_value) if environment_value else None

        if configured is not None:
            if configured.is_file():
                return str(configured)
            return shutil.which(str(configured))

        return shutil.which("mineru-kit") or shutil.which("mineru-kit.exe")

    @staticmethod
    def _failure(
        output_dir: Path,
        profile: ParseProfile,
        error_type: RunnerErrorType,
        error_message: str,
        started: float,
        *,
        stdout: str = "",
        stderr: str = "",
    ) -> RunResult:
        return RunResult(
            success=False,
            return_code=None,
            output_dir=output_dir,
            stdout=stdout,
            stderr=stderr,
            duration_seconds=time.monotonic() - started,
            profile=profile,
            error_type=error_type,
            error_message=error_message,
        )


def _output_artifact_snapshot(output_dir: Path) -> dict[str, tuple[int, int, int]]:
    if not output_dir.is_dir():
        return {}
    filenames = {"markdown.md", "structured_content.json", "middle_json.json"}
    snapshot = {}
    for path in output_dir.rglob("*"):
        if not path.is_file() or not (
            path.name in filenames or path.suffix.lower() == ".zip"
        ):
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        snapshot[path.relative_to(output_dir).as_posix()] = (
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        )
    return snapshot


def _has_new_output_artifacts(
    output_dir: Path,
    previous_artifacts: dict[str, tuple[int, int, int]],
) -> bool:
    current_artifacts = _output_artifact_snapshot(output_dir)
    return any(
        previous_artifacts.get(path) != signature
        for path, signature in current_artifacts.items()
    )


def _as_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value)


__all__ = ["MinerURunner", "RunResult", "RunnerErrorType"]
