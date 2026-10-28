"""Durable, device-independent state for operator-paced model evaluation.

No model, camera or robot is started by this module.  The caller owns readiness
and execution; only explicit method calls change the persisted trial state.
"""

from __future__ import annotations

import copy
import json
import math
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


NORMAL_END_REASONS = frozenset(("step_limit", "early_finish"))
SCORABLE_END_REASONS = NORMAL_END_REASONS | {"time_limit"}
END_REASONS = SCORABLE_END_REASONS | {"paused", "error", "interrupted"}


def export_eligible(attempt: dict[str, Any]) -> bool:
    """Eligibility is independent of counting, which happens after export."""
    score = attempt.get("score")
    steps = attempt.get("steps_sent")
    return (attempt.get("status") == "completed"
            and attempt.get("end_reason") in SCORABLE_END_REASONS
            and attempt.get("recording_complete") is True
            and isinstance(score, (int, float)) and not isinstance(score, bool)
            and math.isfinite(score) and 0 <= score <= 100
            and isinstance(steps, int) and not isinstance(steps, bool) and steps > 0
            and (attempt.get("end_reason") != "step_limit" or steps == attempt.get("steps")))


def is_counted(attempt: dict[str, Any]) -> bool:
    return (export_eligible(attempt)
            and attempt.get("export", {}).get("status") == "succeeded"
            and bool(attempt.get("export", {}).get("path")))


class SessionError(ValueError):
    """An invalid operator transition; state remains unchanged."""


class SessionLocked(SessionError):
    """Another process or Session instance already owns this session."""


def select_task_model_ids(task: str, model_ids=None) -> list[str]:
    """Preserve the explicit operator order for any named task."""
    if not isinstance(task, str) or not task.strip():
        raise SessionError("A nonempty task identifier is required")
    if (not isinstance(model_ids, list) or not model_ids
            or any(not isinstance(model, str) or not model.strip() for model in model_ids)
            or len(set(model_ids)) != len(model_ids)):
        raise SessionError("Select a nonempty list of unique model IDs")
    return list(model_ids)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SessionError(f"{label} must be a positive integer")
    return value


def _json_copy(value: Any) -> Any:
    """Reject non-durable values before any state mutation."""
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise SessionError(f"State metadata must be finite JSON values: {exc}") from exc


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, allow_nan=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        if os.name != "nt":
            directory_fd = os.open(str(path.parent), os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()  # Only our reconstructable, uncommitted temp file.


class _WriterLock:
    def __init__(self, path: Path):
        self.handle = path.open("a+b")
        self.handle.seek(0, os.SEEK_END)
        if self.handle.tell() == 0:
            self.handle.write(b"\0")
            self.handle.flush()
        self.handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.handle.close()
            self.handle = None
            raise SessionLocked(f"Session is already open: {path.parent}") from exc

    def close(self) -> None:
        if self.handle is None:
            return
        self.handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        finally:
            self.handle.close()
            self.handle = None


class Session:
    """One locked writer, with atomic persisted transitions and no device effects.

    Keep the Session open throughout interactive use. Opening recovers an
    unfinished attempt as interrupted, so a crashed process never creates a
    completed trial. Read-only displays can consume ``session.json`` directly.
    """

    def __init__(self, root: Path, lock: _WriterLock, state: dict[str, Any]):
        self.root = root
        self._lock = lock
        self._state = state
        self._closed = False

    @classmethod
    def create(cls, path: str | Path, task: str, model_ids=None,
               repeats: int = 25, steps: int = 800, mode: str = "live") -> "Session":
        if mode not in ("live", "demo"):
            raise SessionError("mode must be live or demo")
        _positive_int(repeats, "repeats")
        _positive_int(steps, "steps")
        models = select_task_model_ids(task, model_ids)
        root = Path(path).resolve()
        root.mkdir(parents=True, exist_ok=True)
        lock = _WriterLock(root / ".session.lock")
        try:
            if (root / "session.json").exists() or (root / "attempts").exists():
                raise SessionError(f"Existing session/attempt records must not be overwritten: {root}")
            timestamp = _now()
            state = {
                "version": 1, "id": uuid.uuid4().hex, "task": task, "mode": mode,
                "created_at": timestamp, "updated_at": timestamp,
                "steps": steps, "default_repeats": repeats,
                "model_ids": models, "targets": {model: repeats for model in models},
                "current_index": 0, "attempts": [],
            }
            _atomic_json(root / "session.json", state)
            return cls(root, lock, state)
        except BaseException:
            lock.close()
            raise

    @classmethod
    def open(cls, path: str | Path) -> "Session":
        root = Path(path).resolve()
        if not (root / "session.json").is_file():
            raise SessionError(f"No session.json at {root}")
        lock = _WriterLock(root / ".session.lock")
        try:
            state = json.loads((root / "session.json").read_text(encoding="utf-8"))
            if state.get("version") != 1:
                raise SessionError("Unsupported session version")
            session = cls(root, lock, state)
            session._recover()
            return session
        except BaseException:
            lock.close()
            raise

    def __enter__(self) -> "Session":
        self._ensure_open()
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def close(self) -> None:
        if not self._closed:
            self._lock.close()
            self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise SessionError("Session is closed")

    def _commit(self, candidate: dict[str, Any]) -> None:
        self._ensure_open()
        candidate["updated_at"] = _now()
        _atomic_json(self.root / "session.json", candidate)
        self._state = candidate

    def _recover(self) -> None:
        candidate = copy.deepcopy(self._state)
        changed = False
        for attempt in candidate["attempts"]:
            if attempt["status"] == "running":
                attempt.update(status="interrupted", end_reason="interrupted",
                               finished_at=_now(), recording_complete=False,
                               counted=False, recovery_note="Writer exited before recording completion")
                changed = True
            if attempt["export"]["status"] == "in_progress":
                attempt["export"].update(status="failed", updated_at=_now(),
                                         error="Export was interrupted; inspect and retry the existing recording")
                changed = True
        if changed:
            self._commit(candidate)

    @property
    def state(self) -> dict[str, Any]:
        return self.snapshot()

    @property
    def current_model(self) -> str | None:
        index = self._state["current_index"]
        models = self._state["model_ids"]
        return models[index] if index < len(models) else None

    def snapshot(self) -> dict[str, Any]:
        result = copy.deepcopy(self._state)
        # Old versions counted before export. Correct read views without silently
        # rewriting historical session files; expose discrepancies for review.
        result["counting_corrections"] = []
        for attempt in result["attempts"]:
            effective = is_counted(attempt)
            if attempt.get("counted") != effective:
                result["counting_corrections"].append({
                    "attempt_id": attempt["id"], "stored": attempt.get("counted"),
                    "effective": effective, "export_status": attempt["export"]["status"]})
            attempt["counted"] = effective
            attempt["export_eligible"] = export_eligible(attempt)
        result["current_model"] = self.current_model
        result["complete"] = self.current_model is None
        result["progress"] = {
            model: {"completed": self._completed(model),
                    "target": self._state["targets"][model],
                    "attempts": sum(a["model_id"] == model for a in self._state["attempts"])}
            for model in self._state["model_ids"]
        }
        return result

    def _completed(self, model: str) -> int:
        return sum(a["model_id"] == model and is_counted(a) for a in self._state["attempts"])

    def _ensure_no_pending_trial(self) -> None:
        if any(a["status"] in ("running", "awaiting_score") for a in self._state["attempts"]):
            raise SessionError("Finish and score the pending trial before this action")

    @staticmethod
    def _attempt(state: dict[str, Any], attempt_id: str) -> dict[str, Any]:
        for attempt in state["attempts"]:
            if attempt["id"] == attempt_id:
                return attempt
        raise SessionError(f"Unknown attempt: {attempt_id}")

    def begin_attempt(self, *, steps: int | None = None, instruction: str | None = None,
                      contract: dict[str, Any] | None = None) -> dict[str, Any]:
        self._ensure_open()
        self._ensure_no_pending_trial()
        model = self.current_model
        if model is None:
            raise SessionError("Evaluation is complete")
        if self._completed(model) >= self._state["targets"][model]:
            raise SessionError("Current model reached its target; advance or increase target first")
        limit = self._state["steps"] if steps is None else _positive_int(steps, "steps")
        metadata = _json_copy(contract or {})
        if instruction is not None and not isinstance(instruction, str):
            raise SessionError("instruction must be text")
        attempt_id = uuid.uuid4().hex
        attempt_path = self.root / "attempts" / model / attempt_id
        attempt = {
            "id": attempt_id, "model_id": model, "path": str(attempt_path),
            "status": "running", "steps": limit, "instruction": instruction,
            "contract": metadata, "started_at": _now(), "finished_at": None,
            "end_reason": None, "recording_complete": False, "steps_sent": 0,
            "recording_path": None, "result": None, "score": None,
            "note": "", "success": None, "counted": False,
            "export": {"status": "not_eligible", "path": None, "error": None, "updated_at": None},
        }
        candidate = copy.deepcopy(self._state)
        candidate["attempts"].append(attempt)
        attempt_path.mkdir(parents=True, exist_ok=False)
        self._commit(candidate)
        return copy.deepcopy(attempt)

    def finish_attempt(self, attempt_id: str, result: dict[str, Any]) -> dict[str, Any]:
        self._ensure_open()
        result = _json_copy(result)
        if not isinstance(result, dict):
            raise SessionError("result must be an object")
        candidate = copy.deepcopy(self._state)
        attempt = self._attempt(candidate, attempt_id)
        if attempt["status"] != "running":
            if attempt["result"] == result:
                return copy.deepcopy(attempt)
            raise SessionError("Attempt already ended; its result cannot be replaced")
        reason = result.get("end_reason")
        if reason not in END_REASONS:
            raise SessionError(f"Invalid end_reason: {reason}")
        steps_sent = result.get("steps_sent", 0)
        if isinstance(steps_sent, bool) or not isinstance(steps_sent, int) or not 0 <= steps_sent <= attempt["steps"]:
            raise SessionError("steps_sent must be an integer from zero to the configured limit")
        complete = result.get("recording_complete") is True
        scorable = reason in SCORABLE_END_REASONS and complete and steps_sent > 0
        if reason == "step_limit" and steps_sent != attempt["steps"]:
            scorable = False
        attempt.update(result=result, end_reason=reason, recording_complete=complete,
                       steps_sent=steps_sent, recording_path=result.get("recording_path"),
                       finished_at=_now(), status="awaiting_score" if scorable else "interrupted")
        if reason == "time_limit":
            attempt.update(outcome=None, success=None)
        if reason in SCORABLE_END_REASONS and not scorable:
            attempt["validation_error"] = "Scorable completion requires a valid recording and the stated completed steps"
        self._commit(candidate)
        return copy.deepcopy(attempt)

    def score_attempt(self, attempt_id: str, score: int | float, note: str = "", *,
                      outcome: str | None = None) -> dict[str, Any]:
        self._ensure_open()
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 100:
            raise SessionError("score must be a finite number from 0 to 100")
        if not isinstance(note, str):
            raise SessionError("note must be text")
        candidate = copy.deepcopy(self._state)
        attempt = self._attempt(candidate, attempt_id)
        if attempt["status"] == "running":
            raise SessionError("Cannot score a running attempt")
        if attempt["score"] is not None:
            previous_outcome = attempt.get("outcome") or ("success" if attempt["success"] else "failed")
            if (attempt["score"] == score and attempt["note"] == note
                    and outcome in (None, previous_outcome)):
                return copy.deepcopy(attempt)
            raise SessionError("Score already submitted; repeated submission must match")
        if outcome not in ("success", "failed"):
            raise SessionError("Choose the task outcome explicitly: success or failed")
        attempt.update(score=score, note=note,
                       outcome=outcome, outcome_source="manual", success=(outcome == "success"),
                       scored_at=_now())
        if attempt["status"] == "awaiting_score":
            attempt.update(status="completed", counted=False)
            attempt["export"].update(status="pending", updated_at=_now())
        self._commit(candidate)
        return copy.deepcopy(attempt)

    def mark_export(self, attempt_id: str, status: str, *, path: str | None = None,
                    error: str | None = None) -> dict[str, Any]:
        self._ensure_open()
        if status not in ("pending", "in_progress", "succeeded", "failed"):
            raise SessionError(f"Invalid export status: {status}")
        if path is not None and not isinstance(path, str):
            raise SessionError("Export path must be text")
        if error is not None and not isinstance(error, str):
            raise SessionError("Export error must be text")
        if status == "succeeded" and not path:
            raise SessionError("Successful export requires a committed dataset path")
        candidate = copy.deepcopy(self._state)
        attempt = self._attempt(candidate, attempt_id)
        if not export_eligible(attempt):
            raise SessionError("Only valid completed or time-limited, scored trials are export eligible")
        previous = attempt["export"]
        if previous["status"] == "succeeded":
            if status == "succeeded" and previous["path"] == path:
                return copy.deepcopy(attempt)
            raise SessionError("Committed export cannot be overwritten")
        previous.update(status=status, path=path, error=error, updated_at=_now())
        attempt["counted"] = is_counted(attempt)
        self._commit(candidate)
        return copy.deepcopy(attempt)

    def set_repeats(self, value: int, model_id: str | None = None) -> dict[str, Any]:
        self._ensure_open()
        _positive_int(value, "repeats")
        self._ensure_no_pending_trial()
        eligible = self._state["model_ids"][self._state["current_index"]:]
        if model_id is not None and model_id not in eligible:
            raise SessionError("Only the current and future models can change target")
        models = eligible if model_id is None else [model_id]
        if not models:
            raise SessionError("Evaluation is complete")
        if any(value < self._completed(model) for model in models):
            raise SessionError("Target cannot be below already completed trial count")
        candidate = copy.deepcopy(self._state)
        for model in models:
            candidate["targets"][model] = value
        if model_id is None:
            candidate["default_repeats"] = value
        self._commit(candidate)
        return self.snapshot()

    def set_steps(self, value: int) -> dict[str, Any]:
        self._ensure_open()
        _positive_int(value, "steps")
        self._ensure_no_pending_trial()
        candidate = copy.deepcopy(self._state)
        candidate["steps"] = value
        self._commit(candidate)
        return self.snapshot()

    def advance_model(self) -> str | None:
        self._ensure_open()
        self._ensure_no_pending_trial()
        model = self.current_model
        if model is None:
            return None
        if self._completed(model) < self._state["targets"][model]:
            raise SessionError("Complete the current model's target before advancing")
        candidate = copy.deepcopy(self._state)
        candidate["current_index"] += 1
        self._commit(candidate)
        return self.current_model
