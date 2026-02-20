# Copyright 2026
#
# Licensed under the Apache License, Version 2.0.
"""
Coordinator for dual-loop GRPO (questioner + solver), v5 cycle-gated.

Key behavior change from v4:
- Questioner submits candidates per cycle with expected_n/min_required.
- Cycle is finalized atomically:
  - if parsable_count < min_required: mark whole cycle failed (zero-style outcome)
  - else queue exactly min_required candidates to solver.
- Rewards are awaited per candidate id; unresolved entries are polled by questioner.

Endpoints:
- GET  /questioner/seeds
- POST /questioner/cycle/submit
- POST /questioner/rewards/await
- GET  /solver/questions
- POST /solver/results
- GET  /health
- GET  /debug/state
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
import uuid
from collections import Counter, deque
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional

import requests
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

app = FastAPI()

_DEFAULT_SPLIT = os.getenv("GSM8K_SPLIT", "train")
_LOG_ROOT = os.getenv("QUESTION_LOG_ROOT", "/workspace/verl/verl")
_EXPERIMENT_NAME = os.getenv("QUESTION_EXPERIMENT_NAME")
_ASKED_DIR = ""
_ANSWERED_DIR = ""
_SUBMITTER_URL = os.getenv("QUESTION_SUBMITTER_URL", "http://127.0.0.1:8090/questions")
_SUBMITTER_TIMEOUT_S = float(os.getenv("QUESTION_SUBMITTER_TIMEOUT_S", "10"))
_SUBMITTER_POLL_S = float(os.getenv("QUESTION_SUBMITTER_POLL_S", "1"))
_SOLVER_FETCH_WAIT_S = float(os.getenv("SOLVER_FETCH_WAIT_S", "120"))
_SOLVER_RESULT_WAIT_S = float(os.getenv("SOLVER_RESULT_WAIT_S", "180"))
_CYCLE_TIMEOUT_S = float(os.getenv("DUAL_LOOP_CYCLE_TIMEOUT_S", "120"))
_FAIL_ON_TIMEOUT = os.getenv("DUAL_LOOP_FAIL_ON_TIMEOUT", "true").strip().lower() in {"1", "true", "yes", "on"}
_LOG_QUESTIONER_CANDIDATES = os.getenv("DUAL_LOOP_LOG_QUESTIONER_CANDIDATES", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_TRACE_ENABLED = os.getenv("DUAL_LOOP_TRACE", "false").strip().lower() in {"1", "true", "yes", "on"}
_TRACE_INCLUDE_TEXT = os.getenv("DUAL_LOOP_TRACE_INCLUDE_TEXT", "false").strip().lower() in {"1", "true", "yes", "on"}
_TRACE_MAX_TEXT_CHARS = int(os.getenv("DUAL_LOOP_TRACE_MAX_TEXT_CHARS", "180"))

_lock = threading.RLock()
_cond = threading.Condition(_lock)

_seed_queues: Dict[str, Deque[Dict[str, object]]] = {}
_dataset_size_by_split: Dict[str, int] = {}
_received_seed_batch_id = 0
_served_seed_batch_id = 0

_candidate_queue: Deque[str] = deque()
_candidates: Dict[str, Dict[str, object]] = {}
_cycles: Dict[str, Dict[str, object]] = {}
_active_cycle_id_by_base: Dict[str, str] = {}
_cycle_rotate_counter = 0


class QuestionerSeedItem(BaseModel):
    question: str
    answer: Optional[str] = None
    solution: Optional[str] = None
    index: Optional[int] = None
    split: Optional[str] = None
    question_id: Optional[str] = None
    order_id: Optional[int] = None
    source_epoch: Optional[int] = None
    batch_id: Optional[int] = None


class QuestionerCycleCandidateItem(BaseModel):
    candidate_id: str
    question: Optional[str] = None
    answer: Optional[str] = None
    parse_ok: bool = False
    question_ok: bool = False
    answer_ok: bool = False
    numeric_final_ok: bool = False
    parse_mode: Optional[str] = None
    split: Optional[str] = None
    seed_question_id: Optional[str] = None
    seed_question: Optional[str] = None
    seed_answer: Optional[str] = None
    seed_solution: Optional[str] = None
    seed_extra_info: Optional[dict] = None
    meta: Optional[dict] = None


class QuestionerCycleSubmitPayload(BaseModel):
    split: Optional[str] = None
    cycle_id: str
    expected_n: int = Field(..., ge=1, le=4096)
    min_required: Optional[int] = Field(default=None, ge=1, le=4096)
    cycle_timeout_s: Optional[float] = Field(default=None, ge=0)
    candidate: QuestionerCycleCandidateItem


class QuestionerAwaitPayload(BaseModel):
    candidate_ids: List[str]
    wait_s: Optional[float] = None


class SolverResultItem(BaseModel):
    candidate_id: str
    difficulty: Optional[float] = None
    n: Optional[int] = None
    n_correct: Optional[int] = None
    correct: Optional[bool] = None
    meta: Optional[dict] = None


class SolverResultsPayload(BaseModel):
    results: List[SolverResultItem] = Field(default_factory=list)


def _utc_now() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _resolve_log_dirs(experiment_name: Optional[str]) -> tuple[str, str]:
    if experiment_name:
        root = os.path.join(_LOG_ROOT, f"{experiment_name}_questions_asked_answered")
        return os.path.join(root, "questions_asked"), os.path.join(root, "questions_answered")
    return os.path.join(_LOG_ROOT, "questions_asked"), os.path.join(_LOG_ROOT, "questions_answered")


def _set_experiment_name(experiment_name: Optional[str]) -> None:
    global _EXPERIMENT_NAME, _ASKED_DIR, _ANSWERED_DIR
    _EXPERIMENT_NAME = experiment_name
    _ASKED_DIR, _ANSWERED_DIR = _resolve_log_dirs(experiment_name)


def _ensure_dirs() -> None:
    os.makedirs(_ASKED_DIR, exist_ok=True)
    os.makedirs(_ANSWERED_DIR, exist_ok=True)


def _batch_log_path(root_dir: str, prefix: str, batch_id: Optional[int | str]) -> str:
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
    batch_label = "none" if batch_id is None else str(batch_id)
    token = uuid.uuid4().hex[:8]
    filename = f"{prefix}_batch_{batch_label}_{ts}_{token}.jsonl"
    return os.path.join(root_dir, filename)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return value.item()
        except Exception:
            return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_jsonl(path: str, record: dict) -> None:
    _ensure_dirs()
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(_jsonable(record), ensure_ascii=False, separators=(",", ":")) + "\n")


def _preview(text: Optional[str], max_chars: Optional[int] = None) -> Optional[str]:
    if text is None:
        return None
    max_chars = _TRACE_MAX_TEXT_CHARS if max_chars is None else max_chars
    clean = re.sub(r"\s+", " ", str(text)).strip()
    if len(clean) <= max_chars:
        return clean
    return clean[:max_chars] + "...(truncated)"


def _trace(event: str, **fields: Any) -> None:
    if not _TRACE_ENABLED:
        return
    payload = {"ts": _utc_now(), "event": event}
    payload.update({k: _jsonable(v) for k, v in fields.items()})
    print(f"[dual-loop-trace] {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}", flush=True)


def _extract_solution(solution_str: str | None) -> Optional[str]:
    if not solution_str:
        return None
    match = re.search(r"####\s*(\-?[0-9][0-9,\.]*)", solution_str)
    if not match:
        return None
    return match.group(1).replace(",", "")


def _next_received_seed_batch_id() -> int:
    global _received_seed_batch_id
    _received_seed_batch_id += 1
    return _received_seed_batch_id


def _next_served_seed_batch_id() -> int:
    global _served_seed_batch_id
    _served_seed_batch_id += 1
    return _served_seed_batch_id


def _get_seed_queue(split: str) -> Deque[Dict[str, object]]:
    if split not in _seed_queues:
        _seed_queues[split] = deque()
    return _seed_queues[split]


def _enqueue_seed_items(split: str, items: List[Dict[str, object]]) -> None:
    with _cond:
        _get_seed_queue(split).extend(items)
        _cond.notify_all()


def _fetch_seeds_from_submitter(split: str, n: int) -> int:
    if not _SUBMITTER_URL or n <= 0:
        return 0

    try:
        r = requests.get(_SUBMITTER_URL, params={"n": n, "split": split}, timeout=_SUBMITTER_TIMEOUT_S)
        r.raise_for_status()
        payload = r.json()
    except Exception:
        return 0

    items = payload.get("questions") or []
    if not items:
        _trace("receiver.seed_fetch.empty", split=split, requested=n, submitter_url=_SUBMITTER_URL)
        return 0

    dataset_size = payload.get("dataset_size")
    if dataset_size is not None:
        _dataset_size_by_split[split] = int(dataset_size)

    received_at = _utc_now()
    batch_id = _next_received_seed_batch_id()
    for item in items:
        item.setdefault("split", split)
        item.setdefault("batch_id", batch_id)
        item.setdefault("received_at", received_at)
        idx = int(item.get("index", 0))
        item.setdefault("question_id", f"{split}:{idx}")

    _enqueue_seed_items(split, items)
    asked_path = _batch_log_path(_ASKED_DIR, "questions_asked", batch_id)
    _write_jsonl(
        asked_path,
        {
            "batch_id": batch_id,
            "split": split,
            "received_at": received_at,
            "source": "submitter",
            "dataset_size": _dataset_size_by_split.get(split),
            "questions": items,
        },
    )
    _trace(
        "receiver.seed_fetch.ok",
        split=split,
        requested=n,
        received=len(items),
        batch_id=batch_id,
        dataset_size=_dataset_size_by_split.get(split),
    )
    return len(items)


def _dequeue_seeds(split: str, n: int, wait_s: float) -> List[Dict[str, object]]:
    with _cond:
        q = _get_seed_queue(split)
        deadline = None if wait_s < 0 else time.monotonic() + max(wait_s, 0.0)

        while len(q) < n and (deadline is None or time.monotonic() < deadline):
            remaining = None if deadline is None else max(deadline - time.monotonic(), 0.0)
            timeout = _SUBMITTER_POLL_S if remaining is None else min(_SUBMITTER_POLL_S, remaining)
            _cond.wait(timeout=timeout)

        take_n = n if _FAIL_ON_TIMEOUT else min(n, len(q))
        if len(q) < take_n:
            return []
        return [q.popleft() for _ in range(take_n)]


def _get_seeds_with_refill(split: str, n: int, wait_s: float) -> List[Dict[str, object]]:
    deadline = None if wait_s < 0 else time.monotonic() + max(wait_s, 0.0)
    while True:
        with _lock:
            qsize = len(_get_seed_queue(split))
        if qsize >= n:
            break

        _fetch_seeds_from_submitter(split, n - qsize)
        with _lock:
            qsize = len(_get_seed_queue(split))
        if qsize >= n:
            break
        if wait_s == 0:
            break
        if deadline is not None and time.monotonic() >= deadline:
            break
        time.sleep(_SUBMITTER_POLL_S)

    return _dequeue_seeds(split, n, wait_s)


def _parsable_for_solver(rec: Dict[str, object]) -> bool:
    return bool(rec.get("parse_ok", False)) and bool(rec.get("solution")) and bool(rec.get("question"))


def _ensure_cycle_locked(
    cycle_id: str,
    split: str,
    expected_n: int,
    min_required: int,
    cycle_timeout_s: float,
) -> Dict[str, object]:
    now = time.monotonic()
    base_cycle_id = str(cycle_id)

    # If a rotated active cycle already exists for this base id, always reuse it
    # so all candidates for the current cycle stay grouped together.
    active_id = _active_cycle_id_by_base.get(base_cycle_id)
    if active_id is not None:
        active_cycle = _cycles.get(active_id)
        if active_cycle is not None and not bool(active_cycle.get("finalized", False)):
            if (
                int(active_cycle.get("expected_n", expected_n)) != expected_n
                or int(active_cycle.get("min_required", min_required)) != min_required
            ):
                _trace(
                    "receiver.cycle.mismatch",
                    cycle_id=active_id,
                    base_cycle_id=base_cycle_id,
                    expected_existing=active_cycle.get("expected_n"),
                    min_required_existing=active_cycle.get("min_required"),
                    expected_new=expected_n,
                    min_required_new=min_required,
                )
            return active_cycle
        # stale pointer
        _active_cycle_id_by_base.pop(base_cycle_id, None)

    existing = _cycles.get(base_cycle_id)
    if existing is not None:
        if not bool(existing.get("finalized", False)):
            if (
                int(existing.get("expected_n", expected_n)) != expected_n
                or int(existing.get("min_required", min_required)) != min_required
            ):
                _trace(
                    "receiver.cycle.mismatch",
                    cycle_id=base_cycle_id,
                    base_cycle_id=base_cycle_id,
                    expected_existing=existing.get("expected_n"),
                    min_required_existing=existing.get("min_required"),
                    expected_new=expected_n,
                    min_required_new=min_required,
                )
            _active_cycle_id_by_base[base_cycle_id] = base_cycle_id
            return existing

    # If the requested base id was already finalized, rotate to a fresh internal id.
    # This prevents stale client-side cycle ids (e.g. constant "1") from poisoning
    # subsequent rounds.
    internal_cycle_id = base_cycle_id
    if existing is not None and bool(existing.get("finalized", False)):
        global _cycle_rotate_counter
        _cycle_rotate_counter += 1
        internal_cycle_id = f"{base_cycle_id}::r{_cycle_rotate_counter}"
        _trace(
            "receiver.cycle.rotated",
            base_cycle_id=base_cycle_id,
            internal_cycle_id=internal_cycle_id,
        )

    cycle = {
        "cycle_id": internal_cycle_id,
        "base_cycle_id": base_cycle_id,
        "split": split,
        "expected_n": int(expected_n),
        "min_required": int(min_required),
        "created_at": _utc_now(),
        "created_mono": now,
        "deadline_mono": now + max(0.0, float(cycle_timeout_s)),
        "submitted": 0,
        "candidate_ids": [],
        "finalized": False,
        "failed": False,
        "failure_reason": None,
        "selected_candidate_ids": [],
        "finalized_at": None,
        "resolved_selected": 0,
    }
    _cycles[internal_cycle_id] = cycle
    _active_cycle_id_by_base[base_cycle_id] = internal_cycle_id
    _trace(
        "receiver.cycle.created",
        cycle_id=internal_cycle_id,
        base_cycle_id=base_cycle_id,
        split=split,
        expected_n=expected_n,
        min_required=min_required,
        cycle_timeout_s=cycle_timeout_s,
    )
    return cycle


def _register_cycle_candidate_locked(payload: QuestionerCycleSubmitPayload) -> Dict[str, object]:
    item = payload.candidate
    split = item.split or payload.split or _DEFAULT_SPLIT
    requested_cycle_id = str(payload.cycle_id)
    expected_n = int(payload.expected_n)
    min_required = int(payload.min_required if payload.min_required is not None else expected_n)
    min_required = max(1, min(min_required, expected_n))
    cycle_timeout_s = float(payload.cycle_timeout_s if payload.cycle_timeout_s is not None else _CYCLE_TIMEOUT_S)

    cycle = _ensure_cycle_locked(
        cycle_id=requested_cycle_id,
        split=split,
        expected_n=expected_n,
        min_required=min_required,
        cycle_timeout_s=cycle_timeout_s,
    )
    cycle_id = str(cycle["cycle_id"])

    candidate_id = str(item.candidate_id)
    existing = _candidates.get(candidate_id)
    if existing is not None:
        _trace(
            "receiver.candidate.duplicate",
            candidate_id=candidate_id,
            cycle_id=existing.get("cycle_id"),
            status=existing.get("status"),
        )
        return existing

    question = (item.question or "").strip()
    answer = (item.answer or "").strip()
    solution = _extract_solution(answer) if item.parse_ok else None

    rec = {
        "candidate_id": candidate_id,
        "cycle_id": cycle_id,
        "split": split,
        "question": question,
        "answer": answer,
        "solution": solution,
        "parse_ok": bool(item.parse_ok),
        "question_ok": bool(item.question_ok),
        "answer_ok": bool(item.answer_ok),
        "numeric_final_ok": bool(item.numeric_final_ok),
        "parse_mode": item.parse_mode,
        "seed_question_id": item.seed_question_id,
        "seed_question": item.seed_question,
        "seed_answer": item.seed_answer,
        "seed_solution": item.seed_solution,
        "seed_extra_info": item.seed_extra_info,
        "meta": item.meta,
        "status": "submitted",
        "resolved": False,
        "difficulty": None,
        "selected": False,
        "batch_failed": False,
        "created_at": _utc_now(),
    }
    _candidates[candidate_id] = rec
    cycle["candidate_ids"].append(candidate_id)
    cycle["submitted"] = int(cycle.get("submitted", 0)) + 1

    _trace(
        "receiver.candidate.submitted",
        candidate_id=candidate_id,
        cycle_id=cycle_id,
        requested_cycle_id=requested_cycle_id,
        split=split,
        parse_ok=bool(item.parse_ok),
        has_solution=solution is not None,
        submitted=cycle["submitted"],
        expected_n=cycle["expected_n"],
        question_preview=_preview(question) if _TRACE_INCLUDE_TEXT else None,
    )
    return rec


def _finalize_cycle_locked(cycle: Dict[str, object], reason: str) -> None:
    if bool(cycle.get("finalized", False)):
        return

    cycle_id = str(cycle["cycle_id"])
    candidate_ids = list(cycle.get("candidate_ids", []))
    min_required = int(cycle.get("min_required", 1))

    parsable = [cid for cid in candidate_ids if cid in _candidates and _parsable_for_solver(_candidates[cid])]

    now = _utc_now()
    if len(parsable) < min_required:
        cycle["failed"] = True
        cycle["failure_reason"] = f"insufficient_parsable:{len(parsable)}<{min_required}"
        cycle["selected_candidate_ids"] = []
        for cid in candidate_ids:
            rec = _candidates[cid]
            rec["status"] = "cycle_failed"
            rec["resolved"] = True
            rec["difficulty"] = 0.0
            rec["selected"] = False
            rec["batch_failed"] = True
            rec["resolved_at"] = now
        _trace(
            "receiver.cycle.finalized.failed",
            cycle_id=cycle_id,
            reason=reason,
            submitted=len(candidate_ids),
            parsable=len(parsable),
            min_required=min_required,
        )
    else:
        selected = parsable[:min_required]
        selected_set = set(selected)
        cycle["failed"] = False
        cycle["failure_reason"] = None
        cycle["selected_candidate_ids"] = selected

        for cid in candidate_ids:
            rec = _candidates[cid]
            if cid in selected_set:
                rec["status"] = "queued"
                rec["resolved"] = False
                rec["difficulty"] = None
                rec["selected"] = True
                rec["batch_failed"] = False
                if not bool(rec.get("queued_to_solver", False)):
                    _candidate_queue.append(cid)
                    rec["queued_to_solver"] = True
            else:
                rec["status"] = "not_selected"
                rec["resolved"] = True
                rec["difficulty"] = 0.0
                rec["selected"] = False
                rec["batch_failed"] = False
                rec["resolved_at"] = now

        _trace(
            "receiver.cycle.finalized.success",
            cycle_id=cycle_id,
            reason=reason,
            submitted=len(candidate_ids),
            parsable=len(parsable),
            min_required=min_required,
            selected_ids=selected,
            solver_queue_size=len(_candidate_queue),
        )

    cycle["finalized"] = True
    cycle["finalized_at"] = now
    base_cycle_id = str(cycle.get("base_cycle_id", cycle_id))
    if _active_cycle_id_by_base.get(base_cycle_id) == cycle_id:
        _active_cycle_id_by_base.pop(base_cycle_id, None)


def _finalize_ready_or_expired_cycles_locked(force_expired_only: bool = False) -> None:
    now = time.monotonic()
    for cycle in list(_cycles.values()):
        if bool(cycle.get("finalized", False)):
            continue

        submitted = int(cycle.get("submitted", 0))
        expected_n = int(cycle.get("expected_n", 1))
        deadline = float(cycle.get("deadline_mono", now))

        if submitted >= expected_n and not force_expired_only:
            _finalize_cycle_locked(cycle, reason="all_submitted")
            continue

        if now >= deadline:
            _finalize_cycle_locked(cycle, reason="cycle_timeout")


def _solver_take_questions(n: int, wait_s: float) -> List[dict]:
    with _cond:
        deadline = None if wait_s < 0 else time.monotonic() + max(wait_s, 0.0)
        while len(_candidate_queue) < n and (deadline is None or time.monotonic() < deadline):
            _finalize_ready_or_expired_cycles_locked(force_expired_only=True)
            remaining = None if deadline is None else max(deadline - time.monotonic(), 0.0)
            timeout = _SUBMITTER_POLL_S if remaining is None else min(_SUBMITTER_POLL_S, remaining)
            _cond.wait(timeout=timeout)

        take_n = n if _FAIL_ON_TIMEOUT else min(n, len(_candidate_queue))
        if len(_candidate_queue) < take_n:
            _trace(
                "receiver.solver_take.timeout",
                requested=n,
                take_n=take_n,
                queue_size=len(_candidate_queue),
                wait_s=wait_s,
            )
            return []

        now = _utc_now()
        out = []
        for _ in range(take_n):
            cid = _candidate_queue.popleft()
            rec = _candidates[cid]
            rec["status"] = "assigned"
            rec["assigned_at"] = now
            _trace(
                "receiver.solver_take.assigned",
                candidate_id=cid,
                cycle_id=rec.get("cycle_id"),
                split=rec.get("split"),
                queue_size_after_pop=len(_candidate_queue),
                question_preview=_preview(rec.get("question")) if _TRACE_INCLUDE_TEXT else None,
            )
            out.append(
                {
                    "candidate_id": cid,
                    "question": rec["question"],
                    "answer": rec["answer"],
                    "solution": rec["solution"],
                    "split": rec["split"],
                    "seed_question_id": rec.get("seed_question_id"),
                    "cycle_id": rec.get("cycle_id"),
                    "meta": rec.get("meta"),
                }
            )
        return out


def _to_difficulty(result: SolverResultItem) -> float:
    if result.difficulty is not None:
        difficulty = float(result.difficulty)
    elif result.n is not None and result.n_correct is not None:
        n = max(int(result.n), 1)
        difficulty = float(result.n_correct) / float(n)
    elif result.correct is not None:
        difficulty = 1.0 if bool(result.correct) else 0.0
    else:
        raise ValueError("Missing difficulty signal. Provide one of difficulty or (n, n_correct) or correct.")
    return min(1.0, max(0.0, difficulty))


def _apply_solver_results(results: List[SolverResultItem]) -> None:
    with _cond:
        for item in results:
            if item.candidate_id not in _candidates:
                raise KeyError(f"Unknown candidate_id: {item.candidate_id}")
            rec = _candidates[item.candidate_id]
            difficulty = _to_difficulty(item)
            rec["difficulty"] = difficulty
            rec["resolved"] = True
            rec["status"] = "solved"
            rec["batch_failed"] = False
            rec["solved_at"] = _utc_now()
            rec["solver_result"] = {
                "difficulty": difficulty,
                "n": item.n,
                "n_correct": item.n_correct,
                "correct": item.correct,
                "meta": item.meta,
            }
            _trace(
                "receiver.solver_result.applied",
                candidate_id=item.candidate_id,
                cycle_id=rec.get("cycle_id"),
                difficulty=difficulty,
                n=item.n,
                n_correct=item.n_correct,
            )

        # Update cycle resolved_selected counters.
        for cycle in _cycles.values():
            selected = cycle.get("selected_candidate_ids") or []
            if not selected:
                continue
            resolved_selected = sum(1 for cid in selected if bool(_candidates.get(cid, {}).get("resolved", False)))
            cycle["resolved_selected"] = resolved_selected
            if bool(cycle.get("finalized", False)) and resolved_selected == len(selected):
                cycle["completed_at"] = _utc_now()

        _cond.notify_all()


def _default_cycle_reward_summary() -> Dict[str, object]:
    return {
        "cycle_expected_n": 0,
        "cycle_min_required": 0,
        "cycle_submitted": 0,
        "cycle_finalized": False,
        "cycle_failed": False,
        "cycle_failure_reason": None,
        "cycle_complete": False,
        "cycle_parsable_count": 0,
        "cycle_parse_rate": 0.0,
        "cycle_all_parsable": False,
        "cycle_selected_count": 0,
        "cycle_selected_resolved_count": 0,
        "cycle_selected_difficulty_mean": 0.0,
        "cycle_selected_difficulty_min": None,
        "cycle_selected_difficulty_max": None,
    }


def _cycle_summary_locked(cycle_id: Optional[str]) -> Dict[str, object]:
    summary = _default_cycle_reward_summary()
    if not cycle_id:
        return summary
    cycle = _cycles.get(str(cycle_id))
    if cycle is None:
        return summary

    candidate_ids = list(cycle.get("candidate_ids") or [])
    expected_n = int(cycle.get("expected_n", 0))
    min_required = int(cycle.get("min_required", 0))
    submitted = int(cycle.get("submitted", len(candidate_ids)))
    finalized = bool(cycle.get("finalized", False))
    failed = bool(cycle.get("failed", False))
    failure_reason = cycle.get("failure_reason")

    parsable_count = sum(
        1 for cid in candidate_ids if cid in _candidates and _parsable_for_solver(_candidates[cid])
    )
    parse_rate = (float(parsable_count) / float(expected_n)) if expected_n > 0 else 0.0
    all_parsable = bool(finalized and expected_n > 0 and submitted >= expected_n and parsable_count >= expected_n)

    selected_ids = list(cycle.get("selected_candidate_ids") or [])
    selected_count = len(selected_ids)
    selected_resolved_count = sum(
        1 for cid in selected_ids if bool(_candidates.get(cid, {}).get("resolved", False))
    )
    selected_difficulties = []
    for cid in selected_ids:
        diff = _candidates.get(cid, {}).get("difficulty", None)
        if diff is None:
            continue
        try:
            selected_difficulties.append(float(diff))
        except Exception:
            continue

    cycle_complete = bool(finalized and (failed or selected_resolved_count >= selected_count))
    difficulty_mean = (
        float(sum(selected_difficulties) / len(selected_difficulties)) if selected_difficulties else 0.0
    )
    difficulty_min = float(min(selected_difficulties)) if selected_difficulties else None
    difficulty_max = float(max(selected_difficulties)) if selected_difficulties else None

    summary.update(
        {
            "cycle_expected_n": expected_n,
            "cycle_min_required": min_required,
            "cycle_submitted": submitted,
            "cycle_finalized": finalized,
            "cycle_failed": failed,
            "cycle_failure_reason": failure_reason,
            "cycle_complete": cycle_complete,
            "cycle_parsable_count": parsable_count,
            "cycle_parse_rate": parse_rate,
            "cycle_all_parsable": all_parsable,
            "cycle_selected_count": selected_count,
            "cycle_selected_resolved_count": selected_resolved_count,
            "cycle_selected_difficulty_mean": difficulty_mean,
            "cycle_selected_difficulty_min": difficulty_min,
            "cycle_selected_difficulty_max": difficulty_max,
        }
    )
    return summary


def _build_candidate_reward_result_locked(
    cid: str,
    rec: Dict[str, object],
    cycle_cache: Optional[Dict[str, Dict[str, object]]] = None,
) -> Dict[str, object]:
    cycle_id = rec.get("cycle_id")
    cycle_key = str(cycle_id) if cycle_id is not None else "__none__"
    if cycle_cache is None:
        cycle_summary = _cycle_summary_locked(cycle_id if cycle_id is not None else None)
    else:
        cycle_summary = cycle_cache.get(cycle_key)
        if cycle_summary is None:
            cycle_summary = _cycle_summary_locked(cycle_id if cycle_id is not None else None)
            cycle_cache[cycle_key] = cycle_summary

    result = {
        "candidate_id": cid,
        "parse_ok": bool(rec.get("parse_ok", False)),
        "question_ok": bool(rec.get("question_ok", False)),
        "answer_ok": bool(rec.get("answer_ok", False)),
        "numeric_final_ok": bool(rec.get("numeric_final_ok", False)),
        "parse_mode": rec.get("parse_mode"),
        "difficulty": rec.get("difficulty"),
        "status": rec.get("status"),
        "resolved": bool(rec.get("resolved", False)),
        "selected": bool(rec.get("selected", False)),
        "batch_failed": bool(rec.get("batch_failed", False)),
        "cycle_id": cycle_id,
        "solution": rec.get("solution"),
    }
    result.update(cycle_summary)
    return result


def _await_candidate_rewards(candidate_ids: List[str], wait_s: float) -> List[dict]:
    with _cond:
        unknown = [cid for cid in candidate_ids if cid not in _candidates]
        if unknown:
            raise KeyError(f"Unknown candidate_ids: {unknown}")

        deadline = None if wait_s < 0 else time.monotonic() + max(wait_s, 0.0)

        def all_resolved() -> bool:
            return all(bool(_candidates[cid].get("resolved", False)) for cid in candidate_ids)

        while not all_resolved() and (deadline is None or time.monotonic() < deadline):
            _finalize_ready_or_expired_cycles_locked(force_expired_only=True)
            if all_resolved():
                break
            remaining = None if deadline is None else max(deadline - time.monotonic(), 0.0)
            timeout = _SUBMITTER_POLL_S if remaining is None else min(_SUBMITTER_POLL_S, remaining)
            _cond.wait(timeout=timeout)

        if not all_resolved():
            unresolved = [cid for cid in candidate_ids if not bool(_candidates[cid].get("resolved", False))]
            if _FAIL_ON_TIMEOUT:
                _trace("receiver.await.timeout", candidate_ids=candidate_ids, unresolved=unresolved, wait_s=wait_s)
                raise TimeoutError(f"Timed out waiting for candidate resolution. unresolved={unresolved}")

        out = []
        cycle_cache: Dict[str, Dict[str, object]] = {}
        for cid in candidate_ids:
            rec = _candidates[cid]
            out.append(_build_candidate_reward_result_locked(cid, rec, cycle_cache=cycle_cache))
        return out


def _log_questioner_cycle_submit(payload: QuestionerCycleSubmitPayload, rec: Dict[str, object]) -> None:
    if not _LOG_QUESTIONER_CANDIDATES:
        return
    answered_path = _batch_log_path(_ANSWERED_DIR, "questions_answered", payload.cycle_id)
    _write_jsonl(
        answered_path,
        {
            "source": "questioner_cycle_submit",
            "received_at": _utc_now(),
            "split": payload.split or _DEFAULT_SPLIT,
            "cycle_id": payload.cycle_id,
            "expected_n": payload.expected_n,
            "min_required": payload.min_required,
            "candidate": rec,
        },
    )


def _log_solver_results(payload: SolverResultsPayload) -> None:
    answered_path = _batch_log_path(_ANSWERED_DIR, "questions_answered", "solver")
    _write_jsonl(
        answered_path,
        {
            "source": "solver_results",
            "received_at": _utc_now(),
            "count": len(payload.results),
            "responses": [result.model_dump() for result in payload.results],
        },
    )


@app.get("/health")
def health():
    with _lock:
        _finalize_ready_or_expired_cycles_locked(force_expired_only=True)
        status_counts = Counter(rec.get("status", "unknown") for rec in _candidates.values())
        cycle_state_counts = Counter(
            "failed"
            if bool(c.get("failed", False))
            else ("finalized" if bool(c.get("finalized", False)) else "open")
            for c in _cycles.values()
        )
        seed_queue_sizes = {split: len(q) for split, q in _seed_queues.items()}
        return {
            "ok": True,
            "fail_on_timeout": _FAIL_ON_TIMEOUT,
            "seed_queue_sizes": seed_queue_sizes,
            "solver_queue_size": len(_candidate_queue),
            "candidate_counts": dict(status_counts),
            "cycle_counts": dict(cycle_state_counts),
            "dataset_size": _dataset_size_by_split,
        }


@app.on_event("startup")
def _startup() -> None:
    _set_experiment_name(_EXPERIMENT_NAME)
    _ensure_dirs()


@app.get("/debug/state")
def debug_state():
    with _lock:
        _finalize_ready_or_expired_cycles_locked(force_expired_only=True)
        status_counts = Counter(rec.get("status", "unknown") for rec in _candidates.values())
        cycle_state_counts = Counter(
            "failed"
            if bool(c.get("failed", False))
            else ("finalized" if bool(c.get("finalized", False)) else "open")
            for c in _cycles.values()
        )
        return {
            "solver_queue_size": len(_candidate_queue),
            "candidates_total": len(_candidates),
            "candidate_counts": dict(status_counts),
            "cycles_total": len(_cycles),
            "cycle_counts": dict(cycle_state_counts),
            "sample_candidate_ids": list(_candidates.keys())[:20],
            "sample_cycle_ids": list(_cycles.keys())[:20],
        }


@app.get("/questioner/seeds")
def questioner_seeds(
    n: int = Query(8, ge=1, le=4096),
    split: Optional[str] = Query(None),
    wait_s: Optional[float] = Query(None),
):
    split = split or _DEFAULT_SPLIT
    wait_s = _SOLVER_FETCH_WAIT_S if wait_s is None else float(max(wait_s, 0.0))

    items = _get_seeds_with_refill(split, n, wait_s)
    if len(items) < n:
        _trace("receiver.questioner_seeds.timeout", split=split, requested=n, received=len(items), wait_s=wait_s)
        raise HTTPException(
            status_code=504,
            detail={
                "error": "seed_timeout",
                "needed": n,
                "received": len(items),
                "split": split,
                "wait_s": wait_s,
            },
        )

    served_batch_id = _next_served_seed_batch_id()
    served_at = _utc_now()
    for item in items:
        item["served_at"] = served_at
        item["served_batch_id"] = served_batch_id

    asked_path = _batch_log_path(_ASKED_DIR, "questions_asked", served_batch_id)
    _write_jsonl(
        asked_path,
        {
            "batch_id": served_batch_id,
            "split": split,
            "served_at": served_at,
            "source": "questioner_seeds",
            "dataset_size": _dataset_size_by_split.get(split),
            "questions": items,
        },
    )
    _trace(
        "receiver.questioner_seeds.served",
        split=split,
        requested=n,
        served=len(items),
        batch_id=served_batch_id,
    )

    return {
        "questions": items,
        "split": split,
        "batch_id": served_batch_id,
        "dataset_size": _dataset_size_by_split.get(split),
        "served_at": served_at,
    }


@app.post("/questioner/cycle/submit")
def questioner_cycle_submit(payload: QuestionerCycleSubmitPayload):
    with _cond:
        rec = _register_cycle_candidate_locked(payload)
        _finalize_ready_or_expired_cycles_locked(force_expired_only=False)
        cycle = _cycles[str(rec.get("cycle_id"))]
        _log_questioner_cycle_submit(payload, rec)
        _cond.notify_all()

    return {
        "ok": True,
        "candidate_id": rec["candidate_id"],
        "cycle_id": rec.get("cycle_id"),
        "requested_cycle_id": payload.cycle_id,
        "submitted": int(cycle.get("submitted", 0)),
        "expected_n": int(cycle.get("expected_n", 1)),
        "min_required": int(cycle.get("min_required", 1)),
        "finalized": bool(cycle.get("finalized", False)),
        "failed": bool(cycle.get("failed", False)),
        "failure_reason": cycle.get("failure_reason"),
        "solver_queue_size": len(_candidate_queue),
    }


@app.post("/questioner/rewards/await")
def questioner_rewards_await(payload: QuestionerAwaitPayload):
    if not payload.candidate_ids:
        raise HTTPException(status_code=400, detail="No candidate_ids provided.")
    wait_s = _SOLVER_RESULT_WAIT_S if payload.wait_s is None else float(max(payload.wait_s, 0.0))

    if wait_s <= 0:
        with _lock:
            _finalize_ready_or_expired_cycles_locked(force_expired_only=True)
            unknown = [cid for cid in payload.candidate_ids if cid not in _candidates]
            if unknown:
                raise HTTPException(status_code=404, detail=f"Unknown candidate_ids: {unknown}")
            results = []
            cycle_cache: Dict[str, Dict[str, object]] = {}
            for cid in payload.candidate_ids:
                rec = _candidates[cid]
                results.append(_build_candidate_reward_result_locked(cid, rec, cycle_cache=cycle_cache))
        return {"ok": True, "results": results}

    try:
        results = _await_candidate_rewards(payload.candidate_ids, wait_s=wait_s)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except TimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail={
                "error": "candidate_result_timeout",
                "wait_s": wait_s,
                "candidate_ids": payload.candidate_ids,
                "message": str(exc),
            },
        ) from exc

    _trace("receiver.questioner_rewards.await_done", candidate_ids=payload.candidate_ids, wait_s=wait_s)
    return {"ok": True, "results": results}


@app.get("/solver/questions")
def solver_questions(
    n: int = Query(8, ge=1, le=4096),
    wait_s: Optional[float] = Query(None),
):
    wait_s = _SOLVER_FETCH_WAIT_S if wait_s is None else float(max(wait_s, 0.0))
    items = _solver_take_questions(n=n, wait_s=wait_s)
    if len(items) < n:
        raise HTTPException(
            status_code=504,
            detail={
                "error": "solver_fetch_timeout",
                "needed": n,
                "received": len(items),
                "wait_s": wait_s,
                "queue_size": len(_candidate_queue),
            },
        )

    _trace("receiver.solver_questions.served", requested=n, served=len(items), wait_s=wait_s)

    asked_path = _batch_log_path(_ASKED_DIR, "questions_asked", None)
    _write_jsonl(
        asked_path,
        {
            "source": "solver_questions",
            "served_at": _utc_now(),
            "count": len(items),
            "questions": items,
        },
    )
    return {"ok": True, "questions": items, "count": len(items)}


@app.post("/solver/results")
def solver_results(payload: SolverResultsPayload):
    if not payload.results:
        raise HTTPException(status_code=400, detail="No solver results provided.")
    try:
        _apply_solver_results(payload.results)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    _log_solver_results(payload)
    _trace("receiver.solver_results.posted", count=len(payload.results))
    return {"ok": True, "count": len(payload.results)}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dual-loop coordinator v5 (cycle-gated).")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--experiment-name", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    import uvicorn

    if args.experiment_name:
        os.environ["QUESTION_EXPERIMENT_NAME"] = args.experiment_name
        _set_experiment_name(args.experiment_name)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
