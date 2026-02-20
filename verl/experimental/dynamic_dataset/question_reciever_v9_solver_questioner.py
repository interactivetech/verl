# Copyright 2026
#
# Licensed under the Apache License, Version 2.0.
"""
Async queue coordinator for dual-loop GRPO v9 (multi-round self-play).

API-compatible endpoints:
- POST /questioner/submit_and_wait
- GET  /solver/questions
- POST /solver/results
- GET  /health
- GET  /debug/state

v9 key behavior:
- Candidate remains unresolved until `rounds_done >= target_rounds`
- Unresolved-first scheduling with lease-based reassignment safety
- Round idempotency via (candidate_id, round_id)
- Configurable final aggregation mode:
  - aggregate_all_rounds
  - final_round_only
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import queue as thread_queue
import re
import threading
import time
import uuid
from collections import Counter, deque
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

app = FastAPI()

_TRACE_ENABLED = os.getenv("DUAL_LOOP_TRACE", "false").strip().lower() in {"1", "true", "yes", "on"}
_TRACE_INCLUDE_TEXT = os.getenv("DUAL_LOOP_TRACE_INCLUDE_TEXT", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_TRACE_MAX_TEXT_CHARS = int(os.getenv("DUAL_LOOP_TRACE_MAX_TEXT_CHARS", "180"))
_TRACE_QUEUE_MAX = int(os.getenv("DUAL_LOOP_TRACE_QUEUE_MAX", "10000"))
_WAIT_POLL_S = float(os.getenv("DUAL_LOOP_WAIT_POLL_S", "0.25"))
_DEFAULT_SOLVER_FETCH_WAIT_S = float(os.getenv("SOLVER_FETCH_WAIT_S", "-1"))
_DEFAULT_QUESTIONER_WAIT_S = float(os.getenv("QUESTIONER_RESULT_WAIT_S", "-1"))
_MAX_CANDIDATES = int(os.getenv("DUAL_LOOP_MAX_CANDIDATES", "200000"))
_MAX_ACTIVE_WAITERS = int(os.getenv("DUAL_LOOP_MAX_ACTIVE_WAITERS", "4096"))

_MULTIROUND_TARGET_ROUNDS = max(1, int(os.getenv("MULTIROUND_TARGET_ROUNDS", "1")))
_MULTIROUND_LEASE_TTL_S = max(1.0, float(os.getenv("MULTIROUND_LEASE_TTL_S", "900")))
_MULTIROUND_MAX_UNRESOLVED = max(1, int(os.getenv("MULTIROUND_MAX_UNRESOLVED", "50000")))
_MULTIROUND_WARMUP_STEPS = max(0, int(os.getenv("MULTIROUND_WARMUP_STEPS", "0")))
_MULTIROUND_WARMUP_FORCE_SINGLE_ROUND = os.getenv("MULTIROUND_WARMUP_FORCE_SINGLE_ROUND", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_MULTIROUND_STRICT_IDEMPOTENCY = os.getenv("MULTIROUND_STRICT_IDEMPOTENCY", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_DEFAULT_AGGREGATION_MODE = os.getenv("MULTIROUND_AGGREGATION_MODE", "aggregate_all_rounds").strip().lower()
_ALLOWED_AGGREGATION_MODES = {"aggregate_all_rounds", "final_round_only"}
if _DEFAULT_AGGREGATION_MODE not in _ALLOWED_AGGREGATION_MODES:
    _DEFAULT_AGGREGATION_MODE = "aggregate_all_rounds"

_cond = asyncio.Condition()
_candidate_queue: Deque[str] = deque()
_candidates: Dict[str, Dict[str, object]] = {}
_candidate_events: Dict[str, asyncio.Event] = {}
_active_waiters = 0
_lease_expiry_requeues = 0
_questioner_submit_count = 0

_trace_queue: thread_queue.Queue[str] = thread_queue.Queue(maxsize=max(256, _TRACE_QUEUE_MAX))


def _trace_writer() -> None:
    while True:
        line = _trace_queue.get()
        if line == "__STOP__":
            return
        print(line, flush=True)


_trace_thread = threading.Thread(target=_trace_writer, daemon=True, name="dual-loop-trace-writer")
_trace_thread.start()


class QuestionerCandidateItem(BaseModel):
    candidate_id: Optional[str] = None
    question: str
    answer: str
    parse_ok: bool = False
    question_ok: bool = False
    answer_ok: bool = False
    numeric_final_ok: bool = False
    parse_mode: Optional[str] = None
    split: Optional[str] = None
    meta: Optional[dict] = None
    target_rounds: Optional[int] = None
    aggregation_mode: Optional[str] = None


class QuestionerSubmitWaitPayload(BaseModel):
    split: Optional[str] = None
    batch_id: Optional[str] = None
    wait_s: Optional[float] = None
    target_rounds: Optional[int] = None
    aggregation_mode: Optional[str] = None
    candidates: List[QuestionerCandidateItem] = Field(default_factory=list)


class SolverResultItem(BaseModel):
    candidate_id: str
    round_id: Optional[str] = None
    difficulty: Optional[float] = None
    n: Optional[int] = None
    n_correct: Optional[int] = None
    correct: Optional[bool] = None
    meta: Optional[dict] = None


class SolverResultsPayload(BaseModel):
    results: List[SolverResultItem] = Field(default_factory=list)


def _utc_now() -> str:
    return datetime.utcnow().isoformat() + "Z"


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
    payload.update(fields)
    line = f"[dual-loop-trace] {json.dumps(payload, ensure_ascii=False, separators=(',', ':'), default=str)}"
    try:
        _trace_queue.put_nowait(line)
    except thread_queue.Full:
        # Drop trace lines under pressure to keep coordinator responsive.
        pass


def _extract_solution(answer: str | None) -> Optional[str]:
    if not answer:
        return None
    match = re.search(r"####\s*(\-?[0-9][0-9,\.]*)", answer)
    if not match:
        return None
    return match.group(1).replace(",", "")


def _sanitize_target_rounds(raw_value: Optional[int]) -> int:
    if raw_value is None:
        return _MULTIROUND_TARGET_ROUNDS
    try:
        return max(1, int(raw_value))
    except Exception as exc:
        raise ValueError(f"Invalid target_rounds={raw_value}") from exc


def _sanitize_aggregation_mode(raw_value: Optional[str]) -> str:
    mode = str(raw_value or _DEFAULT_AGGREGATION_MODE).strip().lower()
    if mode not in _ALLOWED_AGGREGATION_MODES:
        raise ValueError(
            f"Invalid aggregation_mode={raw_value}. Allowed: {sorted(_ALLOWED_AGGREGATION_MODES)}"
        )
    return mode


def _unresolved_count_locked() -> int:
    return sum(1 for rec in _candidates.values() if not bool(rec.get("resolved", False)))


def _ensure_in_queue_locked(candidate_id: str, front: bool = False) -> None:
    rec = _candidates.get(candidate_id)
    if rec is None:
        return
    if bool(rec.get("resolved", False)):
        rec["_in_queue"] = False
        return
    if bool(rec.get("_in_queue", False)):
        return
    if front:
        _candidate_queue.appendleft(candidate_id)
    else:
        _candidate_queue.append(candidate_id)
    rec["_in_queue"] = True


def _requeue_expired_leases_locked(now_monotonic: float) -> list[str]:
    global _lease_expiry_requeues
    expired: list[str] = []
    for cid, rec in _candidates.items():
        if bool(rec.get("resolved", False)):
            continue
        if str(rec.get("status")) != "assigned":
            continue
        lease_until = float(rec.get("lease_until", 0.0) or 0.0)
        if lease_until <= now_monotonic:
            rec["status"] = "queued"
            rec["lease_owner"] = None
            rec["lease_until"] = 0.0
            rec["last_lease_expired_at"] = _utc_now()
            rec["lease_expired_count"] = int(rec.get("lease_expired_count", 0)) + 1
            _ensure_in_queue_locked(cid, front=True)
            expired.append(cid)
            _lease_expiry_requeues += 1
    return expired


def _extract_round_id(item: SolverResultItem, rec: dict[str, object]) -> str:
    from_item = item.round_id
    from_meta = None
    if isinstance(item.meta, dict):
        from_meta = item.meta.get("round_id") or item.meta.get("assignment_id")

    raw_round_id = from_item if from_item not in (None, "") else from_meta
    if raw_round_id in (None, ""):
        raw_round_id = rec.get("lease_owner")

    if raw_round_id in (None, ""):
        if _MULTIROUND_STRICT_IDEMPOTENCY:
            raise ValueError(
                f"Missing round_id for candidate_id={item.candidate_id}. "
                "Provide round_id or meta.round_id/assignment_id."
            )
        raw_round_id = f"fallback:{uuid.uuid4().hex}"

    return str(raw_round_id)


def _to_difficulty(item: SolverResultItem) -> float:
    if item.difficulty is not None:
        value = float(item.difficulty)
    elif item.n is not None and item.n_correct is not None:
        n = max(1, int(item.n))
        value = float(item.n_correct) / float(n)
    elif item.correct is not None:
        value = 1.0 if bool(item.correct) else 0.0
    else:
        raise ValueError("Missing difficulty signal. Provide one of difficulty or (n, n_correct) or correct.")
    return max(0.0, min(1.0, value))


def _finalize_candidate_locked(rec: dict[str, object]) -> None:
    mode = str(rec.get("aggregation_mode") or _DEFAULT_AGGREGATION_MODE)

    last_difficulty = rec.get("last_difficulty")
    if last_difficulty is None:
        last_difficulty = 0.0

    if mode == "final_round_only":
        final_difficulty = float(last_difficulty)
        final_n = rec.get("last_n")
        final_n_correct = rec.get("last_n_correct")
        if final_n is None:
            final_n = int(rec.get("agg_round_count", 0) or 0)
        if final_n_correct is None and final_n is not None:
            final_n_correct = int(round(float(final_difficulty) * max(1, int(final_n))))
    else:
        agg_n = int(rec.get("agg_n", 0) or 0)
        agg_n_correct = int(rec.get("agg_n_correct", 0) or 0)
        agg_round_count = int(rec.get("agg_round_count", 0) or 0)
        agg_diff_sum = float(rec.get("agg_diff_sum", 0.0) or 0.0)

        if agg_n > 0:
            final_difficulty = float(max(0.0, min(1.0, agg_n_correct / max(1, agg_n))))
            final_n = agg_n
            final_n_correct = agg_n_correct
        elif agg_round_count > 0:
            final_difficulty = float(max(0.0, min(1.0, agg_diff_sum / max(1, agg_round_count))))
            final_n = agg_round_count
            final_n_correct = int(round(final_difficulty * max(1, agg_round_count)))
        else:
            final_difficulty = float(last_difficulty)
            final_n = 1
            final_n_correct = int(round(final_difficulty))

    rec["difficulty"] = float(max(0.0, min(1.0, final_difficulty)))
    rec["n"] = int(final_n) if final_n is not None else None
    rec["n_correct"] = int(final_n_correct) if final_n_correct is not None else None
    rec["status"] = "solved"
    rec["resolved"] = True
    rec["solved_at"] = _utc_now()


async def _register_candidates(payload: QuestionerSubmitWaitPayload) -> tuple[List[str], List[dict[str, Any]]]:
    global _questioner_submit_count
    split_default = payload.split or "train"
    payload_target_rounds = _sanitize_target_rounds(payload.target_rounds)
    payload_aggregation_mode = _sanitize_aggregation_mode(payload.aggregation_mode)

    ids: List[str] = []
    trace_events: List[dict[str, Any]] = []

    async with _cond:
        _questioner_submit_count += 1
        submit_index = _questioner_submit_count
        warmup_active = _MULTIROUND_WARMUP_STEPS > 0 and submit_index <= _MULTIROUND_WARMUP_STEPS
        warmup_remaining = max(0, _MULTIROUND_WARMUP_STEPS - submit_index)

        if len(_candidates) >= _MAX_CANDIDATES:
            raise RuntimeError("Receiver candidate capacity reached.")
        unresolved_count = _unresolved_count_locked()

        for item in payload.candidates:
            candidate_id = str(item.candidate_id or uuid.uuid4().hex)
            existing = _candidates.get(candidate_id)
            if existing is not None:
                ids.append(candidate_id)
                continue

            if unresolved_count >= _MULTIROUND_MAX_UNRESOLVED:
                raise RuntimeError(
                    "Receiver unresolved candidate cap reached. "
                    f"unresolved={unresolved_count} max={_MULTIROUND_MAX_UNRESOLVED}"
                )

            split = item.split or split_default
            question = (item.question or "").strip()
            answer = (item.answer or "").strip()
            parse_ok = bool(item.parse_ok)
            target_rounds = _sanitize_target_rounds(item.target_rounds if item.target_rounds is not None else payload_target_rounds)
            if warmup_active and _MULTIROUND_WARMUP_FORCE_SINGLE_ROUND:
                target_rounds = 1
            aggregation_mode = _sanitize_aggregation_mode(
                item.aggregation_mode if item.aggregation_mode is not None else payload_aggregation_mode
            )

            rec: dict[str, object] = {
                "candidate_id": candidate_id,
                "split": split,
                "batch_id": payload.batch_id,
                "question": question,
                "answer": answer,
                "solution": _extract_solution(answer),
                "parse_ok": parse_ok,
                "question_ok": bool(item.question_ok),
                "answer_ok": bool(item.answer_ok),
                "numeric_final_ok": bool(item.numeric_final_ok),
                "parse_mode": item.parse_mode,
                "meta": item.meta,
                "status": "queued",
                "resolved": False,
                "difficulty": None,
                "n": None,
                "n_correct": None,
                "created_at": _utc_now(),
                "target_rounds": target_rounds,
                "aggregation_mode": aggregation_mode,
                "rounds_done": 0,
                "round_ids_seen": set(),
                "duplicate_round_posts": 0,
                "agg_n": 0,
                "agg_n_correct": 0,
                "agg_round_count": 0,
                "agg_diff_sum": 0.0,
                "last_round_id": None,
                "last_difficulty": None,
                "last_n": None,
                "last_n_correct": None,
                "lease_owner": None,
                "lease_until": 0.0,
                "lease_expired_count": 0,
                "warmup_applied": bool(warmup_active and _MULTIROUND_WARMUP_FORCE_SINGLE_ROUND),
                "_in_queue": False,
            }

            _candidates[candidate_id] = rec
            _candidate_events[candidate_id] = asyncio.Event()
            _ensure_in_queue_locked(candidate_id, front=False)
            unresolved_count += 1
            ids.append(candidate_id)

            trace_events.append(
                {
                    "event": "v9.receiver.candidate.registered",
                    "candidate_id": candidate_id,
                    "parse_ok": parse_ok,
                    "split": split,
                    "queue_size": len(_candidate_queue),
                    "question_len": len(question),
                    "answer_len": len(answer),
                    "question_empty": bool(not question),
                    "answer_empty": bool(not answer),
                    "has_numeric_solution": bool(rec.get("solution") is not None),
                    "target_rounds": target_rounds,
                    "aggregation_mode": aggregation_mode,
                    "submit_index": submit_index,
                    "warmup_active": warmup_active,
                    "warmup_remaining": warmup_remaining,
                    "question_preview": _preview(question) if _TRACE_INCLUDE_TEXT else None,
                }
            )

        _cond.notify_all()

    return ids, trace_events


async def _solver_take_questions(n: int, wait_s: float) -> tuple[List[dict[str, Any]], Optional[dict[str, Any]]]:
    deadline = None if wait_s < 0 else (time.monotonic() + max(0.0, wait_s))

    while True:
        async with _cond:
            now_monotonic = time.monotonic()
            expired_ids = _requeue_expired_leases_locked(now_monotonic)
            if expired_ids:
                _trace(
                    "v9.receiver.lease.expired_requeue",
                    expired_count=len(expired_ids),
                    expired_candidate_ids=expired_ids[:50],
                    queue_size=len(_candidate_queue),
                )

            assignment_id = uuid.uuid4().hex
            provisional_ids: list[str] = []
            out: List[dict[str, Any]] = []
            parse_ok_count = 0
            empty_question_count = 0
            empty_answer_count = 0

            while len(out) < n and _candidate_queue:
                cid = _candidate_queue.popleft()
                rec = _candidates.get(cid)
                if rec is None:
                    continue
                rec["_in_queue"] = False

                if bool(rec.get("resolved", False)):
                    continue

                if str(rec.get("status")) == "assigned":
                    continue

                rec["status"] = "assigned"
                rec["assigned_at"] = _utc_now()
                rec["lease_owner"] = assignment_id
                rec["lease_until"] = now_monotonic + _MULTIROUND_LEASE_TTL_S
                provisional_ids.append(cid)

                if bool(rec.get("parse_ok", False)):
                    parse_ok_count += 1
                if not str(rec.get("question") or "").strip():
                    empty_question_count += 1
                if not str(rec.get("answer") or "").strip():
                    empty_answer_count += 1

                out.append(
                    {
                        "candidate_id": cid,
                        "question": rec.get("question"),
                        "answer": rec.get("answer"),
                        "solution": rec.get("solution"),
                        "split": rec.get("split"),
                        "parse_ok": bool(rec.get("parse_ok", False)),
                        "meta": rec.get("meta"),
                        "target_rounds": int(rec.get("target_rounds", 1) or 1),
                        "rounds_done": int(rec.get("rounds_done", 0) or 0),
                        "aggregation_mode": str(rec.get("aggregation_mode") or _DEFAULT_AGGREGATION_MODE),
                        "assignment_id": assignment_id,
                    }
                )

            if len(out) >= n:
                trace_payload = {
                    "event": "v9.receiver.queue.assigned",
                    "requested": n,
                    "assigned": len(out),
                    "queue_size_after": len(_candidate_queue),
                    "parse_ok_count": parse_ok_count,
                    "empty_question_count": empty_question_count,
                    "empty_answer_count": empty_answer_count,
                    "candidate_ids": provisional_ids,
                    "assignment_id": assignment_id,
                }
                return out, trace_payload

            # Roll back partial assignment so endpoint remains all-or-nothing.
            if provisional_ids:
                for cid in provisional_ids:
                    rec = _candidates.get(cid)
                    if rec is None or bool(rec.get("resolved", False)):
                        continue
                    rec["status"] = "queued"
                    rec["lease_owner"] = None
                    rec["lease_until"] = 0.0
                    _ensure_in_queue_locked(cid, front=True)

            if deadline is not None and time.monotonic() >= deadline:
                return [], None

        timeout = _WAIT_POLL_S
        if deadline is not None:
            timeout = min(_WAIT_POLL_S, max(0.0, deadline - time.monotonic()))
            if timeout <= 0:
                return [], None
        await asyncio.sleep(timeout)


async def _apply_solver_results(results: List[SolverResultItem]) -> List[dict[str, Any]]:
    trace_events: List[dict[str, Any]] = []
    async with _cond:
        for item in results:
            cid = str(item.candidate_id)
            rec = _candidates.get(cid)
            if rec is None:
                raise KeyError(f"Unknown candidate_id: {cid}")

            if bool(rec.get("resolved", False)):
                trace_events.append(
                    {
                        "event": "v9.receiver.solver_result.ignored_resolved",
                        "candidate_id": cid,
                        "round_id": item.round_id,
                    }
                )
                continue

            round_id = _extract_round_id(item, rec)
            round_ids_seen = rec.get("round_ids_seen")
            if not isinstance(round_ids_seen, set):
                round_ids_seen = set()
                rec["round_ids_seen"] = round_ids_seen

            if round_id in round_ids_seen:
                rec["duplicate_round_posts"] = int(rec.get("duplicate_round_posts", 0)) + 1
                trace_events.append(
                    {
                        "event": "v9.receiver.solver_result.duplicate_ignored",
                        "candidate_id": cid,
                        "round_id": round_id,
                        "duplicate_round_posts": int(rec.get("duplicate_round_posts", 0)),
                    }
                )
                continue

            round_ids_seen.add(round_id)

            diff = _to_difficulty(item)
            n_val = int(item.n) if item.n is not None else None
            n_correct_val = int(item.n_correct) if item.n_correct is not None else None
            if n_val is not None:
                n_val = max(1, n_val)
                if n_correct_val is None and item.correct is not None:
                    n_correct_val = 1 if bool(item.correct) else 0
                if n_correct_val is not None:
                    n_correct_val = max(0, min(n_val, n_correct_val))

            rec["last_round_id"] = round_id
            rec["last_difficulty"] = diff
            rec["last_n"] = n_val
            rec["last_n_correct"] = n_correct_val
            rec["solver_meta"] = item.meta

            rec["agg_round_count"] = int(rec.get("agg_round_count", 0)) + 1
            rec["agg_diff_sum"] = float(rec.get("agg_diff_sum", 0.0) or 0.0) + float(diff)
            if n_val is not None and n_correct_val is not None:
                rec["agg_n"] = int(rec.get("agg_n", 0)) + int(n_val)
                rec["agg_n_correct"] = int(rec.get("agg_n_correct", 0)) + int(n_correct_val)

            rec["rounds_done"] = int(rec.get("rounds_done", 0)) + 1
            rec["status"] = "queued"
            rec["lease_owner"] = None
            rec["lease_until"] = 0.0

            rounds_done = int(rec.get("rounds_done", 0))
            target_rounds = int(rec.get("target_rounds", 1) or 1)
            if rounds_done >= target_rounds:
                _finalize_candidate_locked(rec)
                ev = _candidate_events.get(cid)
                if ev is not None:
                    ev.set()
                trace_events.append(
                    {
                        "event": "v9.receiver.solver_result.finalized",
                        "candidate_id": cid,
                        "round_id": round_id,
                        "rounds_done": rounds_done,
                        "target_rounds": target_rounds,
                        "difficulty": float(rec.get("difficulty", 0.0) or 0.0),
                        "aggregation_mode": rec.get("aggregation_mode"),
                    }
                )
            else:
                # unresolved-first scheduling
                _ensure_in_queue_locked(cid, front=True)
                trace_events.append(
                    {
                        "event": "v9.receiver.solver_result.round_applied",
                        "candidate_id": cid,
                        "round_id": round_id,
                        "rounds_done": rounds_done,
                        "target_rounds": target_rounds,
                        "difficulty": diff,
                        "queue_size": len(_candidate_queue),
                    }
                )

        _cond.notify_all()
    return trace_events


async def _await_candidates(candidate_ids: List[str], wait_s: float) -> List[dict]:
    global _active_waiters
    async with _cond:
        unknown = [cid for cid in candidate_ids if cid not in _candidates]
        if unknown:
            raise KeyError(f"Unknown candidate_ids: {unknown}")
        if _active_waiters >= _MAX_ACTIVE_WAITERS:
            raise OverflowError("Too many concurrent waiters.")
        _active_waiters += 1
        events = [_candidate_events[cid] for cid in candidate_ids]

    try:
        wait_coro = asyncio.gather(*(ev.wait() for ev in events))
        if wait_s < 0:
            await wait_coro
        else:
            await asyncio.wait_for(wait_coro, timeout=max(0.0, float(wait_s)))
    finally:
        async with _cond:
            _active_waiters = max(0, _active_waiters - 1)

    async with _cond:
        out = []
        for cid in candidate_ids:
            rec = _candidates[cid]
            out.append(
                {
                    "candidate_id": cid,
                    "parse_ok": bool(rec.get("parse_ok", False)),
                    "question_ok": bool(rec.get("question_ok", False)),
                    "answer_ok": bool(rec.get("answer_ok", False)),
                    "numeric_final_ok": bool(rec.get("numeric_final_ok", False)),
                    "difficulty": rec.get("difficulty"),
                    "n": rec.get("n"),
                    "n_correct": rec.get("n_correct"),
                    "status": rec.get("status"),
                    "resolved": bool(rec.get("resolved", False)),
                    "rounds_done": int(rec.get("rounds_done", 0) or 0),
                    "target_rounds": int(rec.get("target_rounds", 1) or 1),
                    "aggregation_mode": str(rec.get("aggregation_mode") or _DEFAULT_AGGREGATION_MODE),
                    "last_round_id": rec.get("last_round_id"),
                    "duplicate_round_posts": int(rec.get("duplicate_round_posts", 0) or 0),
                }
            )
        return out


@app.get("/health")
async def health():
    async with _cond:
        _requeue_expired_leases_locked(time.monotonic())
        status_counts = Counter(str(rec.get("status", "unknown")) for rec in _candidates.values())
        unresolved_total = _unresolved_count_locked()
        resolved_total = len(_candidates) - unresolved_total
        return {
            "ok": True,
            "solver_queue_size": len(_candidate_queue),
            "candidates_total": len(_candidates),
            "unresolved_total": unresolved_total,
            "resolved_total": resolved_total,
            "candidate_counts": dict(status_counts),
            "active_waiters": _active_waiters,
            "lease_expiry_requeues": _lease_expiry_requeues,
            "questioner_submit_count": _questioner_submit_count,
            "warmup_steps": _MULTIROUND_WARMUP_STEPS,
            "warmup_active": bool(
                _MULTIROUND_WARMUP_STEPS > 0 and _questioner_submit_count < _MULTIROUND_WARMUP_STEPS
            ),
        }


@app.get("/debug/state")
async def debug_state():
    async with _cond:
        _requeue_expired_leases_locked(time.monotonic())
        status_counts = Counter(str(rec.get("status", "unknown")) for rec in _candidates.values())
        unresolved_total = _unresolved_count_locked()
        resolved_total = len(_candidates) - unresolved_total

        sample_candidates = []
        for cid, rec in list(_candidates.items())[:20]:
            q = str(rec.get("question") or "")
            a = str(rec.get("answer") or "")
            sample_candidates.append(
                {
                    "candidate_id": cid,
                    "status": rec.get("status"),
                    "resolved": bool(rec.get("resolved", False)),
                    "parse_ok": bool(rec.get("parse_ok", False)),
                    "question_len": len(q),
                    "answer_len": len(a),
                    "question_empty": bool(not q.strip()),
                    "answer_empty": bool(not a.strip()),
                    "target_rounds": int(rec.get("target_rounds", 1) or 1),
                    "rounds_done": int(rec.get("rounds_done", 0) or 0),
                    "aggregation_mode": rec.get("aggregation_mode"),
                    "warmup_applied": bool(rec.get("warmup_applied", False)),
                    "lease_owner": rec.get("lease_owner"),
                    "lease_until": rec.get("lease_until"),
                    "question_preview": _preview(q) if _TRACE_INCLUDE_TEXT else None,
                }
            )

        return {
            "ok": True,
            "solver_queue_size": len(_candidate_queue),
            "candidates_total": len(_candidates),
            "unresolved_total": unresolved_total,
            "resolved_total": resolved_total,
            "candidate_counts": dict(status_counts),
            "active_waiters": _active_waiters,
            "lease_expiry_requeues": _lease_expiry_requeues,
            "questioner_submit_count": _questioner_submit_count,
            "warmup_steps": _MULTIROUND_WARMUP_STEPS,
            "warmup_active": bool(
                _MULTIROUND_WARMUP_STEPS > 0 and _questioner_submit_count < _MULTIROUND_WARMUP_STEPS
            ),
            "sample_candidate_ids": list(_candidates.keys())[:20],
            "sample_candidates": sample_candidates,
        }


@app.post("/questioner/submit_and_wait")
async def questioner_submit_and_wait(payload: QuestionerSubmitWaitPayload):
    if not payload.candidates:
        raise HTTPException(status_code=400, detail="No candidates provided.")

    wait_s = _DEFAULT_QUESTIONER_WAIT_S if payload.wait_s is None else float(payload.wait_s)

    try:
        ids, trace_events = await _register_candidates(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    for event in trace_events:
        name = event.pop("event")
        _trace(name, **event)

    try:
        results = await _await_candidates(ids, wait_s=wait_s)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except OverflowError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except asyncio.TimeoutError as exc:
        # Fail-hard timeout policy: caller should treat this as training failure.
        raise HTTPException(
            status_code=504,
            detail={
                "error": "solver_result_timeout",
                "wait_s": wait_s,
                "candidate_ids": ids,
                "message": str(exc),
            },
        ) from exc

    return {"ok": True, "count": len(results), "results": results}


@app.get("/solver/questions")
async def solver_questions(
    n: int = Query(8, ge=1, le=4096),
    wait_s: Optional[float] = Query(None),
):
    wait_s = _DEFAULT_SOLVER_FETCH_WAIT_S if wait_s is None else float(wait_s)
    items, trace_payload = await _solver_take_questions(n=n, wait_s=wait_s)
    if len(items) < n:
        async with _cond:
            queue_size = len(_candidate_queue)
            unresolved_total = _unresolved_count_locked()
        raise HTTPException(
            status_code=504,
            detail={
                "error": "solver_fetch_timeout",
                "needed": n,
                "received": len(items),
                "wait_s": wait_s,
                "queue_size": queue_size,
                "unresolved_total": unresolved_total,
            },
        )

    if trace_payload is not None:
        name = trace_payload.pop("event")
        _trace(name, **trace_payload)

    _trace(
        "v9.receiver.solver_questions.served",
        requested=n,
        served=len(items),
        wait_s=wait_s,
        candidate_ids=[it.get("candidate_id") for it in items if isinstance(it, dict)],
        parse_ok_count=sum(int(bool(it.get("parse_ok", False))) for it in items if isinstance(it, dict)),
        empty_question_count=sum(int(not str(it.get("question") or "").strip()) for it in items if isinstance(it, dict)),
        empty_answer_count=sum(int(not str(it.get("answer") or "").strip()) for it in items if isinstance(it, dict)),
    )
    return {"ok": True, "count": len(items), "questions": items}


@app.post("/solver/results")
async def solver_results(payload: SolverResultsPayload):
    if not payload.results:
        raise HTTPException(status_code=400, detail="No solver results provided.")
    try:
        trace_events = await _apply_solver_results(payload.results)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    for event in trace_events:
        name = event.pop("event")
        _trace(name, **event)
    return {"ok": True, "count": len(payload.results)}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dual-loop queue coordinator v9 (multi-round self-play).")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
