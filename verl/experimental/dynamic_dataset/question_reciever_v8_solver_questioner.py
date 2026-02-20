# Copyright 2026
#
# Licensed under the Apache License, Version 2.0.
"""
Async queue coordinator for dual-loop GRPO v8.

This is API-compatible with v6 receiver endpoints:
- POST /questioner/submit_and_wait
- GET  /solver/questions
- POST /solver/results
- GET  /health
- GET  /debug/state

Key changes vs v6:
- Fully async endpoints and wait paths (no threadpool blocking on long waits)
- Candidate completion is tracked with per-candidate asyncio.Event
- Shared state guarded by asyncio.Condition
- Trace logging is decoupled via a background writer thread so logging cannot
  block request handling
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

_cond = asyncio.Condition()
_candidate_queue: Deque[str] = deque()
_candidates: Dict[str, Dict[str, object]] = {}
_candidate_events: Dict[str, asyncio.Event] = {}
_active_waiters = 0

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


class QuestionerSubmitWaitPayload(BaseModel):
    split: Optional[str] = None
    batch_id: Optional[str] = None
    wait_s: Optional[float] = None
    candidates: List[QuestionerCandidateItem] = Field(default_factory=list)


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


async def _register_candidates(payload: QuestionerSubmitWaitPayload) -> tuple[List[str], List[dict[str, Any]]]:
    split_default = payload.split or "train"
    ids: List[str] = []
    trace_events: List[dict[str, Any]] = []

    async with _cond:
        if len(_candidates) >= _MAX_CANDIDATES:
            raise RuntimeError("Receiver candidate capacity reached.")

        for item in payload.candidates:
            candidate_id = str(item.candidate_id or uuid.uuid4().hex)
            existing = _candidates.get(candidate_id)
            if existing is not None:
                ids.append(candidate_id)
                continue

            split = item.split or split_default
            question = (item.question or "").strip()
            answer = (item.answer or "").strip()
            parse_ok = bool(item.parse_ok)

            rec = {
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
            }

            _candidates[candidate_id] = rec
            _candidate_events[candidate_id] = asyncio.Event()
            _candidate_queue.append(candidate_id)
            ids.append(candidate_id)

            trace_events.append(
                {
                    "event": "v8.receiver.candidate.registered",
                    "candidate_id": candidate_id,
                    "parse_ok": parse_ok,
                    "split": split,
                    "queue_size": len(_candidate_queue),
                    "question_len": len(question),
                    "answer_len": len(answer),
                    "question_empty": bool(not question),
                    "answer_empty": bool(not answer),
                    "has_numeric_solution": bool(rec.get("solution") is not None),
                    "question_preview": _preview(question) if _TRACE_INCLUDE_TEXT else None,
                }
            )

        _cond.notify_all()

    return ids, trace_events


async def _solver_take_questions(n: int, wait_s: float) -> tuple[List[dict[str, Any]], Optional[dict[str, Any]]]:
    deadline = None if wait_s < 0 else (time.monotonic() + max(0.0, wait_s))

    while True:
        async with _cond:
            if len(_candidate_queue) >= n:
                out: List[dict[str, Any]] = []
                now = _utc_now()
                assigned_ids: List[str] = []
                parse_ok_count = 0
                empty_question_count = 0
                empty_answer_count = 0

                for _ in range(n):
                    cid = _candidate_queue.popleft()
                    rec = _candidates[cid]
                    rec["status"] = "assigned"
                    rec["assigned_at"] = now
                    assigned_ids.append(cid)
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
                        }
                    )

                trace_payload = {
                    "event": "v8.receiver.queue.assigned",
                    "requested": n,
                    "assigned": len(out),
                    "queue_size_after": len(_candidate_queue),
                    "parse_ok_count": parse_ok_count,
                    "empty_question_count": empty_question_count,
                    "empty_answer_count": empty_answer_count,
                    "candidate_ids": assigned_ids,
                }
                return out, trace_payload

            # Not enough items available right now.
            if deadline is not None and time.monotonic() >= deadline:
                return [], None

        # Wait outside lock and retry.
        timeout = _WAIT_POLL_S
        if deadline is not None:
            timeout = min(_WAIT_POLL_S, max(0.0, deadline - time.monotonic()))
            if timeout <= 0:
                return [], None
        await asyncio.sleep(timeout)


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


async def _apply_solver_results(results: List[SolverResultItem]) -> List[dict[str, Any]]:
    trace_events: List[dict[str, Any]] = []
    async with _cond:
        for item in results:
            cid = str(item.candidate_id)
            if cid not in _candidates:
                raise KeyError(f"Unknown candidate_id: {cid}")
            rec = _candidates[cid]
            diff = _to_difficulty(item)
            rec["difficulty"] = diff
            rec["n"] = item.n
            rec["n_correct"] = item.n_correct
            rec["status"] = "solved"
            rec["resolved"] = True
            rec["solver_meta"] = item.meta
            rec["solved_at"] = _utc_now()
            ev = _candidate_events.get(cid)
            if ev is not None:
                ev.set()

            trace_events.append(
                {
                    "event": "v8.receiver.solver_result.applied",
                    "candidate_id": cid,
                    "difficulty": diff,
                    "n": item.n,
                    "n_correct": item.n_correct,
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
                }
            )
        return out


@app.get("/health")
async def health():
    async with _cond:
        status_counts = Counter(str(rec.get("status", "unknown")) for rec in _candidates.values())
        return {
            "ok": True,
            "solver_queue_size": len(_candidate_queue),
            "candidates_total": len(_candidates),
            "candidate_counts": dict(status_counts),
            "active_waiters": _active_waiters,
        }


@app.get("/debug/state")
async def debug_state():
    async with _cond:
        status_counts = Counter(str(rec.get("status", "unknown")) for rec in _candidates.values())
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
                    "question_preview": _preview(q) if _TRACE_INCLUDE_TEXT else None,
                }
            )
        return {
            "ok": True,
            "solver_queue_size": len(_candidate_queue),
            "candidates_total": len(_candidates),
            "candidate_counts": dict(status_counts),
            "active_waiters": _active_waiters,
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
        raise HTTPException(
            status_code=504,
            detail={
                "error": "solver_fetch_timeout",
                "needed": n,
                "received": len(items),
                "wait_s": wait_s,
                "queue_size": queue_size,
            },
        )

    if trace_payload is not None:
        name = trace_payload.pop("event")
        _trace(name, **trace_payload)

    _trace(
        "v8.receiver.solver_questions.served",
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
    parser = argparse.ArgumentParser(description="Dual-loop queue coordinator v8.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
