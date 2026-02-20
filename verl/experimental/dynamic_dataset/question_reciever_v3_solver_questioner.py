# Copyright 2026
#
# Licensed under the Apache License, Version 2.0.
"""
Coordinator for dual-loop GRPO (questioner + solver), in-memory and fail-fast.

Endpoints:
- GET  /questioner/seeds
- POST /questioner/candidates
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


class QuestionerCandidateItem(BaseModel):
    candidate_id: Optional[str] = None
    question: str
    answer: str
    parse_ok: bool = True
    split: Optional[str] = None
    seed_question_id: Optional[str] = None
    seed_question: Optional[str] = None
    seed_answer: Optional[str] = None
    seed_solution: Optional[str] = None
    seed_extra_info: Optional[dict] = None
    meta: Optional[dict] = None


class QuestionerCandidatesPayload(BaseModel):
    split: Optional[str] = None
    cycle_id: Optional[str] = None
    candidates: List[QuestionerCandidateItem] = Field(default_factory=list)


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


def _batch_log_path(root_dir: str, prefix: str, batch_id: Optional[int]) -> str:
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
        # Compact JSON keeps coordinator request handlers fast under high concurrency.
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


def _register_candidate(item: QuestionerCandidateItem, split_default: str, cycle_id: Optional[str]) -> dict:
    candidate_id = item.candidate_id or uuid.uuid4().hex
    split = item.split or split_default
    question = (item.question or "").strip()
    answer = (item.answer or "").strip()
    parse_ok = bool(item.parse_ok)
    solution = _extract_solution(answer) if parse_ok else None

    can_queue_solver = parse_ok and bool(question) and bool(answer) and solution is not None
    if not parse_ok:
        status = "unparseable"
        difficulty = 0.0
        resolved = True
    elif solution is None:
        status = "no_ground_truth"
        difficulty = 0.0
        resolved = True
    else:
        status = "queued"
        difficulty = None
        resolved = False

    now = _utc_now()
    with _cond:
        existing = _candidates.get(candidate_id)
        if existing is not None:
            _trace(
                "receiver.candidate.duplicate",
                candidate_id=candidate_id,
                status=existing.get("status"),
                split=existing.get("split"),
            )
            return existing

        record = {
            "candidate_id": candidate_id,
            "question": question,
            "answer": answer,
            "solution": solution,
            "split": split,
            "parse_ok": parse_ok,
            "status": status,
            "difficulty": difficulty,
            "resolved": resolved,
            "cycle_id": cycle_id,
            "created_at": now,
            "seed_question_id": item.seed_question_id,
            "seed_question": item.seed_question,
            "seed_answer": item.seed_answer,
            "seed_solution": item.seed_solution,
            "seed_extra_info": item.seed_extra_info,
            "meta": item.meta,
        }
        _candidates[candidate_id] = record
        if can_queue_solver:
            _candidate_queue.append(candidate_id)
        _cond.notify_all()
        _trace(
            "receiver.candidate.registered",
            candidate_id=candidate_id,
            status=status,
            split=split,
            cycle_id=cycle_id,
            parse_ok=parse_ok,
            has_solution=solution is not None,
            queued_for_solver=can_queue_solver,
            queue_size=len(_candidate_queue),
            question_len=len(question),
            answer_len=len(answer),
            question_preview=_preview(question) if _TRACE_INCLUDE_TEXT else None,
        )
        return record


def _solver_take_questions(n: int, wait_s: float) -> List[dict]:
    with _cond:
        deadline = None if wait_s < 0 else time.monotonic() + max(wait_s, 0.0)
        while len(_candidate_queue) < n and (deadline is None or time.monotonic() < deadline):
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
        taken: List[dict] = []
        for _ in range(take_n):
            cid = _candidate_queue.popleft()
            record = _candidates[cid]
            record["status"] = "assigned"
            record["assigned_at"] = now
            _trace(
                "receiver.solver_take.assigned",
                candidate_id=cid,
                split=record.get("split"),
                cycle_id=record.get("cycle_id"),
                queue_size_after_pop=len(_candidate_queue),
                question_preview=_preview(record.get("question")) if _TRACE_INCLUDE_TEXT else None,
            )
            taken.append(
                {
                    "candidate_id": cid,
                    "question": record["question"],
                    "answer": record["answer"],
                    "solution": record["solution"],
                    "split": record["split"],
                    "seed_question_id": record["seed_question_id"],
                    "cycle_id": record["cycle_id"],
                    "meta": record.get("meta"),
                }
            )
        return taken


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
            record = _candidates[item.candidate_id]
            difficulty = _to_difficulty(item)
            record["difficulty"] = difficulty
            record["resolved"] = True
            record["status"] = "solved"
            record["solved_at"] = _utc_now()
            record["solver_result"] = {
                "difficulty": difficulty,
                "n": item.n,
                "n_correct": item.n_correct,
                "correct": item.correct,
                "meta": item.meta,
            }
            _trace(
                "receiver.solver_result.applied",
                candidate_id=item.candidate_id,
                difficulty=difficulty,
                n=item.n,
                n_correct=item.n_correct,
                split=record.get("split"),
                cycle_id=record.get("cycle_id"),
            )
        _cond.notify_all()


def _log_questioner_candidates(
    split: str,
    cycle_id: Optional[str],
    accepted: int,
    queued: int,
    resolved_immediate: int,
    records: List[dict],
) -> None:
    answered_path = _batch_log_path(_ANSWERED_DIR, "questions_answered", cycle_id or "questioner")
    _write_jsonl(
        answered_path,
        {
            "source": "questioner_candidates",
            "split": split,
            "cycle_id": cycle_id,
            "received_at": _utc_now(),
            "accepted": accepted,
            "queued_for_solver": queued,
            "resolved_immediate": resolved_immediate,
            "responses": records,
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


def _await_candidate_rewards(candidate_ids: List[str], wait_s: float) -> List[dict]:
    with _cond:
        unknown = [cid for cid in candidate_ids if cid not in _candidates]
        if unknown:
            raise KeyError(f"Unknown candidate_ids: {unknown}")

        deadline = None if wait_s < 0 else time.monotonic() + max(wait_s, 0.0)

        def all_resolved() -> bool:
            return all(bool(_candidates[cid].get("resolved", False)) for cid in candidate_ids)

        while not all_resolved() and (deadline is None or time.monotonic() < deadline):
            remaining = None if deadline is None else max(deadline - time.monotonic(), 0.0)
            timeout = _SUBMITTER_POLL_S if remaining is None else min(_SUBMITTER_POLL_S, remaining)
            _cond.wait(timeout=timeout)

        if not all_resolved():
            unresolved = [cid for cid in candidate_ids if not bool(_candidates[cid].get("resolved", False))]
            if _FAIL_ON_TIMEOUT:
                _trace(
                    "receiver.await.timeout",
                    candidate_ids=candidate_ids,
                    unresolved=unresolved,
                    wait_s=wait_s,
                )
                raise TimeoutError(f"Timed out waiting for solver results. unresolved={unresolved}")

        out = []
        for cid in candidate_ids:
            rec = _candidates[cid]
            out.append(
                {
                    "candidate_id": cid,
                    "parse_ok": bool(rec.get("parse_ok", False)),
                    "difficulty": rec.get("difficulty"),
                    "status": rec.get("status"),
                    "resolved": bool(rec.get("resolved", False)),
                    "solution": rec.get("solution"),
                }
            )
        return out


@app.get("/health")
def health():
    with _lock:
        status_counts = Counter(rec.get("status", "unknown") for rec in _candidates.values())
        seed_queue_sizes = {split: len(q) for split, q in _seed_queues.items()}
        return {
            "ok": True,
            "fail_on_timeout": _FAIL_ON_TIMEOUT,
            "seed_queue_sizes": seed_queue_sizes,
            "solver_queue_size": len(_candidate_queue),
            "candidate_counts": dict(status_counts),
            "dataset_size": _dataset_size_by_split,
        }


@app.on_event("startup")
def _startup() -> None:
    _set_experiment_name(_EXPERIMENT_NAME)
    _ensure_dirs()


@app.get("/debug/state")
def debug_state():
    with _lock:
        status_counts = Counter(rec.get("status", "unknown") for rec in _candidates.values())
        return {
            "solver_queue_size": len(_candidate_queue),
            "candidates_total": len(_candidates),
            "candidate_counts": dict(status_counts),
            "sample_candidate_ids": list(_candidates.keys())[:20],
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


@app.post("/questioner/candidates")
def questioner_candidates(payload: QuestionerCandidatesPayload):
    if not payload.candidates:
        raise HTTPException(status_code=400, detail="No candidates provided.")

    split = payload.split or _DEFAULT_SPLIT
    accepted = 0
    queued = 0
    resolved_immediate = 0
    candidate_ids = []
    accepted_records: List[dict] = []

    for item in payload.candidates:
        rec = _register_candidate(item, split_default=split, cycle_id=payload.cycle_id)
        candidate_ids.append(rec["candidate_id"])
        accepted_records.append(rec)
        accepted += 1
        if rec.get("status") == "queued":
            queued += 1
        if rec.get("resolved"):
            resolved_immediate += 1

    if _LOG_QUESTIONER_CANDIDATES:
        _log_questioner_candidates(
            split=split,
            cycle_id=payload.cycle_id,
            accepted=accepted,
            queued=queued,
            resolved_immediate=resolved_immediate,
            records=accepted_records,
        )

    _trace(
        "receiver.questioner_candidates.accepted",
        split=split,
        cycle_id=payload.cycle_id,
        accepted=accepted,
        queued_for_solver=queued,
        resolved_immediate=resolved_immediate,
        queue_size=len(_candidate_queue),
    )

    return {
        "ok": True,
        "accepted": accepted,
        "queued_for_solver": queued,
        "resolved_immediate": resolved_immediate,
        "candidate_ids": candidate_ids,
    }


@app.post("/questioner/rewards/await")
def questioner_rewards_await(payload: QuestionerAwaitPayload):
    if not payload.candidate_ids:
        raise HTTPException(status_code=400, detail="No candidate_ids provided.")
    wait_s = _SOLVER_RESULT_WAIT_S if payload.wait_s is None else float(max(payload.wait_s, 0.0))

    # Non-blocking polling mode: return current state without 504s.
    # This keeps the endpoint cheap for frequent pollers (questioner reward loop).
    if wait_s <= 0:
        with _lock:
            unknown = [cid for cid in payload.candidate_ids if cid not in _candidates]
            if unknown:
                raise HTTPException(status_code=404, detail=f"Unknown candidate_ids: {unknown}")
            results = []
            for cid in payload.candidate_ids:
                rec = _candidates[cid]
                results.append(
                    {
                        "candidate_id": cid,
                        "parse_ok": bool(rec.get("parse_ok", False)),
                        "difficulty": rec.get("difficulty"),
                        "status": rec.get("status"),
                        "resolved": bool(rec.get("resolved", False)),
                        "solution": rec.get("solution"),
                    }
                )
        return {"ok": True, "results": results}

    try:
        results = _await_candidate_rewards(payload.candidate_ids, wait_s=wait_s)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except TimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail={
                "error": "solver_result_timeout",
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
    parser = argparse.ArgumentParser(description="Dual-loop coordinator (questioner + solver).")
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
