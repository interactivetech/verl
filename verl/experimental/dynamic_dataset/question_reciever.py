# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
QuestionReceiver: accepts GSM8K question batches and serves them to HttpQuestionGenerator.

Endpoints:
  POST /submit
    {"batch_id": 0, "split": "train", "questions": [{"question": "...", "answer": "...", "index": 0, "split": "train"}]}

  GET /questions?n=8&split=train
    -> {"ids": [...], "questions": [...], "split": "train"}

  POST /responses
    {"batch_id": 0, "split": "train", "responses": [...]}  # stored verbatim

Environment variables:
  GSM8K_SPLIT:        Default split when not provided (default: train)
  QUESTION_LOG_ROOT:  Base directory for logs (default: /workspace/verl/verl)
  QUESTION_EXPERIMENT_NAME: Optional experiment name for log subfolder (default: unset)
  QUESTION_QUEUE_WAIT_S: Seconds to wait for questions before returning (default: -1, wait forever)
  QUESTION_SUBMITTER_URL: URL to fetch questions on demand (optional)
  QUESTION_SUBMITTER_TIMEOUT_S: HTTP timeout when calling submitter (default: 10)
  QUESTION_SUBMITTER_POLL_S: Seconds between retry attempts (default: 1)
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from collections import deque
from datetime import datetime
from typing import Deque, Dict, List, Optional
import uuid

import requests
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

app = FastAPI()

_DEFAULT_SPLIT = os.getenv("GSM8K_SPLIT", "train")
_LOG_ROOT = os.getenv("QUESTION_LOG_ROOT", "/workspace/verl/verl")
_EXPERIMENT_NAME = os.getenv("QUESTION_EXPERIMENT_NAME")
_ASKED_DIR = ""
_ANSWERED_DIR = ""
_QUEUE_WAIT_S = float(os.getenv("QUESTION_QUEUE_WAIT_S", "-1"))
_SUBMITTER_URL = os.getenv("QUESTION_SUBMITTER_URL")
_SUBMITTER_TIMEOUT_S = float(os.getenv("QUESTION_SUBMITTER_TIMEOUT_S", "10"))
_SUBMITTER_POLL_S = float(os.getenv("QUESTION_SUBMITTER_POLL_S", "1"))

_queue_lock = threading.Lock()
_queue_cond = threading.Condition(_queue_lock)
_queues: Dict[str, Deque[Dict[str, object]]] = {}
_received_batch_lock = threading.Lock()
_served_batch_lock = threading.Lock()
_received_batch_id = 0
_served_batch_id = 0


class QuestionItem(BaseModel):
    question: Optional[str] = None
    answer: Optional[str] = None
    solution: Optional[str] = None
    index: Optional[int] = None
    split: Optional[str] = None


class SubmitPayload(BaseModel):
    batch_id: Optional[int] = None
    split: Optional[str] = None
    questions: List[QuestionItem]


class ResponsesPayload(BaseModel):
    batch_id: Optional[int] = None
    split: Optional[str] = None
    responses: List[dict]
    question_metrics: Optional[List[dict]] = None


def _ensure_dirs() -> None:
    os.makedirs(_ASKED_DIR, exist_ok=True)
    os.makedirs(_ANSWERED_DIR, exist_ok=True)


def _resolve_log_dirs(experiment_name: Optional[str]) -> tuple[str, str]:
    if experiment_name:
        root = os.path.join(_LOG_ROOT, f"{experiment_name}_questions_asked_answered")
        return os.path.join(root, "questions_asked"), os.path.join(root, "questions_answered")
    return os.path.join(_LOG_ROOT, "questions_asked"), os.path.join(_LOG_ROOT, "questions_answered")


def _set_experiment_name(experiment_name: Optional[str]) -> None:
    global _EXPERIMENT_NAME, _ASKED_DIR, _ANSWERED_DIR
    _EXPERIMENT_NAME = experiment_name
    _ASKED_DIR, _ANSWERED_DIR = _resolve_log_dirs(experiment_name)


def _batch_log_path(root_dir: str, prefix: str, batch_id: Optional[int]) -> str:
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
    batch_label = "none" if batch_id is None else str(batch_id)
    token = uuid.uuid4().hex[:8]
    filename = f"{prefix}_batch_{batch_label}_{ts}_{token}.jsonl"
    return os.path.join(root_dir, filename)


def _write_jsonl(path: str, record: dict) -> None:
    _ensure_dirs()
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(record, indent=2, sort_keys=True) + "\n")


def _get_queue(split: str) -> Deque[Dict[str, object]]:
    if split not in _queues:
        _queues[split] = deque()
    return _queues[split]


def _next_received_batch_id() -> int:
    global _received_batch_id
    with _received_batch_lock:
        _received_batch_id += 1
        return _received_batch_id


def _next_served_batch_id() -> int:
    global _served_batch_id
    with _served_batch_lock:
        _served_batch_id += 1
        return _served_batch_id


def _enqueue(split: str, items: List[Dict[str, object]]) -> None:
    with _queue_cond:
        q = _get_queue(split)
        q.extend(items)
        _queue_cond.notify_all()


def _dequeue(split: str, n: int, wait_s: float) -> List[Dict[str, object]]:
    with _queue_cond:
        q = _get_queue(split)
        if wait_s < 0:
            end = None
        else:
            end = time.monotonic() + max(wait_s, 0.0)
        while len(q) < n and (end is None or time.monotonic() < end):
            remaining = None if end is None else max(end - time.monotonic(), 0.0)
            timeout = _SUBMITTER_POLL_S if remaining is None else min(_SUBMITTER_POLL_S, remaining)
            _queue_cond.wait(timeout=timeout)
        take_n = min(n, len(q))
        return [q.popleft() for _ in range(take_n)]


def _fetch_from_submitter(split: str, n: int) -> int:
    if not _SUBMITTER_URL or n <= 0:
        return 0
    try:
        r = requests.get(_SUBMITTER_URL, params={"n": n, "split": split}, timeout=_SUBMITTER_TIMEOUT_S)
        r.raise_for_status()
        payload = r.json()
        items = payload.get("questions") or []
    except Exception:
        return 0

    if not items:
        return 0

    received_at = datetime.utcnow().isoformat() + "Z"
    batch_id = _next_received_batch_id()
    for item in items:
        if item.get("batch_id") is None:
            item["batch_id"] = batch_id
        item.setdefault("split", split)
    _enqueue(split, items)
    asked_path = _batch_log_path(_ASKED_DIR, "questions_asked", batch_id)
    _write_jsonl(
        asked_path,
        {
            "batch_id": batch_id,
            "split": split,
            "received_at": received_at,
            "source": "submitter",
            "questions": items,
        },
    )
    return len(items)


def _get_questions_with_refill(split: str, n: int, wait_s: float) -> List[Dict[str, object]]:
    deadline = None if wait_s < 0 else time.monotonic() + max(wait_s, 0.0)

    while True:
        with _queue_lock:
            qsize = len(_get_queue(split))
        if qsize >= n:
            break

        needed = n - qsize
        fetched = _fetch_from_submitter(split, needed)
        if fetched > 0:
            continue

        if wait_s == 0:
            break
        if deadline is not None and time.monotonic() >= deadline:
            break
        time.sleep(_SUBMITTER_POLL_S)

    return _dequeue(split, n, wait_s)


@app.on_event("startup")
def _startup() -> None:
    _set_experiment_name(_EXPERIMENT_NAME)
    _ensure_dirs()


@app.get("/health")
def health():
    with _queue_lock:
        sizes = {split: len(q) for split, q in _queues.items()}
    return {"ok": True, "queues": sizes, "log_root": _LOG_ROOT}


@app.post("/submit")
def submit(payload: SubmitPayload):
    if not payload.questions:
        raise HTTPException(status_code=400, detail="No questions provided.")

    split = payload.split or _DEFAULT_SPLIT
    received_at = datetime.utcnow().isoformat() + "Z"
    batch_id = payload.batch_id if payload.batch_id is not None else _next_received_batch_id()

    items: List[Dict[str, object]] = []
    for q in payload.questions:
        q_split = q.split or split
        if not q.question:
            continue
        items.append(
            {
                "question": q.question,
                "answer": q.answer,
                "solution": q.solution,
                "index": q.index,
                "split": q_split,
                "batch_id": batch_id,
                "received_at": received_at,
            }
        )

    if not items:
        raise HTTPException(status_code=400, detail="No valid questions in payload.")

    _enqueue(split, items)
    asked_path = _batch_log_path(_ASKED_DIR, "questions_asked", batch_id)
    _write_jsonl(
        asked_path,
        {
            "batch_id": batch_id,
            "split": split,
            "received_at": received_at,
            "source": "submit",
            "questions": items,
        },
    )

    return {"ok": True, "count": len(items), "split": split, "batch_id": batch_id}


@app.get("/questions")
def questions(
    n: int = Query(8, ge=1, le=4096),
    split: Optional[str] = Query(None),
    wait_s: Optional[float] = Query(None),
):
    split = split or _DEFAULT_SPLIT
    wait_s = _QUEUE_WAIT_S if wait_s is None else max(wait_s, 0.0)

    items = _get_questions_with_refill(split, n, wait_s)
    if not items:
        raise HTTPException(status_code=503, detail="No questions available.")

    served_batch_id = _next_served_batch_id()
    served_at = datetime.utcnow().isoformat() + "Z"
    for item in items:
        if item.get("batch_id") is not None:
            item.setdefault("source_batch_id", item.get("batch_id"))
        item["batch_id"] = served_batch_id
        item["served_at"] = served_at

    asked_path = _batch_log_path(_ASKED_DIR, "questions_asked", served_batch_id)
    _write_jsonl(
        asked_path,
        {
            "batch_id": served_batch_id,
            "split": split,
            "served_at": served_at,
            "source": "serve",
            "questions": items,
        },
    )

    ids = [item.get("index") for item in items]
    return {"ids": ids, "questions": items, "split": split, "batch_id": served_batch_id}


@app.post("/responses")
def responses(payload: ResponsesPayload):
    received_at = datetime.utcnow().isoformat() + "Z"
    batch_id = payload.batch_id if payload.batch_id is not None else _next_served_batch_id()
    answered_path = _batch_log_path(_ANSWERED_DIR, "questions_answered", batch_id)
    record = {
        "batch_id": batch_id,
        "split": payload.split or _DEFAULT_SPLIT,
        "received_at": received_at,
        "source": "responses",
        "responses": payload.responses,
    }
    if payload.question_metrics is not None:
        record["question_metrics"] = payload.question_metrics
    _write_jsonl(answered_path, record)
    return {"ok": True, "count": len(payload.responses), "batch_id": batch_id}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QuestionReceiver server.")
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
