# Copyright 2026
#
# Licensed under the Apache License, Version 2.0.
"""
V4 synth datagen for dual-loop setup.

Questioner loop:
- pulls real seeds from coordinator /questioner/seeds
- generates synthetic QA candidates
- custom reward submits candidates to coordinator and awaits solver-derived difficulty
"""

from __future__ import annotations

import json
import os
import random
import re
import time
import uuid
from typing import Any, Iterable, Optional

import datasets
import numpy as np
import requests
from omegaconf import DictConfig

try:
    from verl.experimental.dynamic_dataset.dynamicgen_dataset_v3 import AbstractDataGenerator
except Exception:
    from verl.experimental.dynamic_dataset.dynamicgen_dataset_v2 import AbstractDataGenerator


_DEFAULT_SYSTEM_PROMPT = "You are a helpful math question generator."
_DEFAULT_USER_TEMPLATE = (
    "You are given one real seed question and its answer.\n\n"
    "Seed Question:\n{seed_question}\n\n"
    "Seed Answer:\n{seed_answer}\n\n"
    "Task:\n"
    "Generate ONE new, different math word problem inspired by the seed. Make sure its diverse and complementary, to help a student study how to solve a question like this.\n\n"
    "Output rules:\n"
    '1) Return valid JSON only (no markdown, no code fences).\n'
    '2) Use exactly two keys: "question" and "answer".\n'
    '3) "question" must be a complete standalone math word problem.\n'
    '4) "answer" should include brief reasoning and end with `#### <final_number>`.\n'
)
_TRACE_ENABLED = os.getenv("DUAL_LOOP_TRACE", "false").strip().lower() in {"1", "true", "yes", "on"}
_TRACE_INCLUDE_TEXT = os.getenv("DUAL_LOOP_TRACE_INCLUDE_TEXT", "false").strip().lower() in {"1", "true", "yes", "on"}
_TRACE_MAX_TEXT_CHARS = int(os.getenv("DUAL_LOOP_TRACE_MAX_TEXT_CHARS", "180"))


def _extract_solution(solution_str: str) -> Optional[str]:
    if not solution_str:
        return None
    match = re.search(r"####\s*(\-?[0-9][0-9,\.]*)", solution_str)
    if not match:
        return None
    return match.group(1).replace(",", "")


def _int_or_none(value) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


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
    payload = {"event": event, "ts": time.time()}
    payload.update(fields)
    print(f"[dual-loop-trace] {json.dumps(payload, ensure_ascii=False, separators=(',', ':'), default=str)}", flush=True)


def parse_synth_qa(solution_str: str) -> tuple[Optional[str], Optional[str], str]:
    if not isinstance(solution_str, str) or not solution_str.strip():
        return None, None, "empty"

    text = solution_str.strip()
    candidates: list[str] = []

    for match in re.finditer(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL):
        candidates.append(match.group(1).strip())
    if text.startswith("{") and text.endswith("}"):
        candidates.append(text)
    left = text.find("{")
    right = text.rfind("}")
    if 0 <= left < right:
        candidates.append(text[left : right + 1])

    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        question = obj.get("question")
        answer = obj.get("answer")
        if isinstance(question, str) and isinstance(answer, str):
            q = _normalize_ws(question)
            a = _normalize_ws(answer)
            if q and a:
                return q, a, "json"

    q_match = re.search(r"(?is)<question>\s*(.*?)\s*</question>", text)
    a_match = re.search(r"(?is)<answer>\s*(.*?)\s*</answer>", text)
    if q_match and a_match:
        q = _normalize_ws(q_match.group(1))
        a = _normalize_ws(a_match.group(1))
        if q and a:
            return q, a, "xml_tags"

    qa_match = re.search(r"(?is)\bquestion\s*:\s*(.+?)\s+\banswer\s*:\s*(.+)$", text)
    if qa_match:
        q = _normalize_ws(qa_match.group(1))
        a = _normalize_ws(qa_match.group(2))
        if q and a:
            return q, a, "qa_labels"

    return None, None, "unparsed"


def _coordinator_base_url(reward_kwargs: dict) -> str:
    return (
        reward_kwargs.get("coordinator_url")
        or os.getenv("DUAL_LOOP_COORDINATOR_URL")
        or "http://127.0.0.1:8080"
    ).rstrip("/")


def _post_or_raise(url: str, payload: dict, timeout_s: float) -> dict:
    _trace("questioner.reward.http_post.start", url=url, timeout_s=timeout_s)
    r = requests.post(url, json=payload, timeout=timeout_s)
    if r.status_code >= 400:
        _trace("questioner.reward.http_post.error", url=url, status=r.status_code, detail=r.text[:200])
        raise RuntimeError(f"Coordinator request failed: {url} status={r.status_code} detail={r.text[:800]}")
    _trace("questioner.reward.http_post.ok", url=url, status=r.status_code)
    return r.json()


def _await_reward_by_polling(
    base_url: str,
    candidate_id: str,
    wait_timeout_s: float,
    poll_interval_s: float,
    request_timeout_s: float,
) -> dict:
    """Poll coordinator with wait_s=0 to avoid long-lived blocking HTTP handlers."""
    _trace(
        "questioner.reward.await.start",
        candidate_id=candidate_id,
        wait_timeout_s=wait_timeout_s,
        poll_interval_s=poll_interval_s,
    )
    deadline = time.monotonic() + max(wait_timeout_s, 0.0)
    last_error: Optional[Exception] = None

    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break

        timeout_s = max(2.0, min(request_timeout_s, remaining + 1.0))
        payload = {"candidate_ids": [candidate_id], "wait_s": 0.0}
        try:
            r = requests.post(f"{base_url}/questioner/rewards/await", json=payload, timeout=timeout_s)
            if r.status_code == 200:
                data = r.json()
                results = data.get("results") or []
                if results:
                    result = results[0]
                    if bool(result.get("resolved", False)) and result.get("difficulty") is not None:
                        _trace(
                            "questioner.reward.await.resolved",
                            candidate_id=candidate_id,
                            difficulty=result.get("difficulty"),
                            status=result.get("status"),
                        )
                        return result
            elif r.status_code != 504:
                raise RuntimeError(
                    f"Coordinator request failed: {base_url}/questioner/rewards/await "
                    f"status={r.status_code} detail={r.text[:800]}"
                )
        except requests.RequestException as exc:
            last_error = exc

        time.sleep(max(0.05, min(poll_interval_s, max(remaining, 0.05))))

    if last_error is not None:
        _trace("questioner.reward.await.timeout", candidate_id=candidate_id, error=str(last_error))
        raise RuntimeError(
            f"Timed out waiting for solver-derived difficulty for candidate_id={candidate_id}"
        ) from last_error
    _trace("questioner.reward.await.timeout", candidate_id=candidate_id, error=None)
    raise RuntimeError(f"Timed out waiting for solver-derived difficulty for candidate_id={candidate_id}")


def _to_jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_to_jsonable(v) for v in value]
    return value


def compute_synth_difficulty_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    min_question_chars: int = 8,
    min_answer_chars: int = 3,
    require_numeric_final_answer: bool = False,
    difficulty_target: float = 0.5,
    w_parse: float = 0.2,
    w_diff: float = 0.8,
    submit_timeout_s: float = 10.0,
    wait_timeout_s: float = 180.0,
    poll_interval_s: float = 0.25,
    fail_on_timeout: bool = True,
    coordinator_url: Optional[str] = None,
):
    """Parse-gated questioner reward using solver-derived difficulty from coordinator."""
    del data_source, ground_truth

    candidate_id = uuid.uuid4().hex
    question, answer, parse_mode = parse_synth_qa(solution_str)
    question_ok = question is not None and len(question) >= int(min_question_chars)
    answer_ok = answer is not None and len(answer) >= int(min_answer_chars)
    numeric_final_ok = True
    if require_numeric_final_answer:
        numeric_final_ok = bool(answer) and bool(re.search(r"####\s*\-?[0-9][0-9,\.]*", answer))

    parse_ok = question_ok and answer_ok and numeric_final_ok
    _trace(
        "questioner.reward.parse",
        candidate_id=candidate_id,
        parse_ok=parse_ok,
        question_ok=question_ok,
        answer_ok=answer_ok,
        numeric_final_ok=numeric_final_ok,
        parse_mode=parse_mode,
        question_len=len(question) if question is not None else 0,
        answer_len=len(answer) if answer is not None else 0,
        question_preview=_preview(question) if _TRACE_INCLUDE_TEXT else None,
    )

    parse_score = 1.0 if parse_ok else 0.0
    if not parse_ok:
        return {
            "score": 0.0,
            "parse_ok": 0.0,
            "question_ok": float(question_ok),
            "answer_ok": float(answer_ok),
            "numeric_final_ok": float(numeric_final_ok),
            "parse_mode": parse_mode,
            "difficulty": 0.0,
            "difficulty_target_score": 0.0,
            "w_parse": float(w_parse),
            "w_diff": float(w_diff),
            "candidate_id": candidate_id,
        }

    extra_info = extra_info or {}
    base_url = (coordinator_url or _coordinator_base_url(locals())).rstrip("/")
    split = extra_info.get("split", "train")

    candidate_payload = {
        "split": split,
        "cycle_id": str(extra_info.get("batch_id", "unknown")),
        "candidates": [
            {
                "candidate_id": candidate_id,
                "question": question,
                "answer": answer,
                "parse_ok": True,
                "split": split,
                "seed_question_id": extra_info.get("question_id"),
                "seed_question": extra_info.get("question"),
                "seed_answer": extra_info.get("answer"),
                "seed_solution": extra_info.get("solution"),
                "seed_extra_info": _to_jsonable(extra_info),
            }
        ],
    }

    try:
        _trace(
            "questioner.reward.submit",
            candidate_id=candidate_id,
            split=split,
            cycle_id=str(extra_info.get("batch_id", "unknown")),
            seed_question_id=extra_info.get("question_id"),
        )
        _post_or_raise(f"{base_url}/questioner/candidates", candidate_payload, timeout_s=float(submit_timeout_s))
        result = _await_reward_by_polling(
            base_url=base_url,
            candidate_id=candidate_id,
            wait_timeout_s=float(wait_timeout_s),
            poll_interval_s=float(poll_interval_s),
            request_timeout_s=float(submit_timeout_s),
        )
    except Exception as exc:
        _trace("questioner.reward.error", candidate_id=candidate_id, error=str(exc), fail_on_timeout=fail_on_timeout)
        if fail_on_timeout:
            raise RuntimeError(f"Failed to obtain solver-derived difficulty for candidate_id={candidate_id}") from exc
        return {
            "score": float(w_parse),
            "parse_ok": parse_score,
            "question_ok": 1.0,
            "answer_ok": 1.0,
            "numeric_final_ok": float(numeric_final_ok),
            "parse_mode": parse_mode,
            "difficulty": 0.0,
            "difficulty_target_score": 0.0,
            "timeout_fallback": 1.0,
            "w_parse": float(w_parse),
            "w_diff": float(w_diff),
            "candidate_id": candidate_id,
        }
    difficulty = result.get("difficulty")
    if difficulty is None:
        raise RuntimeError(f"Coordinator returned no difficulty for candidate_id={candidate_id}: {result}")
    difficulty = float(difficulty)

    target = float(difficulty_target)
    target_score = max(0.0, 1.0 - 2.0 * abs(difficulty - target))

    w_parse = float(w_parse)
    w_diff = float(w_diff)
    denom = max(1e-8, w_parse + w_diff)
    w_parse = w_parse / denom
    w_diff = w_diff / denom

    score = parse_score * (w_parse + w_diff * target_score)
    _trace(
        "questioner.reward.final",
        candidate_id=candidate_id,
        difficulty=difficulty,
        target=difficulty_target,
        target_score=target_score,
        score=score,
    )

    return {
        "score": float(score),
        "parse_ok": parse_score,
        "question_ok": 1.0,
        "answer_ok": 1.0,
        "numeric_final_ok": float(numeric_final_ok),
        "parse_mode": parse_mode,
        "difficulty": float(difficulty),
        "difficulty_target_score": float(target_score),
        "w_parse": float(w_parse),
        "w_diff": float(w_diff),
        "candidate_id": candidate_id,
    }


class HttpQuestionGeneratorV4Synth(AbstractDataGenerator):
    """Generate questioner prompts from coordinator-provided real seeds."""

    def __init__(self, config: DictConfig):
        super().__init__(config)
        base_url = (getattr(config, "coordinator_url", None) or os.getenv("DUAL_LOOP_COORDINATOR_URL") or "").rstrip("/")
        self.seed_url = (
            getattr(config, "seed_url", None)
            or os.getenv("QUESTIONER_SEED_URL")
            or (f"{base_url}/questioner/seeds" if base_url else "http://127.0.0.1:8080/questioner/seeds")
        )
        self.n_per_batch = int(getattr(config, "n_per_batch", 1))
        self.timeout_s = int(getattr(config, "timeout_s", 10))
        self.wait_s = getattr(config, "wait_s", None)
        self.retry_sleep_s = float(getattr(config, "retry_sleep_s", 1))

        self.snapshot_size = int(getattr(config, "snapshot_size", 0) or 0)
        self.block_until_full_snapshot = bool(getattr(config, "block_until_full_snapshot", True))
        self.allow_duplicates_within_epoch = bool(getattr(config, "allow_duplicates_within_epoch", False))

        self.split = getattr(config, "split", "train")
        self.seed = int(getattr(config, "seed", 0))
        self.system_prompt = getattr(config, "system_prompt", _DEFAULT_SYSTEM_PROMPT)
        self.user_prompt_template = getattr(config, "user_prompt_template", _DEFAULT_USER_TEMPLATE)
        self.data_source = getattr(config, "data_source", "dual_loop/questioner")

        self.rng = random.Random(self.seed)
        self._dataset_size_hint: Optional[int] = None
        self._seen_question_ids_in_epoch: set[str] = set()

    def _extract_dataset_size_hint(self, payload: dict) -> None:
        if not isinstance(payload, dict):
            return
        for key in ("dataset_size", "total_questions", "total_size"):
            value = _int_or_none(payload.get(key))
            if value is not None and value > 0:
                self._dataset_size_hint = value
                return

    def _effective_snapshot_size(self) -> int:
        if self.snapshot_size > 0:
            return self.snapshot_size
        if self._dataset_size_hint is not None and self._dataset_size_hint > 0:
            return self._dataset_size_hint
        return max(1, self.n_per_batch)

    def _fetch_seed_payload(self, n: int) -> dict:
        params = {"n": n, "split": self.split}
        if self.wait_s is not None:
            params["wait_s"] = self.wait_s

        while True:
            try:
                r = requests.get(self.seed_url, params=params, timeout=self.timeout_s)
                if r.status_code == 503:
                    _trace("questioner.datagen.fetch.retry_503", url=self.seed_url, requested=n)
                    time.sleep(self.retry_sleep_s)
                    continue
                r.raise_for_status()
                payload = r.json()
                if isinstance(payload, dict):
                    self._extract_dataset_size_hint(payload)
                    items = payload.get("questions") or []
                    _trace(
                        "questioner.datagen.fetch.ok",
                        url=self.seed_url,
                        requested=n,
                        received=len(items),
                        batch_id=payload.get("batch_id"),
                    )
                    return payload
                return {}
            except requests.exceptions.Timeout:
                _trace("questioner.datagen.fetch.timeout", url=self.seed_url, requested=n, timeout_s=self.timeout_s)
                time.sleep(self.retry_sleep_s)
            except requests.exceptions.RequestException as exc:
                if getattr(exc, "response", None) is not None and exc.response.status_code in {503, 504}:
                    _trace(
                        "questioner.datagen.fetch.retry_http",
                        url=self.seed_url,
                        requested=n,
                        status=exc.response.status_code,
                    )
                    time.sleep(self.retry_sleep_s)
                    continue
                raise

    def _build_user_prompt(self, seed_question: str, seed_answer: Optional[str], seed_solution: Optional[str]) -> str:
        answer_text = seed_answer or ""
        if seed_solution and "####" not in answer_text:
            answer_text = (answer_text + f"\nFinal numeric answer: {seed_solution}").strip()
        return self.user_prompt_template.format(seed_question=seed_question, seed_answer=answer_text)

    def _rows_from_items(self, items: list[dict]) -> list[dict]:
        rows = []
        for item in items:
            question_raw = item.get("question")
            answer_raw = item.get("answer")
            index = item.get("index", 0)
            split = item.get("split", self.split)
            if not question_raw:
                continue
            solution = item.get("solution")
            if solution is None and isinstance(answer_raw, str):
                solution = _extract_solution(answer_raw)
            qid = item.get("question_id") or f"{split}:{int(index) if index is not None else 0}"
            prompt = [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": self._build_user_prompt(
                        seed_question=str(question_raw),
                        seed_answer=answer_raw if isinstance(answer_raw, str) else None,
                        seed_solution=solution,
                    ),
                },
            ]
            rows.append(
                {
                    "data_source": self.data_source,
                    "prompt": prompt,
                    "ability": "math",
                    "reward_model": {"style": "rule", "ground_truth": None},
                    "extra_info": {
                        "split": split,
                        "index": int(index) if index is not None else 0,
                        "answer": answer_raw,
                        "solution": solution,
                        "question": question_raw,
                        "batch_id": item.get("batch_id"),
                        "question_id": qid,
                        "order_id": item.get("order_id"),
                        "source_epoch": item.get("source_epoch"),
                    },
                }
            )
            _trace(
                "questioner.datagen.row_built",
                question_id=qid,
                split=split,
                seed_batch_id=item.get("batch_id"),
                question_len=len(str(question_raw)),
                answer_len=len(answer_raw) if isinstance(answer_raw, str) else None,
                seed_question_preview=_preview(str(question_raw)) if _TRACE_INCLUDE_TEXT else None,
            )
        return rows

    def _get_items(self, n: int) -> list[dict]:
        payload = self._fetch_seed_payload(n)
        items = payload.get("questions") or []
        if not items:
            raise RuntimeError(f"No seed questions returned from coordinator: {self.seed_url}")
        return items

    def on_epoch_start(self, dataset, epoch: int) -> None:
        del dataset, epoch
        self._seen_question_ids_in_epoch.clear()

    def generate(self, dataset) -> datasets.Dataset:
        del dataset
        target_size = self._effective_snapshot_size()
        unique_rows: list[dict] = []
        unique_qids: set[str] = set()
        _trace(
            "questioner.datagen.generate.start",
            target_size=target_size,
            n_per_batch=self.n_per_batch,
            snapshot_size=self.snapshot_size,
        )

        while len(unique_rows) < target_size:
            need = max(1, min(self.n_per_batch, target_size - len(unique_rows)))
            items = self._get_items(need)
            rows = self._rows_from_items(items)

            progress = False
            for row in rows:
                extra = row.get("extra_info", {})
                qid = str(extra.get("question_id", ""))
                if not qid:
                    continue
                if qid in unique_qids:
                    continue
                if not self.allow_duplicates_within_epoch and qid in self._seen_question_ids_in_epoch:
                    continue
                unique_qids.add(qid)
                unique_rows.append(row)
                progress = True
                if len(unique_rows) >= target_size:
                    break

            if len(unique_rows) >= target_size:
                break
            if not self.block_until_full_snapshot:
                break
            if not progress:
                time.sleep(self.retry_sleep_s)

        if self.block_until_full_snapshot and len(unique_rows) < target_size:
            raise RuntimeError(
                f"Unable to build full unique snapshot: got {len(unique_rows)} / {target_size}. "
                "Check coordinator seed availability."
            )

        for row in unique_rows:
            qid = row.get("extra_info", {}).get("question_id")
            if qid is not None:
                self._seen_question_ids_in_epoch.add(str(qid))

        _trace(
            "questioner.datagen.generate.done",
            produced=len(unique_rows),
            question_ids=[row.get("extra_info", {}).get("question_id") for row in unique_rows],
        )
        return datasets.Dataset.from_list(unique_rows)
