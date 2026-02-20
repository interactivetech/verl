# Copyright 2026
#
# Licensed under the Apache License, Version 2.0.
"""
V6 solver datagen for direct queue-based dual-GRPO.

Behavior:
- pull questioner-generated questions from `/solver/questions`
- build solver prompts with system+user messages
- solver reward uses batch self-consistency (no questioner-provided answer)
- after rollout, compute per-question majority agreement and post `/solver/results`
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections import Counter
from typing import Any, Optional

import datasets
import requests
from omegaconf import DictConfig

try:
    from verl.experimental.dynamic_dataset.dynamicgen_dataset_v3 import AbstractDataGenerator
except Exception:
    from verl.experimental.dynamic_dataset.dynamicgen_dataset_v2 import AbstractDataGenerator


_DEFAULT_SOLVER_SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."
_DEFAULT_SOLVER_USER_TEMPLATE = "{problem_statement}"
_TRACE_ENABLED = os.getenv("DUAL_LOOP_TRACE", "false").strip().lower() in {"1", "true", "yes", "on"}
_TRACE_INCLUDE_TEXT = os.getenv("DUAL_LOOP_TRACE_INCLUDE_TEXT", "false").strip().lower() in {"1", "true", "yes", "on"}
_TRACE_MAX_TEXT_CHARS = int(os.getenv("DUAL_LOOP_TRACE_MAX_TEXT_CHARS", "180"))


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


def _to_cpu(tensor):
    if tensor is None:
        return None
    if hasattr(tensor, "detach"):
        tensor = tensor.detach()
    if hasattr(tensor, "cpu"):
        tensor = tensor.cpu()
    return tensor


def _extract_last_boxed_content(text: str) -> Optional[str]:
    if not isinstance(text, str) or not text:
        return None

    idx = text.rfind("\\boxed{")
    if idx < 0:
        return None

    i = idx + len("\\boxed{")
    depth = 1
    out = []
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
            out.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = "".join(out).strip()
                return candidate if candidate else None
            out.append(ch)
        else:
            out.append(ch)
        i += 1
    return None


def _normalize_answer_token(value: str) -> str:
    token = str(value).strip()
    token = token.strip("$")
    token = re.sub(r"\s+", "", token)
    numeric = re.fullmatch(r"[+-]?[0-9][0-9,]*(?:\.[0-9]+)?", token)
    if numeric:
        token = token.replace(",", "")
        if token.startswith("+"):
            token = token[1:]
    return token


def _extract_prediction_token(solution_str: str) -> tuple[Optional[str], str]:
    if not isinstance(solution_str, str) or not solution_str.strip():
        return None, "empty"

    boxed = _extract_last_boxed_content(solution_str)
    if boxed:
        return _normalize_answer_token(boxed), "boxed"

    m = re.findall(r"####\s*([^\n\r]+)", solution_str)
    if m:
        return _normalize_answer_token(m[-1]), "####"

    m2 = re.findall(r"([+-]?[0-9][0-9,]*(?:\.[0-9]+)?)", solution_str)
    if m2:
        return _normalize_answer_token(m2[-1]), "numeric_fallback"

    return None, "unparsed"


def _stable_group_key(extra: dict[str, Any], data_source: Any, idx: int) -> tuple[str, str]:
    """Build a deterministic grouping key for self-consistency reward.

    Preferred key order:
    1) candidate_id (expected path)
    2) content hash from split/question/answer
    3) per-row fallback (idx)
    """
    candidate_id = extra.get("candidate_id")
    if candidate_id not in (None, ""):
        return f"cid:{candidate_id}", "candidate_id"

    split = str(extra.get("split", ""))
    question = str(extra.get("question", "")).strip()
    answer = str(extra.get("answer", "")).strip()
    if question:
        digest = hashlib.sha1(f"{split}\n{question}\n{answer}".encode("utf-8")).hexdigest()[:16]
        return f"qa:{digest}", "question_answer_hash"

    if data_source not in (None, ""):
        return f"data_source:{data_source}:idx:{idx}", "data_source_idx_fallback"
    return f"idx:{idx}", "row_index_fallback"


def _is_gsm8k_source(data_source: Any) -> bool:
    return str(data_source).strip() == "openai/gsm8k"


def _gsm8k_score_entry(solution_str: str, ground_truth: Any) -> dict[str, Any]:
    """Compute strict GSM8K exact-match score for held-out validation."""
    from verl.utils.reward_score import gsm8k

    pred = gsm8k.extract_solution(solution_str=solution_str, method="strict")
    score = float(gsm8k.compute_score(solution_str=solution_str, ground_truth=ground_truth, method="strict"))
    parsed = pred is not None
    return {
        "score": float(score),
        "pred": pred,
        "pred_mode": "gsm8k_strict",
        "majority_answer": str(ground_truth) if ground_truth is not None else None,
        "match_majority": float(score),
        "parse_ok": float(parsed),
        "group_size": 1.0,
        "group_majority_count": float(score),
        "p_hat": float(score),
        "candidate_id": None,
        "group_key": "gsm8k_eval",
        "group_source": "gsm8k_exact_match",
    }


def compute_solver_self_consistency_scores_batch(
    data_sources=None,
    solution_strs=None,
    ground_truths=None,
    extra_infos=None,
    data_source=None,
    solution_str=None,
    ground_truth=None,
    extra_info=None,
    positive_score: float = 1.0,
    negative_score: float = 0.0,
    invalid_score: float = 0.0,
    **_unused_kwargs,
):
    """Solver reward supporting both batch and single-item invocation styles."""
    single_mode = solution_str is not None or data_source is not None or extra_info is not None
    if solution_strs is None and single_mode:
        data_sources = [data_source]
        solution_strs = [solution_str]
        ground_truths = [ground_truth]
        extra_infos = [extra_info or {}]

    if solution_strs is None:
        raise TypeError("Missing `solution_strs` for batch mode or `solution_str` for single mode.")

    if extra_infos is None:
        extra_infos = [{} for _ in range(len(solution_strs))]
    if data_sources is None:
        data_sources = ["dual_loop/solver_v6_consistency" for _ in range(len(solution_strs))]
    if ground_truths is None:
        ground_truths = [None for _ in range(len(solution_strs))]

    n_items = len(solution_strs)
    if len(extra_infos) != n_items:
        raise RuntimeError(
            f"Length mismatch in solver batch reward: len(solution_strs)={n_items} len(extra_infos)={len(extra_infos)}"
        )

    outputs: list[Optional[dict[str, Any]]] = [None for _ in range(n_items)]
    active_indices: list[int] = []
    for i in range(n_items):
        data_source_i = data_sources[i] if i < len(data_sources) else None
        if _is_gsm8k_source(data_source_i):
            outputs[i] = _gsm8k_score_entry(solution_str=solution_strs[i], ground_truth=ground_truths[i])
        else:
            active_indices.append(i)

    if not active_indices:
        out = [o for o in outputs if o is not None]
        return out[0] if single_mode else out

    grouped: dict[str, dict[str, Any]] = {}
    index_info: list[tuple[int, str, Optional[str], str, str, str]] = []
    for i in active_indices:
        sol = solution_strs[i]
        extra = extra_infos[i] or {}
        data_source_i = data_sources[i] if i < len(data_sources) else None
        group_key, group_source = _stable_group_key(extra, data_source_i, i)
        cid = extra.get("candidate_id")
        candidate_id = str(cid) if cid not in (None, "") else ""
        parse_ok = bool(extra.get("parse_ok", True))
        pred, mode = _extract_prediction_token(sol)
        rec = grouped.setdefault(
            group_key,
            {
                "indices": [],
                "parse_ok": parse_ok,
                "counts": Counter(),
                "first_seen": {},
                "group_source": group_source,
                "candidate_id": candidate_id,
            },
        )
        rec["indices"].append(i)
        rec["parse_ok"] = bool(rec.get("parse_ok", True)) and parse_ok
        if pred is not None:
            if pred not in rec["first_seen"]:
                rec["first_seen"][pred] = len(rec["first_seen"])
            rec["counts"][pred] += 1
        index_info.append((i, group_key, pred, mode, candidate_id, group_source))

    majority_by_group: dict[str, tuple[Optional[str], int, bool]] = {}
    for group_key, rec in grouped.items():
        parse_ok = bool(rec.get("parse_ok", True))
        counts: Counter = rec["counts"]
        first_seen = rec["first_seen"]
        if (not parse_ok) or not counts:
            majority_by_group[group_key] = (None, 0, parse_ok)
            continue
        majority_answer, majority_count = max(
            counts.items(),
            key=lambda kv: (int(kv[1]), -int(first_seen.get(kv[0], 10**9))),
        )
        majority_by_group[group_key] = (str(majority_answer), int(majority_count), parse_ok)

    for i, group_key, pred, mode, candidate_id, group_source in index_info:
        majority_answer, majority_count, parse_ok = majority_by_group[group_key]
        n_total = len(grouped[group_key]["indices"])
        if (not parse_ok) or majority_answer is None or pred is None:
            score = float(invalid_score)
            match_majority = False
        else:
            match_majority = bool(pred == majority_answer)
            score = float(positive_score if match_majority else negative_score)

        p_hat = float(majority_count / max(1, n_total)) if n_total > 0 else 0.0
        outputs[i] = {
            "score": float(score),
            "pred": pred,
            "pred_mode": mode,
            "majority_answer": majority_answer,
            "match_majority": float(match_majority),
            "parse_ok": float(parse_ok),
            "group_size": float(n_total),
            "group_majority_count": float(majority_count),
            "p_hat": float(p_hat),
            "candidate_id": candidate_id or None,
            "group_key": group_key,
            "group_source": group_source,
        }

    out = [o for o in outputs if o is not None]
    return out[0] if single_mode else out


class HttpQuestionGeneratorV6Solver(AbstractDataGenerator):
    """Solver-side dynamic datagen for direct queue flow."""

    def __init__(self, config: DictConfig):
        super().__init__(config)
        base_url = (getattr(config, "coordinator_url", None) or os.getenv("DUAL_LOOP_COORDINATOR_URL") or "").rstrip("/")
        self.questions_url = (
            getattr(config, "questions_url", None)
            or os.getenv("SOLVER_QUESTIONS_URL")
            or (f"{base_url}/solver/questions" if base_url else "http://127.0.0.1:8080/solver/questions")
        )
        self.results_url = (
            getattr(config, "results_url", None)
            or os.getenv("SOLVER_RESULTS_URL")
            or (f"{base_url}/solver/results" if base_url else "http://127.0.0.1:8080/solver/results")
        )

        self.n_per_batch = int(getattr(config, "n_per_batch", 4))
        self.timeout_s = int(getattr(config, "timeout_s", 20))
        self.wait_s = getattr(config, "wait_s", -1)
        self.retry_sleep_s = float(getattr(config, "retry_sleep_s", 1))

        self.snapshot_size = int(getattr(config, "snapshot_size", 0) or 0)
        self.block_until_full_snapshot = bool(getattr(config, "block_until_full_snapshot", True))
        self.strict_response_post = bool(getattr(config, "strict_response_post", True))

        self.split = getattr(config, "split", "train")
        self.data_source = getattr(config, "data_source", "dual_loop/solver_v6_consistency")
        self.system_prompt = getattr(config, "system_prompt", _DEFAULT_SOLVER_SYSTEM_PROMPT)
        self.user_template = getattr(config, "user_template", _DEFAULT_SOLVER_USER_TEMPLATE)
        self.allow_duplicates_within_epoch = bool(getattr(config, "allow_duplicates_within_epoch", False))
        # Keep malformed/questioner-noisy samples bounded so prompt-length filtering
        # does not collapse a full solver batch into an empty dataloader.
        self.max_question_chars = int(getattr(config, "max_question_chars", 1500))

        self._seen_candidate_ids_in_epoch: set[str] = set()

    def _effective_snapshot_size(self) -> int:
        if self.snapshot_size > 0:
            return self.snapshot_size
        return max(1, self.n_per_batch)

    def _fetch_solver_payload(self, n: int) -> dict:
        params = {"n": n}
        if self.wait_s is not None:
            params["wait_s"] = self.wait_s

        # Intentionally retry forever: this gives an "indefinite wait" behavior
        # even if the queue service uses bounded wait windows internally.
        while True:
            try:
                r = requests.get(self.questions_url, params=params, timeout=self.timeout_s)
                if r.status_code in {503, 504}:
                    detail = None
                    try:
                        detail = r.text[:800]
                    except Exception:
                        detail = None
                    _trace(
                        "v6.solver.fetch.retry_http",
                        url=self.questions_url,
                        requested=n,
                        status=r.status_code,
                        detail=detail,
                    )
                    time.sleep(self.retry_sleep_s)
                    continue
                r.raise_for_status()
                payload = r.json()
                if isinstance(payload, dict):
                    items = payload.get("questions") or []
                    parse_ok_count = 0
                    empty_question_count = 0
                    empty_answer_count = 0
                    for it in items:
                        if not isinstance(it, dict):
                            continue
                        if bool(it.get("parse_ok", False)):
                            parse_ok_count += 1
                        if not str(it.get("question") or "").strip():
                            empty_question_count += 1
                        if not str(it.get("answer") or "").strip():
                            empty_answer_count += 1
                    _trace(
                        "v6.solver.fetch.ok",
                        url=self.questions_url,
                        requested=n,
                        received=len(items),
                        candidate_ids=[it.get("candidate_id") for it in items if isinstance(it, dict)],
                        parse_ok_count=parse_ok_count,
                        empty_question_count=empty_question_count,
                        empty_answer_count=empty_answer_count,
                    )
                    return payload
                return {}
            except requests.exceptions.Timeout:
                _trace("v6.solver.fetch.timeout", url=self.questions_url, requested=n, timeout_s=self.timeout_s)
                time.sleep(self.retry_sleep_s)
            except requests.exceptions.RequestException as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status in {503, 504}:
                    detail = None
                    try:
                        detail = getattr(exc.response, "text", None)
                        if detail is not None:
                            detail = str(detail)[:800]
                    except Exception:
                        detail = None
                    _trace(
                        "v6.solver.fetch.retry_exc",
                        url=self.questions_url,
                        requested=n,
                        status=status,
                        detail=detail,
                    )
                    time.sleep(self.retry_sleep_s)
                    continue
                raise

    def _build_user_prompt(self, problem_statement: str) -> str:
        try:
            return self.user_template.format(problem_statement=problem_statement)
        except Exception:
            return problem_statement

    def _rows_from_items(self, items: list[dict]) -> list[dict]:
        rows = []
        for item in items:
            candidate_id = item.get("candidate_id")
            question = item.get("question")
            if not question or not candidate_id:
                _trace(
                    "v6.solver.row_dropped",
                    reason="missing_candidate_or_question",
                    candidate_id=candidate_id,
                    has_question=bool(question),
                    has_answer=bool(item.get("answer")),
                    parse_ok=bool(item.get("parse_ok", False)),
                    item_preview=_preview(json.dumps(item, ensure_ascii=False)) if _TRACE_INCLUDE_TEXT else None,
                )
                continue

            answer = item.get("answer")
            solution = None
            question_str = str(question)
            truncated = False
            if self.max_question_chars > 0 and len(question_str) > self.max_question_chars:
                question_str = question_str[: self.max_question_chars]
                truncated = True
                _trace(
                    "v6.solver.question_truncated",
                    candidate_id=candidate_id,
                    original_len=len(str(question)),
                    truncated_len=len(question_str),
                    max_question_chars=self.max_question_chars,
                )

            rows.append(
                {
                    "data_source": self.data_source,
                    "prompt": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": self._build_user_prompt(question_str)},
                    ],
                    "ability": "math",
                    "reward_model": {"style": "rule", "ground_truth": None},
                    "extra_info": {
                        "split": item.get("split", self.split),
                        "candidate_id": candidate_id,
                        "question": question_str,
                        "answer": answer,
                        "solution": solution,
                        "parse_ok": bool(item.get("parse_ok", True)),
                        "question_truncated": bool(truncated),
                        "meta": item.get("meta"),
                    },
                }
            )
            _trace(
                "v6.solver.row_built",
                candidate_id=candidate_id,
                split=item.get("split", self.split),
                parse_ok=bool(item.get("parse_ok", True)),
                has_solution=False,
                question_len=len(question_str),
                answer_len=len(str(answer or "")),
                question_preview=_preview(question_str) if _TRACE_INCLUDE_TEXT else None,
            )
        return rows

    def _get_items(self, n: int) -> list[dict]:
        payload = self._fetch_solver_payload(n)
        items = payload.get("questions") or []
        if not items:
            raise RuntimeError(f"No questions returned from queue endpoint: {self.questions_url}")
        return items

    def on_epoch_start(self, dataset, epoch: int) -> None:
        del dataset, epoch
        self._seen_candidate_ids_in_epoch.clear()

    def generate(self, dataset) -> datasets.Dataset:
        del dataset
        target_size = self._effective_snapshot_size()
        unique_rows: list[dict] = []
        unique_candidate_ids: set[str] = set()
        _trace(
            "v6.solver.generate.start",
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
                cid = str(row.get("extra_info", {}).get("candidate_id", ""))
                if not cid:
                    continue
                if cid in unique_candidate_ids:
                    continue
                if not self.allow_duplicates_within_epoch and cid in self._seen_candidate_ids_in_epoch:
                    continue

                unique_candidate_ids.add(cid)
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
                f"Unable to build full solver snapshot: got {len(unique_rows)} / {target_size}. "
                "Check questioner supply and queue availability."
            )

        for row in unique_rows:
            cid = row.get("extra_info", {}).get("candidate_id")
            if cid is not None:
                self._seen_candidate_ids_in_epoch.add(str(cid))

        _trace(
            "v6.solver.generate.done",
            produced=len(unique_rows),
            candidate_ids=[row.get("extra_info", {}).get("candidate_id") for row in unique_rows],
        )
        return datasets.Dataset.from_list(unique_rows)

    def on_batch_end(self, dataset, batch) -> None:
        if not self.results_url or batch is None:
            return

        try:
            responses = dataset.tokenizer.batch_decode(_to_cpu(batch.batch["responses"]), skip_special_tokens=True)
        except Exception:
            _trace("v6.solver.batch_end.decode_failed")
            return

        extra_info_arr = batch.non_tensor_batch.get("extra_info") if batch.non_tensor_batch else None
        if extra_info_arr is None:
            _trace("v6.solver.batch_end.no_extra_info")
            return

        by_candidate: dict[str, dict[str, Any]] = {}
        for i, response in enumerate(responses):
            if i >= len(extra_info_arr):
                break
            extra = extra_info_arr[i] or {}
            cid = extra.get("candidate_id")
            if cid is None:
                continue
            cid = str(cid)

            rec = by_candidate.setdefault(
                cid,
                {
                    "n_total": 0,
                    "parsed_counts": Counter(),
                    "first_seen": {},
                    "parse_ok": bool(extra.get("parse_ok", True)),
                },
            )
            rec["n_total"] += 1

            pred, mode = _extract_prediction_token(response)
            if pred is None:
                _trace("v6.solver.pred.unparsed", candidate_id=cid, mode=mode)
                continue

            if pred not in rec["first_seen"]:
                rec["first_seen"][pred] = len(rec["first_seen"])
            rec["parsed_counts"][pred] += 1

        results = []
        for cid, stats in by_candidate.items():
            n_total = int(stats["n_total"])
            parse_ok = bool(stats.get("parse_ok", True))
            parsed_counts: Counter = stats["parsed_counts"]
            first_seen = stats["first_seen"]

            if (not parse_ok) or n_total <= 0 or not parsed_counts:
                majority_answer = None
                majority_count = 0
                agreement_ratio = 0.0
            else:
                majority_answer, majority_count = max(
                    parsed_counts.items(),
                    key=lambda kv: (int(kv[1]), -int(first_seen.get(kv[0], 10**9))),
                )
                majority_count = int(majority_count)
                agreement_ratio = float(majority_count / max(1, n_total))

            results.append(
                {
                    "candidate_id": cid,
                    "difficulty": float(agreement_ratio),
                    "n": int(n_total),
                    "n_correct": int(majority_count),
                    "meta": {
                        "majority_answer": majority_answer,
                        "majority_count": int(majority_count),
                        "parsed_rollouts": int(sum(parsed_counts.values())),
                        "unparsed_rollouts": int(max(0, n_total - sum(parsed_counts.values()))),
                        "agreement_ratio": float(agreement_ratio),
                        "parse_ok": bool(parse_ok),
                        "source": "solver_v6_majority_vote",
                    },
                }
            )
            _trace(
                "v6.solver.majority",
                candidate_id=cid,
                n_total=n_total,
                majority_count=majority_count,
                agreement_ratio=agreement_ratio,
                majority_answer=_preview(majority_answer),
            )

        if not results:
            _trace("v6.solver.batch_end.no_results")
            return

        try:
            requests.post(self.results_url, json={"results": results}, timeout=self.timeout_s).raise_for_status()
            _trace("v6.solver.batch_end.posted", url=self.results_url, count=len(results))
        except Exception as exc:
            _trace("v6.solver.batch_end.post_failed", url=self.results_url, error=str(exc), count=len(results))
            if self.strict_response_post:
                raise RuntimeError(f"Failed to post solver results to {self.results_url}") from exc
