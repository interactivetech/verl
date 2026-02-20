# Copyright 2026
#
# Licensed under the Apache License, Version 2.0.
"""
V4 solver datagen for dual-loop setup.

Solver loop:
- pulls synthetic questions from coordinator /solver/questions
- computes reward vs synthetic ground truth using existing rule reward path
- posts per-candidate difficulty back to coordinator /solver/results
"""

from __future__ import annotations

import json
import os
import re
import time
from collections import defaultdict
from typing import Any, Optional

import datasets
import requests
from omegaconf import DictConfig

try:
    from verl.experimental.dynamic_dataset.dynamicgen_dataset_v3 import AbstractDataGenerator
except Exception:
    from verl.experimental.dynamic_dataset.dynamicgen_dataset_v2 import AbstractDataGenerator


_DEFAULT_SOLVER_INSTRUCTION = 'Solve the problem step by step and output the final answer after "####".'
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


class HttpQuestionGeneratorV4Solver(AbstractDataGenerator):
    """Generate solver training rows from synthetic question queue."""

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
        self.timeout_s = int(getattr(config, "timeout_s", 10))
        self.wait_s = getattr(config, "wait_s", None)
        self.retry_sleep_s = float(getattr(config, "retry_sleep_s", 1))

        self.snapshot_size = int(getattr(config, "snapshot_size", 0) or 0)
        self.block_until_full_snapshot = bool(getattr(config, "block_until_full_snapshot", True))
        self.strict_response_post = bool(getattr(config, "strict_response_post", True))

        self.split = getattr(config, "split", "train")
        self.data_source = getattr(config, "data_source", "openai/gsm8k")
        self.instruction_following = getattr(config, "instruction_following", _DEFAULT_SOLVER_INSTRUCTION)
        self.allow_duplicates_within_epoch = bool(getattr(config, "allow_duplicates_within_epoch", False))

        self._seen_candidate_ids_in_epoch: set[str] = set()

    def _effective_snapshot_size(self) -> int:
        if self.snapshot_size > 0:
            return self.snapshot_size
        return max(1, self.n_per_batch)

    def _fetch_solver_payload(self, n: int) -> dict:
        params = {"n": n}
        if self.wait_s is not None:
            params["wait_s"] = self.wait_s

        while True:
            try:
                r = requests.get(self.questions_url, params=params, timeout=self.timeout_s)
                if r.status_code == 503:
                    _trace("solver.datagen.fetch.retry_503", url=self.questions_url, requested=n)
                    time.sleep(self.retry_sleep_s)
                    continue
                r.raise_for_status()
                payload = r.json()
                if isinstance(payload, dict):
                    items = payload.get("questions") or []
                    _trace(
                        "solver.datagen.fetch.ok",
                        url=self.questions_url,
                        requested=n,
                        received=len(items),
                        candidate_ids=[it.get("candidate_id") for it in items if isinstance(it, dict)],
                    )
                    return payload
                return {}
            except requests.exceptions.Timeout:
                _trace("solver.datagen.fetch.timeout", url=self.questions_url, requested=n, timeout_s=self.timeout_s)
                time.sleep(self.retry_sleep_s)
            except requests.exceptions.RequestException as exc:
                if getattr(exc, "response", None) is not None and exc.response.status_code in {503, 504}:
                    _trace(
                        "solver.datagen.fetch.retry_http",
                        url=self.questions_url,
                        requested=n,
                        status=exc.response.status_code,
                    )
                    time.sleep(self.retry_sleep_s)
                    continue
                raise

    def _rows_from_items(self, items: list[dict]) -> list[dict]:
        rows = []
        for item in items:
            question = item.get("question")
            answer = item.get("answer")
            solution = item.get("solution")
            candidate_id = item.get("candidate_id")
            split = item.get("split", self.split)
            if not question or not candidate_id:
                continue

            if solution is None and isinstance(answer, str):
                solution = _extract_solution(answer)
            if solution is None:
                # Drop items without numeric ground truth; coordinator should mostly gate these already.
                continue

            prompt_text = question
            if self.instruction_following and self.instruction_following not in prompt_text:
                prompt_text = f"{prompt_text} {self.instruction_following}"

            rows.append(
                {
                    "data_source": self.data_source,
                    "prompt": [{"role": "user", "content": prompt_text}],
                    "ability": "math",
                    "reward_model": {"style": "rule", "ground_truth": solution},
                    "extra_info": {
                        "split": split,
                        "candidate_id": candidate_id,
                        "question": question,
                        "answer": answer,
                        "solution": solution,
                        "seed_question_id": item.get("seed_question_id"),
                        "cycle_id": item.get("cycle_id"),
                        "meta": item.get("meta"),
                    },
                }
            )
            _trace(
                "solver.datagen.row_built",
                candidate_id=candidate_id,
                split=split,
                has_solution=solution is not None,
                question_len=len(question),
                answer_len=len(answer) if isinstance(answer, str) else None,
                question_preview=_preview(question) if _TRACE_INCLUDE_TEXT else None,
            )
        return rows

    def _get_items(self, n: int) -> list[dict]:
        payload = self._fetch_solver_payload(n)
        items = payload.get("questions") or []
        if not items:
            raise RuntimeError(f"No synthetic questions returned from coordinator: {self.questions_url}")
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
            "solver.datagen.generate.start",
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
                "Check questioner supply and coordinator queue."
            )

        for row in unique_rows:
            cid = row.get("extra_info", {}).get("candidate_id")
            if cid is not None:
                self._seen_candidate_ids_in_epoch.add(str(cid))

        _trace(
            "solver.datagen.generate.done",
            produced=len(unique_rows),
            candidate_ids=[row.get("extra_info", {}).get("candidate_id") for row in unique_rows],
        )
        return datasets.Dataset.from_list(unique_rows)

    def on_batch_end(self, dataset, batch) -> None:
        if not self.results_url or batch is None:
            return

        # Rewards are scalarized from token-level scores; >0 indicates solver got the item correct.
        rewards = None
        if "token_level_scores" in batch.batch.keys():
            rewards = _to_cpu(batch.batch["token_level_scores"]).sum(-1).tolist()
        elif "token_level_rewards" in batch.batch.keys():
            rewards = _to_cpu(batch.batch["token_level_rewards"]).sum(-1).tolist()
        else:
            return

        extra_info_arr = batch.non_tensor_batch.get("extra_info") if batch.non_tensor_batch else None
        if extra_info_arr is None:
            return

        by_candidate = defaultdict(lambda: {"n": 0, "n_correct": 0})
        for i, reward in enumerate(rewards):
            if i >= len(extra_info_arr):
                break
            extra = extra_info_arr[i] or {}
            cid = extra.get("candidate_id")
            if cid is None:
                continue
            stats = by_candidate[str(cid)]
            stats["n"] += 1
            if reward is not None and reward > 0:
                stats["n_correct"] += 1

        results = []
        for cid, stats in by_candidate.items():
            n = int(stats["n"])
            n_correct = int(stats["n_correct"])
            difficulty = (n_correct / n) if n > 0 else 0.0
            results.append(
                {
                    "candidate_id": cid,
                    "difficulty": float(difficulty),
                    "n": n,
                    "n_correct": n_correct,
                }
            )

        if not results:
            _trace("solver.datagen.batch_end.no_results")
            return

        try:
            requests.post(self.results_url, json={"results": results}, timeout=self.timeout_s).raise_for_status()
            _trace("solver.datagen.batch_end.posted", url=self.results_url, results=results)
        except Exception as exc:
            _trace("solver.datagen.batch_end.post_failed", url=self.results_url, error=str(exc), results=results)
            if self.strict_response_post:
                raise RuntimeError(f"Failed to post solver results to {self.results_url}") from exc
