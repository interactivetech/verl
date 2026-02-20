# Copyright 2026
#
# Licensed under the Apache License, Version 2.0.
"""
V8 questioner datagen + reward for direct queue-based dual-GRPO.

Behavior:
- generate prompts from a fixed system prompt (no seed path)
- parse questioner rollout into `<question>...</question>` + `\\boxed{...}`
- submit all candidates in one request and wait for solver results
- use uncertainty reward for valid parses:
  r_uncertainty = 1 - 2 * abs(p_hat - 0.5)
- invalid parse receives configurable negative reward
- optional R-Zero-style diversity penalty over parse-valid questions
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from collections import Counter
from typing import Any, Optional

import datasets
import requests
from omegaconf import DictConfig

try:
    import numpy as np
except Exception:
    np = None

try:
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
except Exception:
    SmoothingFunction = None
    sentence_bleu = None

try:
    from sklearn.cluster import AgglomerativeClustering
except Exception:
    AgglomerativeClustering = None

try:
    from verl.experimental.dynamic_dataset.dynamicgen_dataset_v3 import AbstractDataGenerator
except Exception:
    from verl.experimental.dynamic_dataset.dynamicgen_dataset_v2 import AbstractDataGenerator


_DEFAULT_SYSTEM_PROMPT = (
    "You are an expert competition-math problem setter. FIRST, in your private scratch-pad, think "
    "step-by-step to design a brand-new, non-trivial problem. The problem could come from any field "
    "of mathematics, including but not limited to algebra, geometry, number theory, combinatorics, "
    "prealgebra, probability, statistics, and calculus. Aim for a difficulty such that fewer than 30% "
    "of advanced high-school students could solve it. Avoid re-using textbook cliches or famous contest "
    "problems. THEN, without revealing any of your private thoughts, output exactly the following two "
    "blocks:\n<question>{The full problem statement on one or more lines}</question>\\boxed{final answer}\n"
    "Do NOT output anything else: no explanations, no extra markup."
)
_DEFAULT_USER_PROMPT = (
    "Generate one new, challenging reasoning question now. Remember to format the output exactly as instructed."
)
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


def _bleu_distance_matrix(sentences: list[str]):
    if np is None or sentence_bleu is None or SmoothingFunction is None:
        return None
    n = len(sentences)
    dist = np.zeros((n, n))
    smoother = SmoothingFunction().method1
    for i in range(n):
        for j in range(i, n):
            if i == j:
                score = 1.0
            else:
                ref = [sentences[j].split()]
                hyp = sentences[i].split()
                score = sentence_bleu(ref, hyp, smoothing_function=smoother)
            dist[i, j] = dist[j, i] = 1 - score
    return dist


def _cluster_share_per_problem(
    problems: list[str],
    distance_threshold: float = 0.5,
    linkage: str = "average",
) -> list[float]:
    if not problems:
        return []
    if AgglomerativeClustering is None:
        return [0.0 for _ in problems]

    dist_mat = _bleu_distance_matrix(problems)
    if dist_mat is None:
        return [0.0 for _ in problems]

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=float(distance_threshold),
        metric="precomputed",
        linkage=str(linkage),
    )
    labels = clustering.fit_predict(dist_mat)
    total = len(problems)
    cluster_size = Counter(labels)
    cluster_ratio = {lab: sz / total for lab, sz in cluster_size.items()}
    return [float(cluster_ratio[lab]) for lab in labels]


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


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


def _to_numeric_answer_token(boxed_answer: str) -> Optional[str]:
    if not isinstance(boxed_answer, str):
        return None
    token = boxed_answer.strip().strip("$")
    token = token.replace(",", "")
    token = re.sub(r"\s+", "", token)

    if re.fullmatch(r"[+-]?[0-9]+(?:\.[0-9]+)?", token):
        if token.startswith("+"):
            token = token[1:]
        return token

    frac = re.fullmatch(r"([+-]?[0-9]+)\s*/\s*([1-9][0-9]*)", token)
    if frac:
        numer = int(frac.group(1))
        denom = int(frac.group(2))
        value = numer / denom
        out = f"{value:.10f}".rstrip("0").rstrip(".")
        if out == "-0":
            out = "0"
        return out
    return None


def parse_question_and_answer(solution_str: str) -> tuple[Optional[str], Optional[str], str]:
    if not isinstance(solution_str, str) or not solution_str.strip():
        return None, None, "empty"

    text = solution_str.strip()
    q_match = re.search(r"(?is)<question>\s*(.*?)\s*</question>", text)
    boxed_answer = _extract_last_boxed_content(text)
    if q_match and boxed_answer:
        question = _normalize_ws(q_match.group(1))
        answer = _normalize_ws(boxed_answer)
        if question and answer:
            return question, answer, "question_tag+boxed"

    return None, None, "unparsed"


def _resolve_submit_wait_url(coordinator_url: Optional[str], submit_wait_url: Optional[str]) -> str:
    base = (coordinator_url or os.getenv("DUAL_LOOP_COORDINATOR_URL") or "http://127.0.0.1:8080").rstrip("/")
    return submit_wait_url or os.getenv("QUESTIONER_SUBMIT_WAIT_URL") or f"{base}/questioner/submit_and_wait"


def _score_from_result(can_queue_solver: bool, result: dict) -> dict:
    difficulty = result.get("difficulty")
    p_hat = float(max(0.0, min(1.0, float(difficulty)))) if difficulty is not None else 0.0
    uncertainty_reward = max(0.0, min(1.0, 1.0 - (2.0 * abs(p_hat - 0.5))))
    score = uncertainty_reward if can_queue_solver else 0.0
    return {
        "score": float(score),
        "uncertainty_reward": float(uncertainty_reward),
        "difficulty": float(p_hat),
        "p_hat": float(p_hat),
        "status": str(result.get("status")),
        "resolved": float(bool(result.get("resolved", False))),
    }


def _is_gsm8k_source(data_source: Any) -> bool:
    return str(data_source).strip() == "openai/gsm8k"


def _score_gsm8k_eval(solution_str: str, ground_truth: Any) -> dict[str, float]:
    """Held-out eval path: local GSM8K exact-match, no coordinator traffic."""
    from verl.utils.reward_score import gsm8k

    score = float(gsm8k.compute_score(solution_str=solution_str, ground_truth=ground_truth, method="strict"))
    score = max(0.0, min(1.0, score))
    # Keep the same scalar fields used by training/metrics code paths.
    return {
        "score": score,
        "uncertainty_reward": score,
        "difficulty": score,
        "p_hat": score,
        "status": "local_eval",
        "resolved": 1.0,
    }


def compute_synth_difficulty_scores_batch(
    data_sources=None,
    solution_strs=None,
    ground_truths=None,
    extra_infos=None,
    data_source=None,
    solution_str=None,
    ground_truth=None,
    extra_info=None,
    min_question_chars: int = 8,
    require_numeric_ground_truth: bool = True,
    submit_timeout_s: float = 20.0,
    wait_timeout_s: float = -1.0,
    fail_on_timeout: bool = True,
    coordinator_url: Optional[str] = None,
    submit_wait_url: Optional[str] = None,
    invalid_parse_penalty: float = -1.0,
    enable_diversity_penalty: bool = False,
    diversity_penalty_weight: float = 1.0,
    diversity_distance_threshold: float = 0.5,
    diversity_linkage: str = "average",
    **_unused_kwargs,
):
    """Batch reward for questioner: submit all generated candidates and wait once."""
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
        data_sources = ["dual_loop/questioner_v8" for _ in range(len(solution_strs))]
    if ground_truths is None:
        ground_truths = [None for _ in range(len(solution_strs))]

    n_items = len(solution_strs)
    if len(extra_infos) != n_items:
        raise RuntimeError(
            f"Length mismatch in questioner batch reward: len(solution_strs)={n_items} len(extra_infos)={len(extra_infos)}"
        )

    # Eval rows (e.g. openai/gsm8k) must never hit the queue coordinator.
    local_eval_outputs: dict[int, dict[str, Any]] = {}
    active_indices: list[int] = []
    for i in range(n_items):
        if _is_gsm8k_source(data_sources[i]):
            local_eval_outputs[i] = _score_gsm8k_eval(solution_str=solution_strs[i], ground_truth=ground_truths[i])
        else:
            active_indices.append(i)

    # If all rows are local eval, return immediately.
    if not active_indices:
        out_all = []
        for i in range(n_items):
            row = local_eval_outputs[i]
            row.update(
                {
                    "parse_ok": 1.0,
                    "question_ok": 1.0,
                    "answer_ok": 1.0,
                    "numeric_final_ok": 1.0,
                    "parse_mode": "gsm8k_eval",
                    "candidate_id": None,
                    "timeout_fallback": 0.0,
                }
            )
            out_all.append(row)
        return out_all[0] if single_mode else out_all

    url = _resolve_submit_wait_url(coordinator_url=coordinator_url, submit_wait_url=submit_wait_url)
    candidate_specs: list[dict[str, Any]] = []
    payload_candidates: list[dict[str, Any]] = []

    for i in active_indices:
        solution_str = solution_strs[i]
        extra_info = extra_infos[i] or {}
        candidate_id = uuid.uuid4().hex
        question, boxed_answer, parse_mode = parse_question_and_answer(solution_str)
        numeric_solution = _to_numeric_answer_token(boxed_answer) if boxed_answer is not None else None

        question_ok = question is not None and len(question) >= int(min_question_chars)
        answer_ok = boxed_answer is not None and len(boxed_answer) > 0
        numeric_final_ok = numeric_solution is not None
        parse_ok = bool(question_ok and answer_ok and (numeric_final_ok or not require_numeric_ground_truth))
        can_queue_solver = bool(question_ok and answer_ok and numeric_final_ok)

        split = str(extra_info.get("split", "train"))
        batch_id = str(extra_info.get("batch_id", "v8"))
        candidate_question = question if question is not None else _normalize_ws(str(solution_str or ""))
        candidate_answer = f"#### {numeric_solution}" if numeric_solution is not None else "#### 0"

        candidate_specs.append(
            {
                "row_idx": i,
                "candidate_id": candidate_id,
                "parse_ok": parse_ok,
                "question_ok": question_ok,
                "answer_ok": answer_ok,
                "numeric_final_ok": numeric_final_ok,
                "parse_mode": parse_mode,
                "can_queue_solver": can_queue_solver,
                "split": split,
                "batch_id": batch_id,
                "question_text": candidate_question,
            }
        )
        payload_candidates.append(
            {
                "candidate_id": candidate_id,
                "question": candidate_question,
                "answer": candidate_answer,
                "parse_ok": bool(parse_ok),
                "question_ok": bool(question_ok),
                "answer_ok": bool(answer_ok),
                "numeric_final_ok": bool(numeric_final_ok),
                "parse_mode": parse_mode,
                "split": split,
                "meta": {
                    "source": "questioner_v8",
                    "raw_boxed_answer": boxed_answer,
                    "raw_output_preview": _preview(solution_str, 1200) if _TRACE_INCLUDE_TEXT else None,
                },
            }
        )

    split = candidate_specs[0]["split"] if candidate_specs else "train"
    batch_id = candidate_specs[0]["batch_id"] if candidate_specs else "v8"
    payload = {
        "split": split,
        "batch_id": batch_id,
        "wait_s": float(wait_timeout_s),
        "candidates": payload_candidates,
    }

    try:
        r = requests.post(url, json=payload, timeout=float(submit_timeout_s))
        if r.status_code >= 400:
            raise RuntimeError(f"Request failed: {url} status={r.status_code} detail={r.text[:800]}")
        body = r.json()
        results = body.get("results") or []
        by_cid = {
            str(item.get("candidate_id")): item
            for item in results
            if isinstance(item, dict) and item.get("candidate_id") is not None
        }
    except Exception as exc:
        _trace(
            "v8.questioner.batch_reward.error",
            error=str(exc),
            fail_on_timeout=fail_on_timeout,
            count=len(active_indices),
        )
        if fail_on_timeout:
            raise RuntimeError(f"Failed to obtain questioner batch rewards via {url}") from exc
        out = [None for _ in range(n_items)]
        # Local eval outputs already available.
        for i, row in local_eval_outputs.items():
            out[i] = {
                **row,
                "parse_ok": 1.0,
                "question_ok": 1.0,
                "answer_ok": 1.0,
                "numeric_final_ok": 1.0,
                "parse_mode": "gsm8k_eval",
                "candidate_id": None,
                "timeout_fallback": 0.0,
            }
        for spec in candidate_specs:
            base_score = float(invalid_parse_penalty) if not bool(spec["parse_ok"]) else 0.0
            out[spec["row_idx"]] = {
                "score": float(base_score),
                "base_score": float(base_score),
                "uncertainty_reward": 0.0,
                "difficulty": 0.0,
                "p_hat": 0.0,
                "diversity_penalty": 0.0,
                "diversity_penalty_applied": 0.0,
                "parse_ok": float(spec["parse_ok"]),
                "question_ok": float(spec["question_ok"]),
                "answer_ok": float(spec["answer_ok"]),
                "numeric_final_ok": float(spec["numeric_final_ok"]),
                "parse_mode": spec["parse_mode"],
                "candidate_id": spec["candidate_id"],
                "status": "timeout",
                "resolved": 0.0,
                "timeout_fallback": 1.0,
            }
        result = [x for x in out if x is not None]
        return result[0] if single_mode else result

    out = [None for _ in range(n_items)]
    for i, row in local_eval_outputs.items():
        out[i] = {
            **row,
            "parse_ok": 1.0,
            "question_ok": 1.0,
            "answer_ok": 1.0,
            "numeric_final_ok": 1.0,
            "parse_mode": "gsm8k_eval",
            "candidate_id": None,
            "timeout_fallback": 0.0,
        }

    diversity_penalty_map: dict[int, float] = {}
    if bool(enable_diversity_penalty):
        valid_specs = [spec for spec in candidate_specs if bool(spec.get("parse_ok", False))]
        valid_questions = [str(spec.get("question_text", "")) for spec in valid_specs]
        shares = _cluster_share_per_problem(
            valid_questions,
            distance_threshold=float(diversity_distance_threshold),
            linkage=str(diversity_linkage),
        )
        if shares and len(shares) == len(valid_specs):
            for spec, share in zip(valid_specs, shares):
                diversity_penalty_map[int(spec["row_idx"])] = float(share)

    for spec in candidate_specs:
        result = by_cid.get(spec["candidate_id"], {"candidate_id": spec["candidate_id"], "difficulty": 0.0, "status": "missing"})
        scored = _score_from_result(can_queue_solver=bool(spec["can_queue_solver"]), result=result)
        if not bool(spec["parse_ok"]):
            base_score = float(invalid_parse_penalty)
        else:
            base_score = float(scored.get("score", 0.0))

        diversity_penalty = float(diversity_penalty_map.get(int(spec["row_idx"]), 0.0))
        if bool(spec["parse_ok"]) and bool(enable_diversity_penalty):
            final_score = float(base_score - (float(diversity_penalty_weight) * diversity_penalty))
            diversity_applied = 1.0
        else:
            final_score = float(base_score)
            diversity_applied = 0.0

        scored["base_score"] = float(base_score)
        scored["score"] = float(final_score)
        scored["diversity_penalty"] = float(diversity_penalty)
        scored["diversity_penalty_applied"] = float(diversity_applied)
        scored.update(
            {
                "parse_ok": float(spec["parse_ok"]),
                "question_ok": float(spec["question_ok"]),
                "answer_ok": float(spec["answer_ok"]),
                "numeric_final_ok": float(spec["numeric_final_ok"]),
                "parse_mode": spec["parse_mode"],
                "candidate_id": spec["candidate_id"],
                "timeout_fallback": 0.0,
            }
        )
        out[spec["row_idx"]] = scored

    out_final = [x for x in out if x is not None]
    _trace(
        "v8.questioner.batch_reward.done",
        count=len(out_final),
        queued_count=len(active_indices),
        local_eval_count=len(local_eval_outputs),
        diversity_enabled=bool(enable_diversity_penalty),
        diversity_weight=float(diversity_penalty_weight),
        invalid_parse_penalty=float(invalid_parse_penalty),
        url=url,
    )
    return out_final[0] if single_mode else out_final


def compute_synth_difficulty_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    **kwargs,
):
    """Single-item wrapper for compatibility. Prefer batch reward in production."""
    results = compute_synth_difficulty_scores_batch(
        data_sources=[data_source],
        solution_strs=[solution_str],
        ground_truths=[ground_truth],
        extra_infos=[extra_info or {}],
        **kwargs,
    )
    return results[0]


class HttpQuestionGeneratorV8Questioner(AbstractDataGenerator):
    """Generate questioner prompts from a fixed system template (no seed path)."""

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.n_per_batch = int(getattr(config, "n_per_batch", 1))
        self.snapshot_size = int(getattr(config, "snapshot_size", 0) or 0)
        self.data_source = getattr(config, "data_source", "dual_loop/questioner_v8")
        self.split = getattr(config, "split", "train")
        self.system_prompt = getattr(config, "system_prompt", _DEFAULT_SYSTEM_PROMPT)
        self.user_prompt = getattr(config, "user_prompt", _DEFAULT_USER_PROMPT)

    def _effective_snapshot_size(self) -> int:
        if self.snapshot_size > 0:
            return self.snapshot_size
        return max(1, self.n_per_batch)

    def on_epoch_start(self, dataset, epoch: int) -> None:
        del dataset, epoch

    def generate(self, dataset) -> datasets.Dataset:
        del dataset
        target_size = self._effective_snapshot_size()
        batch_id = uuid.uuid4().hex
        rows = []
        for _ in range(target_size):
            question_id = uuid.uuid4().hex
            rows.append(
                {
                    "data_source": self.data_source,
                    "prompt": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": self.user_prompt},
                    ],
                    "ability": "math",
                    "reward_model": {"style": "rule", "ground_truth": None},
                    "extra_info": {
                        "split": self.split,
                        "question_id": question_id,
                        "batch_id": batch_id,
                    },
                }
            )

        _trace(
            "v8.questioner.generate.done",
            split=self.split,
            produced=len(rows),
            batch_id=batch_id,
        )
        return datasets.Dataset.from_list(rows)


# Backward compatibility aliases.
HttpQuestionGeneratorV6Questioner = HttpQuestionGeneratorV8Questioner
HttpQuestionGeneratorV7Questioner = HttpQuestionGeneratorV8Questioner
