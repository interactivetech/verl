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
Dynamic GSM8K data generator (V3).

V3 additions:
- bounded snapshot generation (no append growth)
- epoch-scoped dedup by question_id
- blocks while filling a full unique snapshot
- synchronous (blocking) response posting by default
"""

from __future__ import annotations

import os
import random
import re
import time
from typing import Iterable, Optional

import datasets
import requests
from omegaconf import DictConfig

try:
    from verl.experimental.dynamic_dataset.dynamicgen_dataset_v3 import AbstractDataGenerator
except Exception:
    from verl.experimental.dynamic_dataset.dynamicgen_dataset_v2 import AbstractDataGenerator


_DEFAULT_INSTRUCTION = 'Let\'s think step by step and output the final answer after "####".'


def _extract_solution(solution_str: str) -> Optional[str]:
    if not solution_str:
        return None
    solution = re.search(r"#### (\-?[0-9\.,]+)", solution_str)
    if solution is None:
        return None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution


def _to_cpu(tensor):
    if tensor is None:
        return None
    if hasattr(tensor, "detach"):
        tensor = tensor.detach()
    if hasattr(tensor, "cpu"):
        tensor = tensor.cpu()
    return tensor


def _derive_response_url(questions_url: Optional[str]) -> Optional[str]:
    if not questions_url:
        return None
    url = questions_url.rstrip("/")
    if url.endswith("/questions"):
        return url[: -len("/questions")] + "/responses"
    return None


def _int_or_none(value) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


class HttpQuestionGeneratorV3(AbstractDataGenerator):
    """Generate GSM8K-style RL rows from remote question API with bounded snapshot semantics."""

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.url = getattr(config, "url", os.environ.get("QUESTION_SERVER_URL"))
        self.n_per_batch = int(getattr(config, "n_per_batch", 8))
        self.timeout_s = int(getattr(config, "timeout_s", 10))
        self.wait_s = getattr(config, "wait_s", None)
        self.retry_sleep_s = float(getattr(config, "retry_sleep_s", 1))

        # Snapshot behavior.
        self.snapshot_size = int(getattr(config, "snapshot_size", 0) or 0)
        self.block_until_full_snapshot = bool(getattr(config, "block_until_full_snapshot", True))
        self.allow_duplicates_within_epoch = bool(getattr(config, "allow_duplicates_within_epoch", False))

        # Response posting behavior.
        self.strict_response_post = bool(getattr(config, "strict_response_post", True))
        self.response_url = getattr(config, "response_url", None) or os.environ.get("QUESTION_RESPONSE_URL")
        if not self.response_url:
            self.response_url = _derive_response_url(self.url)

        self.data_source = getattr(config, "data_source", "openai/gsm8k")
        self.subset = getattr(config, "subset", "main")
        self.split = getattr(config, "split", "train")
        self.seed = int(getattr(config, "seed", 0))
        self.instruction_following = getattr(config, "instruction_following", _DEFAULT_INSTRUCTION)
        self.local_dataset_path = getattr(config, "local_dataset_path", None)

        self.dataset = None
        self.rng = random.Random(self.seed)
        self._last_batch_id = None
        self._dataset_size_hint: Optional[int] = None
        self._epoch = 0
        self._seen_question_ids_in_epoch: set[str] = set()

    def _ensure_dataset(self):
        if self.dataset is not None:
            return
        if self.local_dataset_path:
            dataset = datasets.load_dataset(self.local_dataset_path, self.subset)
        else:
            dataset = datasets.load_dataset(self.data_source, self.subset)
        self.dataset = dataset[self.split]
        if self._dataset_size_hint is None:
            self._dataset_size_hint = len(self.dataset)

    def _select_indices(self) -> Iterable[int]:
        self._ensure_dataset()
        return [self.rng.randrange(0, len(self.dataset)) for _ in range(self.n_per_batch)]

    def _extract_dataset_size_hint(self, payload: dict) -> None:
        if not isinstance(payload, dict):
            return
        for key in ("dataset_size", "total_questions", "total_size"):
            value = _int_or_none(payload.get(key))
            if value is not None and value > 0:
                self._dataset_size_hint = value
                return
        meta = payload.get("meta")
        if isinstance(meta, dict):
            value = _int_or_none(meta.get("dataset_size"))
            if value is not None and value > 0:
                self._dataset_size_hint = value

    def _effective_snapshot_size(self) -> int:
        if self.snapshot_size > 0:
            return self.snapshot_size
        if self._dataset_size_hint is not None and self._dataset_size_hint > 0:
            return self._dataset_size_hint
        return self.n_per_batch

    def _fetch_server_payload(self, n: int) -> dict:
        if not self.url:
            return {}

        params = {"n": n, "split": self.split}
        if self.wait_s is not None:
            params["wait_s"] = self.wait_s

        while True:
            try:
                r = requests.get(self.url, params=params, timeout=self.timeout_s)
                if r.status_code == 503:
                    time.sleep(self.retry_sleep_s)
                    continue
                r.raise_for_status()
                payload = r.json()
                if isinstance(payload, dict):
                    self._extract_dataset_size_hint(payload)
                    if payload.get("batch_id") is not None:
                        self._last_batch_id = payload.get("batch_id")
                    return payload
                return {}
            except requests.exceptions.Timeout:
                time.sleep(self.retry_sleep_s)
            except requests.exceptions.RequestException as exc:
                if getattr(exc, "response", None) is not None and exc.response.status_code == 503:
                    time.sleep(self.retry_sleep_s)
                    continue
                raise

    def _question_id(self, item: dict, split: str, index: int) -> str:
        qid = item.get("question_id")
        if qid is not None:
            return str(qid)
        return f"{split}:{index}"

    def _rows_from_items(self, items: list[dict]) -> list[dict]:
        rows = []
        for item in items:
            question_raw = item.get("question")
            answer_raw = item.get("answer")
            index = item.get("index", 0)
            split = item.get("split", self.split)

            if question_raw is None:
                continue

            if self.instruction_following and self.instruction_following not in question_raw:
                question = f"{question_raw} {self.instruction_following}"
            else:
                question = question_raw

            solution = item.get("solution")
            if solution is None and answer_raw is not None:
                solution = _extract_solution(answer_raw)

            qid = self._question_id(item, split=split, index=int(index) if index is not None else 0)
            rows.append(
                {
                    "data_source": self.data_source,
                    "prompt": [{"role": "user", "content": question}],
                    "ability": "math",
                    "reward_model": {"style": "rule", "ground_truth": solution},
                    "extra_info": {
                        "split": split,
                        "index": int(index) if index is not None else 0,
                        "answer": answer_raw,
                        "question": question_raw,
                        "batch_id": item.get("batch_id", self._last_batch_id),
                        "question_id": qid,
                        "order_id": item.get("order_id"),
                        "source_epoch": item.get("source_epoch"),
                    },
                }
            )
        return rows

    def _get_items(self, n: int) -> list[dict]:
        payload = self._fetch_server_payload(n) if self.url else {}

        if payload and "questions" in payload:
            items = payload["questions"]
        else:
            items = []
            ids = []
            if payload:
                ids = payload.get("ids") or payload.get("indices") or []
            if not ids:
                ids = self._select_indices()
            for idx in ids:
                self._ensure_dataset()
                example = self.dataset[int(idx)]
                items.append(
                    {
                        "question": example.get("question"),
                        "answer": example.get("answer"),
                        "index": int(idx),
                        "split": self.split,
                        "question_id": f"{self.split}:{int(idx)}",
                    }
                )
        return items

    def on_epoch_start(self, dataset, epoch: int) -> None:
        self._epoch = int(epoch)
        self._seen_question_ids_in_epoch.clear()

    def generate(self, dataset) -> datasets.Dataset:
        target_size = self._effective_snapshot_size()
        unique_rows: list[dict] = []
        unique_qids: set[str] = set()

        while len(unique_rows) < target_size:
            need = max(1, min(self.n_per_batch, target_size - len(unique_rows)))
            items = self._get_items(need)
            rows = self._rows_from_items(items)

            progress = False
            for row in rows:
                extra = row.get("extra_info", {})
                qid = str(extra.get("question_id", ""))
                if not qid:
                    split = extra.get("split", self.split)
                    index = int(extra.get("index", 0))
                    qid = f"{split}:{index}"
                    extra["question_id"] = qid

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
                "Check submitter/receiver availability and uniqueness guarantees."
            )

        for row in unique_rows:
            qid = row.get("extra_info", {}).get("question_id")
            if qid is not None:
                self._seen_question_ids_in_epoch.add(str(qid))

        return datasets.Dataset.from_list(unique_rows)

    def on_batch_end(self, dataset, batch) -> None:
        if not self.response_url or batch is None:
            return

        try:
            prompts = dataset.tokenizer.batch_decode(_to_cpu(batch.batch["prompts"]), skip_special_tokens=True)
            responses = dataset.tokenizer.batch_decode(_to_cpu(batch.batch["responses"]), skip_special_tokens=True)
        except Exception:
            return

        rewards = None
        if "token_level_scores" in batch.batch.keys():
            rewards = _to_cpu(batch.batch["token_level_scores"]).sum(-1).tolist()
        elif "token_level_rewards" in batch.batch.keys():
            rewards = _to_cpu(batch.batch["token_level_rewards"]).sum(-1).tolist()

        extra_info_arr = batch.non_tensor_batch.get("extra_info") if batch.non_tensor_batch else None
        reward_model_arr = batch.non_tensor_batch.get("reward_model") if batch.non_tensor_batch else None
        request_id_arr = batch.non_tensor_batch.get("request_id") if batch.non_tensor_batch else None

        items = []
        for i, (prompt, response) in enumerate(zip(prompts, responses, strict=False)):
            extra_info = None
            if extra_info_arr is not None and i < len(extra_info_arr):
                extra_info = extra_info_arr[i]
            reward_model = None
            if reward_model_arr is not None and i < len(reward_model_arr):
                reward_model = reward_model_arr[i]

            item = {
                "prompt": prompt,
                "response": response,
                "reward": rewards[i] if rewards is not None and i < len(rewards) else None,
                "extra_info": extra_info,
                "reward_model": reward_model,
            }
            if request_id_arr is not None and i < len(request_id_arr):
                item["request_id"] = request_id_arr[i]
            items.append(item)

        question_metrics = {}
        for item in items:
            extra_info = item.get("extra_info") or {}
            qid = extra_info.get("question_id")
            split = extra_info.get("split", self.split)
            idx = extra_info.get("index")
            key = str(qid) if qid is not None else f"{split}:{idx}"

            metric = question_metrics.setdefault(
                key,
                {
                    "question_id": qid,
                    "index": int(idx) if idx is not None else None,
                    "split": split,
                    "n": 0,
                    "n_correct": 0,
                    "batch_id": extra_info.get("batch_id", self._last_batch_id),
                },
            )
            metric["n"] += 1
            reward = item.get("reward")
            if reward is not None and reward > 0:
                metric["n_correct"] += 1

        question_metrics_list = []
        for metric in question_metrics.values():
            n = metric.get("n", 0)
            metric["difficulty"] = (metric.get("n_correct", 0) / n) if n else 0.0
            question_metrics_list.append(metric)

        try:
            requests.post(
                self.response_url,
                json={
                    "split": self.split,
                    "batch_id": self._last_batch_id,
                    "responses": items,
                    "question_metrics": question_metrics_list,
                },
                timeout=self.timeout_s,
            ).raise_for_status()
        except Exception as exc:
            if self.strict_response_post:
                raise RuntimeError(f"Failed to post responses to {self.response_url}") from exc
