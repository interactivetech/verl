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
Dynamic GSM8K data generator aligned with VERL's GSM8K preprocessing.

It appends the instruction_following string to each question and returns rows
with the same fields as the offline preprocessing pipeline.
"""

from __future__ import annotations

import os
import random
import re
from typing import Iterable

import datasets
import requests
from omegaconf import DictConfig

try:
    from verl.experimental.dynamic_dataset.dynamicgen_dataset import AbstractDataGenerator
except Exception:
    from verl.utils.dataset.dynamicgen_dataset import AbstractDataGenerator


_DEFAULT_INSTRUCTION = 'Let\'s think step by step and output the final answer after "####".'


def _extract_solution(solution_str: str) -> str:
    solution = re.search(r"#### (\-?[0-9\.\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution


class HttpQuestionGenerator(AbstractDataGenerator):
    """
    Generate GSM8K questions aligned with VERL preprocessing.

    The HTTP server (if configured) is expected to return indices:
      {"ids": [1, 5, 9]} or {"indices": [1, 5, 9]}
    If not provided, it falls back to random sampling from the split.
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.url = getattr(config, "url", os.environ.get("QUESTION_SERVER_URL"))
        self.n_per_batch = int(getattr(config, "n_per_batch", 8))
        self.timeout_s = int(getattr(config, "timeout_s", 10))

        self.data_source = getattr(config, "data_source", "openai/gsm8k")
        self.subset = getattr(config, "subset", "main")
        self.split = getattr(config, "split", "train")
        self.seed = int(getattr(config, "seed", 0))
        self.instruction_following = getattr(config, "instruction_following", _DEFAULT_INSTRUCTION)
        self.local_dataset_path = getattr(config, "local_dataset_path", None)

        self.dataset = None
        self.rng = random.Random(self.seed)

    def _ensure_dataset(self):
        if self.dataset is not None:
            return
        if self.local_dataset_path:
            dataset = datasets.load_dataset(self.local_dataset_path, self.subset)
        else:
            dataset = datasets.load_dataset(self.data_source, self.subset)
        self.dataset = dataset[self.split]

    def _fetch_server_payload(self) -> dict:
        r = requests.get(self.url, params={"n": self.n_per_batch, "split": self.split}, timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()

    def _select_indices(self) -> Iterable[int]:
        self._ensure_dataset()
        return [self.rng.randrange(0, len(self.dataset)) for _ in range(self.n_per_batch)]

    def generate(self, dataset) -> datasets.Dataset:
        rows = []
        payload = None
        if self.url:
            payload = self._fetch_server_payload()

        if payload and "questions" in payload:
            items = payload["questions"]
        else:
            items = []
            for idx in (payload.get("ids") if payload else None) or (payload.get("indices") if payload else None) or self._select_indices():
                self._ensure_dataset()
                example = self.dataset[int(idx)]
                items.append(
                    {
                        "question": example.get("question"),
                        "answer": example.get("answer"),
                        "index": int(idx),
                        "split": self.split,
                    }
                )

        for item in items:
            question_raw = item.get("question")
            answer_raw = item.get("answer")
            idx = item.get("index", 0)
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

            rows.append(
                {
                    "data_source": self.data_source,
                    "prompt": [{"role": "user", "content": question}],
                    "ability": "math",
                    "reward_model": {"style": "rule", "ground_truth": solution},
                    "extra_info": {
                        "split": split,
                        "index": int(idx) if idx is not None else 0,
                        "answer": answer_raw,
                        "question": question_raw,
                    },
                }
            )
        return datasets.Dataset.from_list(rows)
