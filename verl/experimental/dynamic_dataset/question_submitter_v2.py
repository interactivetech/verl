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
Submit GSM8K questions to a remote QuestionReceiver (V2).

V2 additions:
- canonical question_id and order_id fields
- source_epoch indicator to track wrap-around
- dataset_size included in /questions response for auto snapshot sizing
"""

from __future__ import annotations

import argparse
import os
import random
import re
import threading
import time
from typing import Dict, List, Optional

import datasets
import requests
from fastapi import FastAPI, Query


class QuestionSubmitter:
    def __init__(
        self,
        receiver_url: str,
        n_per_batch: int,
        delay_s: float,
        split: str,
        data_source: str,
        subset: str,
        local_dataset_path: str | None,
        seed: int,
        timeout_s: int,
        max_batches: int | None,
    ) -> None:
        self.receiver_url = receiver_url
        self.n_per_batch = n_per_batch
        self.delay_s = delay_s
        self.split = split
        self.data_source = data_source
        self.subset = subset
        self.local_dataset_path = local_dataset_path
        self.seed = seed
        self.timeout_s = timeout_s
        self.max_batches = max_batches

        self._dataset = None
        self._indices: List[int] = []
        self._cursor = 0
        self._rng = random.Random(self.seed)
        self._lock = threading.Lock()

        self._source_epoch = -1
        self._global_order = 0

    @property
    def dataset_size(self) -> int:
        self._ensure_dataset()
        return len(self._dataset)

    def _ensure_dataset(self) -> None:
        if self._dataset is not None:
            return
        if self.local_dataset_path:
            dataset = datasets.load_dataset(self.local_dataset_path, self.subset)
        else:
            dataset = datasets.load_dataset(self.data_source, self.subset)
        self._dataset = dataset[self.split]
        self._reset_indices()

    def _reset_indices(self) -> None:
        assert self._dataset is not None
        self._indices = list(range(len(self._dataset)))
        self._rng.shuffle(self._indices)
        self._cursor = 0
        self._source_epoch += 1

    def _build_question_item(self, idx: int, example: dict) -> Dict[str, object]:
        answer_raw = example.get("answer")
        solution = _extract_solution(answer_raw) if isinstance(answer_raw, str) else None
        question_id = f"{self.split}:{int(idx)}"
        order_id = self._global_order
        self._global_order += 1

        return {
            "question": example.get("question"),
            "answer": answer_raw,
            "solution": solution,
            "index": int(idx),
            "split": self.split,
            "question_id": question_id,
            "order_id": order_id,
            "source_epoch": self._source_epoch,
        }

    def get_batch(self, n: Optional[int] = None, split: Optional[str] = None) -> List[Dict[str, object]]:
        if split and split != self.split:
            self.split = split
            self._dataset = None
            self._indices = []
            self._cursor = 0
            self._source_epoch = -1
            self._global_order = 0

        self._ensure_dataset()
        assert self._dataset is not None

        batch_n = self.n_per_batch if n is None else int(n)
        items: List[Dict[str, object]] = []

        with self._lock:
            for _ in range(batch_n):
                if self._cursor >= len(self._indices):
                    self._reset_indices()

                idx = self._indices[self._cursor]
                self._cursor += 1
                example = self._dataset[int(idx)]
                items.append(self._build_question_item(idx=int(idx), example=example))

        return items

    def _post_batch(self, batch_id: int, questions: List[Dict[str, object]]) -> None:
        payload = {
            "batch_id": batch_id,
            "split": self.split,
            "dataset_size": self.dataset_size,
            "questions": questions,
        }
        r = requests.post(self.receiver_url, json=payload, timeout=self.timeout_s)
        r.raise_for_status()

    def run(self) -> None:
        batch_id = 0
        while True:
            if self.max_batches is not None and batch_id >= self.max_batches:
                break

            questions = self.get_batch()
            self._post_batch(batch_id, questions)
            batch_id += 1

            if self.delay_s > 0:
                time.sleep(self.delay_s)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit GSM8K batches to QuestionReceiver (V2).")
    parser.add_argument(
        "--receiver-url",
        default=os.environ.get("QUESTION_RECEIVER_URL", "http://127.0.0.1:8080/submit"),
        help="QuestionReceiver submit endpoint.",
    )
    parser.add_argument("--n-per-batch", type=int, default=16)
    parser.add_argument("--delay-s", type=float, default=5.0)
    parser.add_argument("--split", default=os.environ.get("GSM8K_SPLIT", "train"))
    parser.add_argument("--data-source", default=os.environ.get("GSM8K_DATASET", "openai/gsm8k"))
    parser.add_argument("--subset", default=os.environ.get("GSM8K_SUBSET", "main"))
    parser.add_argument("--local-dataset-path", default=os.environ.get("GSM8K_LOCAL_PATH"))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "0")))
    parser.add_argument("--timeout-s", type=int, default=10)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--serve", action="store_true", help="Run as a QuestionSubmitter HTTP server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    return parser.parse_args()


def _extract_solution(solution_str: str) -> Optional[str]:
    if not solution_str:
        return None
    match = re.search(r"#### (\-?[0-9\.,]+)", solution_str)
    if not match:
        return None
    return match.group(1).replace(",", "")


def _build_submitter_from_env() -> QuestionSubmitter:
    return QuestionSubmitter(
        receiver_url=os.environ.get("QUESTION_RECEIVER_URL", "http://127.0.0.1:8080/submit"),
        n_per_batch=int(os.environ.get("QUESTION_N_PER_BATCH", "16")),
        delay_s=float(os.environ.get("QUESTION_DELAY_S", "5")),
        split=os.environ.get("GSM8K_SPLIT", "train"),
        data_source=os.environ.get("GSM8K_DATASET", "openai/gsm8k"),
        subset=os.environ.get("GSM8K_SUBSET", "main"),
        local_dataset_path=os.environ.get("GSM8K_LOCAL_PATH"),
        seed=int(os.environ.get("SEED", "0")),
        timeout_s=int(os.environ.get("QUESTION_TIMEOUT_S", "10")),
        max_batches=None,
    )


app = FastAPI()
_submitter: Optional[QuestionSubmitter] = None


@app.on_event("startup")
def _startup() -> None:
    global _submitter
    _submitter = _build_submitter_from_env()


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/questions")
def questions(
    n: int = Query(8, ge=1, le=4096),
    split: Optional[str] = Query(None),
):
    if _submitter is None:
        raise RuntimeError("Submitter not initialized")
    items = _submitter.get_batch(n=n, split=split)
    return {
        "questions": items,
        "split": split or _submitter.split,
        "dataset_size": _submitter.dataset_size,
    }


def main() -> None:
    args = _parse_args()
    if args.serve:
        import uvicorn

        uvicorn.run("verl.experimental.dynamic_dataset.question_submitter_v2:app", host=args.host, port=args.port)
        return

    submitter = QuestionSubmitter(
        receiver_url=args.receiver_url,
        n_per_batch=args.n_per_batch,
        delay_s=args.delay_s,
        split=args.split,
        data_source=args.data_source,
        subset=args.subset,
        local_dataset_path=args.local_dataset_path,
        seed=args.seed,
        timeout_s=args.timeout_s,
        max_batches=args.max_batches,
    )
    submitter.run()


if __name__ == "__main__":
    main()
