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
GSM8K index server aligned with HttpQuestionGenerator.

Returns GSM8K examples:
  GET /questions?n=8&split=train  -> {"questions": [{"question": "...", "answer": "...", "index": 0, "split": "train"}]}

Environment variables:
  GSM8K_DATASET:   HF dataset name (default: openai/gsm8k)
  GSM8K_SUBSET:    HF subset name (default: main)
  GSM8K_SPLIT:     Default split (default: train)
  GSM8K_LOCAL_PATH: Local dataset path (optional)
  SEED:            Optional int seed for deterministic sampling
"""

from __future__ import annotations

import os
import random
import threading
from typing import Dict

import datasets
from fastapi import FastAPI, HTTPException, Query

app = FastAPI()

_DATA_SOURCE = os.getenv("GSM8K_DATASET", "openai/gsm8k")
_SUBSET = os.getenv("GSM8K_SUBSET", "main")
_DEFAULT_SPLIT = os.getenv("GSM8K_SPLIT", "train")
_LOCAL_PATH = os.getenv("GSM8K_LOCAL_PATH", None)
_SEED = os.getenv("SEED", None)

_rng_lock = threading.Lock()
_rng = random.Random(int(_SEED)) if _SEED is not None else random.SystemRandom()

_splits: Dict[str, datasets.Dataset] = {}
_shuffled_indices: Dict[str, list[int]] = {}
_cursor: Dict[str, int] = {}


def _load_dataset() -> Dict[str, datasets.Dataset]:
    if _LOCAL_PATH:
        dataset = datasets.load_dataset(_LOCAL_PATH, _SUBSET)
    else:
        dataset = datasets.load_dataset(_DATA_SOURCE, _SUBSET)
    return {split: dataset[split] for split in dataset.keys()}


@app.on_event("startup")
def _startup():
    global _splits
    _splits = _load_dataset()
    for split_name, ds in _splits.items():
        _init_split_state(split_name, len(ds))


@app.get("/health")
def health():
    return {"ok": True, "splits": sorted(_splits.keys())}


def _init_split_state(split: str, ds_len: int) -> None:
    indices = list(range(ds_len))
    _rng.shuffle(indices)
    _shuffled_indices[split] = indices
    _cursor[split] = 0


def _next_ids(split: str, n: int, ds_len: int) -> list[int]:
    if split not in _shuffled_indices:
        _init_split_state(split, ds_len)

    ids: list[int] = []
    for _ in range(n):
        if _cursor[split] >= ds_len:
            _init_split_state(split, ds_len)
        ids.append(_shuffled_indices[split][_cursor[split]])
        _cursor[split] += 1
    return ids


@app.get("/questions")
def questions(
    n: int = Query(8, ge=1, le=4096),
    split: str = Query(None),
):
    split = split or _DEFAULT_SPLIT
    if split not in _splits:
        raise HTTPException(status_code=400, detail=f"Unknown split '{split}'. Available: {sorted(_splits.keys())}")

    ds = _splits[split]
    ds_len = len(ds)
    if ds_len == 0:
        raise HTTPException(status_code=400, detail=f"Split '{split}' is empty.")

    with _rng_lock:
        ids = _next_ids(split, n, ds_len)

    questions = []
    for idx in ids:
        example = ds[int(idx)]
        questions.append(
            {
                "question": example.get("question"),
                "answer": example.get("answer"),
                "index": int(idx),
                "split": split,
            }
        )

    # Keep ids for compatibility with clients expecting indices.
    return {"ids": ids, "questions": questions}
