# Copyright 2025 Amazon.com Inc and/or its affiliates
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
Dataset class that enables dynamic data generation strategies.

V3 behavior:
- bounded map-style dataset (replace snapshot, no append growth)
- supports skip_base_dataset for remote-only training data
- calls datagen.on_batch_end for synchronous response posting
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional

import datasets
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl import DataProto
from verl.utils.dataset import RLHFDataset
from verl.utils.import_utils import load_extern_object

logger = logging.getLogger(__name__)


class AbstractDataGenerator(ABC):
    def __init__(self, config: DictConfig):
        self.config = config

    @abstractmethod
    def generate(self, dataset: Dataset) -> datasets.Dataset:
        """Generate a dataset snapshot."""
        pass


class DynamicGenDataset(RLHFDataset):
    """Map-style dynamic dataset with bounded snapshots."""

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        max_samples: int = -1,
    ):
        def _normalize_files(files: str | list[str] | ListConfig | None) -> list[str]:
            if files is None:
                return []
            if isinstance(files, list | ListConfig):
                return list(files)
            return [files]

        train_files = _normalize_files(config.get("train_files"))
        current_files = _normalize_files(data_files)
        is_train = current_files == train_files and len(current_files) > 0

        self._is_train = is_train
        self.skip_base_dataset = bool(config.datagen.get("skip_base_dataset", False)) and is_train
        self.refresh_each_epoch = bool(config.datagen.get("refresh_each_epoch", False))
        self.refresh_on_batch_end = bool(config.datagen.get("refresh_on_batch_end", False))
        self._current_epoch = 0

        super().__init__(data_files, tokenizer, config, processor, max_samples=max_samples)

        assert "datagen" in config and config.datagen.get("path", None) is not None, (
            f"datagen path is not set in config: {config}"
        )

        datagen_cls = load_extern_object(config.datagen.path, config.datagen.name)
        if not issubclass(datagen_cls, AbstractDataGenerator):
            raise TypeError(
                f"The custom datagen class '{config.datagen.name}' from '{config.datagen.path}'"
                + f" must inherit from {AbstractDataGenerator}"
            )

        self.data_generator: AbstractDataGenerator = datagen_cls(config.datagen)

        # Initialize training snapshot once (bounded replace, no append).
        if self._is_train and self.skip_base_dataset:
            self.refresh_snapshot()

    def _download(self, use_origin_parquet: bool = False):
        if self.skip_base_dataset:
            return
        return super()._download(use_origin_parquet=use_origin_parquet)

    def _read_files_and_tokenize(self):
        if self.skip_base_dataset:
            self.dataframe = datasets.Dataset.from_list([])
            return
        return super()._read_files_and_tokenize()

    def _replace_dataframe(self, new_dataframe: datasets.Dataset):
        filtered = self.maybe_filter_out_long_prompts(new_dataframe)
        if len(filtered) == 0:
            raise ValueError("DynamicGenDataset received an empty filtered snapshot.")
        self.dataframe = filtered
        logger.info("dynamic snapshot dataset len: %s", len(self.dataframe))

    def refresh_snapshot(self) -> None:
        new_data = self.data_generator.generate(self)
        self._replace_dataframe(new_data)

    def on_epoch_start(self, epoch: int) -> None:
        self._current_epoch = int(epoch)
        if hasattr(self.data_generator, "on_epoch_start"):
            self.data_generator.on_epoch_start(self, self._current_epoch)
        if self._is_train and self.refresh_each_epoch:
            self.refresh_snapshot()

    def on_batch_end(self, batch: DataProto | None = None) -> None:
        # Keep batch-end side effects (e.g. response posting), but avoid unbounded append growth.
        if hasattr(self.data_generator, "on_batch_end"):
            self.data_generator.on_batch_end(self, batch)

        if self._is_train and self.refresh_on_batch_end:
            self.refresh_snapshot()
