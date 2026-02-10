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
Dataset class that enables dynamic data generation strategies between iterations of training.
This class extends RLHFDataset and uses an AbstractDataGen instance to generate data.

This is especially useful in settings where proposer model generates new tasks based
on rollout data.
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
        """
        Generate method must be implemented by subclasses.
        Args:
            dataset: The dataset to generate from.
        Returns:
            Processed data or result as implemented by the subclass.
        """
        pass


class MockDataGenerator(AbstractDataGenerator):
    """
    A noop data gen class that only reappends the first datapoint.
    This class is useful as a placeholder and testing.
    """

    def __init__(self, config: DictConfig = None):
        super().__init__(config)

    def generate(self, dataset: Dataset) -> datasets.Dataset:
        print("MockDataGenerator: No operation performed on the dataset.")
        return dataset.dataframe.select([0])


class DynamicGenDataset(RLHFDataset):
    """
    A dataset class that uses a data generation strategy to process data.
    This class extends RLHFDataset and uses an AbstractDataGen instance to generate data.
    """

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
        self.skip_base_dataset = bool(config.datagen.get("skip_base_dataset", False)) and is_train
        super().__init__(data_files, tokenizer, config, processor, max_samples=max_samples)
        self.datagen: AbstractDataGenerator = config.datagen
        assert "datagen" in config and config.datagen.get("path", None) is not None, (
            f"datagen path is not set in config: {config}"
        )
        # Dynamically load the custom datagen class
        datagen_cls = load_extern_object(config.datagen.path, config.datagen.name)

        # Verify that the custom datagen class inherits from AbstractDataGenerator
        abs_cls = AbstractDataGenerator
        if not issubclass(datagen_cls, abs_cls):
            raise TypeError(
                f"The custom datagen class '{config.datagen.name}' from '{config.datagen.path}'"
                + " must inherit from {abs_cls}"
            )

        self.data_generator = datagen_cls(config.datagen)
        self.on_batch_end()

    def _download(self, use_origin_parquet: bool = False):
        if self.skip_base_dataset:
            return
        return super()._download(use_origin_parquet=use_origin_parquet)

    def _read_files_and_tokenize(self):
        if self.skip_base_dataset:
            # Start from an empty dataset and rely on datagen to populate it.
            self.dataframe = datasets.Dataset.from_list([])
            return
        return super()._read_files_and_tokenize()

    def append_dataframe(self, new_dataframe: datasets.Dataset):
        new_dataframe = self.maybe_filter_out_long_prompts(new_dataframe)
        if self.dataframe is None or len(getattr(self.dataframe, "column_names", [])) == 0:
            self.dataframe = new_dataframe
        else:
            self.dataframe = datasets.concatenate_datasets([self.dataframe, new_dataframe])

        logger.info(f"new dataset len: {len(self.dataframe)}")

    def on_batch_end(self, batch: DataProto | None = None) -> None:
        """
        Generate data using the provided data generation strategy.
        Note: This method is intended to change the dataset after each training batch.
        """
        if hasattr(self.data_generator, "on_batch_end"):
            try:
                self.data_generator.on_batch_end(self, batch)
            except Exception as exc:
                logger.warning("datagen on_batch_end failed: %s", exc)
        new_data = self.data_generator.generate(self)
        self.append_dataframe(new_data)
