# Copyright 2026
#
# Licensed under the Apache License, Version 2.0.
"""V6 questioner dataset entrypoint for dynamic generation."""

from verl.experimental.dynamic_dataset.dynamicgen_dataset_v3 import AbstractDataGenerator
from verl.experimental.dynamic_dataset.dynamicgen_dataset_v3 import DynamicGenDataset as _DynamicGenDatasetV3


class DynamicGenDataset(_DynamicGenDatasetV3):
    """V6 questioner alias of V3 DynamicGenDataset."""


__all__ = ["AbstractDataGenerator", "DynamicGenDataset"]

