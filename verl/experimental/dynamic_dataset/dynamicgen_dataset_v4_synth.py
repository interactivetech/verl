# Copyright 2026
#
# Licensed under the Apache License, Version 2.0.
"""
V4 synth dataset entrypoint for dynamic generation.

Reuses V3 bounded snapshot dataset behavior; exposed as a new module path.
"""

from verl.experimental.dynamic_dataset.dynamicgen_dataset_v3 import AbstractDataGenerator
from verl.experimental.dynamic_dataset.dynamicgen_dataset_v3 import DynamicGenDataset as _DynamicGenDatasetV3


class DynamicGenDataset(_DynamicGenDatasetV3):
    """V4 synth alias of V3 DynamicGenDataset."""


__all__ = ["AbstractDataGenerator", "DynamicGenDataset"]
