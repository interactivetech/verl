# Copyright 2026
#
# Licensed under the Apache License, Version 2.0.
"""
Synth dataset entrypoint for dynamic generation.

This file intentionally reuses the V3 bounded snapshot dataset behavior and
only provides a dedicated module path for synth experiments.
"""

from verl.experimental.dynamic_dataset.dynamicgen_dataset_v3 import AbstractDataGenerator
from verl.experimental.dynamic_dataset.dynamicgen_dataset_v3 import DynamicGenDataset as _DynamicGenDatasetV3


class DynamicGenDataset(_DynamicGenDatasetV3):
    """Synth alias of V3 DynamicGenDataset."""


__all__ = ["AbstractDataGenerator", "DynamicGenDataset"]
