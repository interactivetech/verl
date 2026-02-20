# Copyright 2026
#
# Licensed under the Apache License, Version 2.0.
"""
Synth QuestionReceiver entrypoint.

This reuses the stable V2 receiver implementation (queue + optional submitter
refill + /questions + /responses) and provides a dedicated module path for
synth experiments.
"""

from verl.experimental.dynamic_dataset.question_reciever_v2 import app
from verl.experimental.dynamic_dataset.question_reciever_v2 import main

__all__ = ["app", "main"]


if __name__ == "__main__":
    main()
