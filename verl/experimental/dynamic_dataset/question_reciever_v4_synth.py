# Copyright 2026
#
# Licensed under the Apache License, Version 2.0.
"""
V4 synth receiver entrypoint.

Uses the dual-loop coordinator implementation and exposes a stable v4 module path.
"""

from verl.experimental.dynamic_dataset.question_reciever_v3_solver_questioner import app
from verl.experimental.dynamic_dataset.question_reciever_v3_solver_questioner import main

__all__ = ["app", "main"]


if __name__ == "__main__":
    main()
