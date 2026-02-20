# Copyright 2026
#
# Licensed under the Apache License, Version 2.0.
"""V5 receiver entrypoint alias."""

from verl.experimental.dynamic_dataset.question_reciever_v5_solver_questioner import app
from verl.experimental.dynamic_dataset.question_reciever_v5_solver_questioner import main

__all__ = ["app", "main"]


if __name__ == "__main__":
    main()
