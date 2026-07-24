#!/usr/bin/env python3
"""Entry point for closed-loop DoorBot GRU student evaluation.

The evaluation implementation is shared with eval_teacher.py so both policies
use exactly the same environment, termination accounting, CSV schema, and
stress-test overrides.
"""

import os
import runpy
import sys


if "--student_checkpoint" not in sys.argv:
    raise SystemExit(
        "eval_student.py requires --student_checkpoint PATH. "
        "Use scripts/test_student.sh for the standard evaluation."
    )

runpy.run_path(os.path.join(os.path.dirname(__file__), "eval_teacher.py"), run_name="__main__")
