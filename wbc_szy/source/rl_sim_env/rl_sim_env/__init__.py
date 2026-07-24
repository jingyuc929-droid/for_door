# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Python module serving as a project/extension template.
"""

import os

# Register Gym environments.
from .tasks import *

# Register UI extensions.
from .ui_extension_example import *

RL_SIM_ENV_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
