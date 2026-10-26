# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause
# 
# # Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""
Python module serving as a project/extension template.
"""

# Register Gym environments.
from .tasks import *

# Register UI extensions (optional — requires Omniverse runtime).
try:
    from .ui_extension_example import *  # noqa: F401, F403
except ModuleNotFoundError:
    pass
