# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

import gymnasium as gym

from . import agents

gym.register(
    id="Flat-Wheeled-Dog-V2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:WheeledDogV2FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheeledDogV2FlatPPORunnerCfg",
    },
)

gym.register(
    id="Rough-Wheeled-Dog-V2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:WheeledDogV2RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheeledDogV2RoughPPORunnerCfg",
    },
)
