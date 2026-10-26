# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp
from rl_training.tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    ActionsCfg,
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
)

##
# Pre-defined configs
##
from rl_training.assets.deeprobotics import WHEELED_DOG_V2_CFG  # isort: skip


@configclass
class WheeledDogV2ActionsCfg(ActionsCfg):
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[""], scale=0.25, use_default_offset=True, clip=None, preserve_order=True
    )

    joint_vel = mdp.JointVelocityActionCfg(
        asset_name="robot", joint_names=[""], scale=5.0, use_default_offset=True, clip=None, preserve_order=True
    )


@configclass
class WheeledDogV2RewardsCfg(RewardsCfg):
    """Reward terms for the MDP."""

    joint_vel_wheel_l2 = RewTerm(
        func=mdp.joint_vel_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    )

    joint_acc_wheel_l2 = RewTerm(
        func=mdp.joint_acc_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    )

    joint_torques_wheel_l2 = RewTerm(
        func=mdp.joint_torques_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    )


@configclass
class WheeledDogV2RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    actions: WheeledDogV2ActionsCfg = WheeledDogV2ActionsCfg()
    rewards: WheeledDogV2RewardsCfg = WheeledDogV2RewardsCfg()

    base_link_name = "base_link"
    foot_link_name = ".*_W_L"

    # fmt: off
    # wheeldog.SLDASM joint names (H1=hipx, H2=hipy, K=knee, W=wheel)
    leg_joint_names = [
        "FL_H1_L_j1", "FL_H2_L_j2", "FL_K_L_j3",
        "FR_H1_L_j1", "FR_H2_L_j2", "FR_K_L_j3",
        "RL_H1_L_j1", "RL_H2_L_j2", "RL_K_L_j3",
        "RR_H1_L_j1", "RR_H2_L_j2", "RR_K_L_j3",
    ]
    wheel_joint_names = [
        "FL_W_L_j4", "FR_W_L_j4", "RL_W_L_j4", "RR_W_L_j4",
    ]

    hipx_joint_names = [
        "FL_H1_L_j1", "FR_H1_L_j1", "RL_H1_L_j1", "RR_H1_L_j1",
    ]

    hipy_joint_names = [
        "FL_H2_L_j2", "FR_H2_L_j2", "RL_H2_L_j2", "RR_H2_L_j2",
    ]

    knee_joint_names = [
        "FL_K_L_j3", "FR_K_L_j3", "RL_K_L_j3", "RR_K_L_j3",
    ]
    joint_names = leg_joint_names + wheel_joint_names
    # fmt: on

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ------------------------------Scene------------------------------
        self.scene.robot = WHEELED_DOG_V2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name

        # Start curriculum easier (was 5 → stuck on mid stairs without climb skill).
        self.scene.terrain.max_init_terrain_level = 2

        # ------------------------------Observations------------------------------
        self.observations.policy.joint_pos.func = mdp.joint_pos_rel_without_wheel
        self.observations.policy.joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.wheel_joint_names
        )
        self.observations.critic.joint_pos.func = mdp.joint_pos_rel_without_wheel
        self.observations.critic.joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.wheel_joint_names
        )
        self.observations.policy.base_lin_vel.scale = 2.0
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.policy.base_lin_vel = None
        self.observations.policy.height_scan = None
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names

        # ------------------------------Actions------------------------------
        # reduce action scale (H1 = hip roll)
        self.actions.joint_pos.scale = {".*_H1_L_j1": 0.125, "^(?!.*_H1_L_j1).*": 0.25}
        self.actions.joint_vel.scale = 5.0
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_vel.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_pos.joint_names = self.leg_joint_names
        self.actions.joint_vel.joint_names = self.wheel_joint_names

        # ------------------------------Events------------------------------
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-1.0, 1.0),
                "y": (-1.0, 1.0),
                "z": (0.0, 0.0),
                "roll": (-0.3, 0.3),
                "pitch": (-0.3, 0.3),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "z": (-0.2, 0.2),
                "roll": (-0.05, 0.05),
                "pitch": (-0.05, 0.05),
                "yaw": (-0.0, 0.0),
            },
        }
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_rigid_body_mass.params["asset_cfg"].body_names = [
            f"^(?!.*{self.base_link_name}).*"
        ]
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]

        # ------------------------------Sub-terrains (stairs up/down dominant)------------------------------
        sub_terrains = self.scene.terrain.terrain_generator.sub_terrains
        sub_terrains["pyramid_stairs"].proportion = 0.35
        sub_terrains["pyramid_stairs"].size = (8.0, 8.0)
        sub_terrains["pyramid_stairs"].flat_patch_sampling = None
        sub_terrains["pyramid_stairs"].border_width = 1.0
        sub_terrains["pyramid_stairs"].step_height_range = (0.02, 0.20)
        sub_terrains["pyramid_stairs"].step_width = 0.30
        sub_terrains["pyramid_stairs"].platform_width = 3.0
        sub_terrains["pyramid_stairs"].holes = False

        sub_terrains["pyramid_stairs_inv"].proportion = 0.35
        sub_terrains["pyramid_stairs_inv"].size = (8.0, 8.0)
        sub_terrains["pyramid_stairs_inv"].flat_patch_sampling = None
        sub_terrains["pyramid_stairs_inv"].border_width = 1.0
        sub_terrains["pyramid_stairs_inv"].step_height_range = (0.02, 0.20)
        sub_terrains["pyramid_stairs_inv"].step_width = 0.30
        sub_terrains["pyramid_stairs_inv"].platform_width = 3.0
        sub_terrains["pyramid_stairs_inv"].holes = False

        sub_terrains["boxes"].proportion = 0.05
        sub_terrains["boxes"].grid_width = 0.45
        sub_terrains["boxes"].grid_height_range = (0.025, 0.2)
        sub_terrains["boxes"].platform_width = 2.0

        sub_terrains["random_rough"].proportion = 0.15
        sub_terrains["random_rough"].noise_range = (0.01, 0.1)
        sub_terrains["random_rough"].noise_step = 0.01

        sub_terrains["hf_pyramid_slope"].proportion = 0.05
        sub_terrains["hf_pyramid_slope"].slope_range = (0.0, 0.4)
        sub_terrains["hf_pyramid_slope_inv"].proportion = 0.05
        sub_terrains["hf_pyramid_slope_inv"].slope_range = (0.0, 0.4)

        self.events.randomize_rigid_body_material.params["static_friction_range"] = [0.2, 1.2]
        self.events.randomize_rigid_body_material.params["dynamic_friction_range"] = [0.2, 1.2]
        self.events.randomize_rigid_body_material.params["restitution_range"] = [0.1, 0.15]

        # ------------------------------Rewards------------------------------
        self.rewards.is_terminated.weight = 0

        # Root penalties — S10-style: allow pitch on stairs
        self.rewards.lin_vel_z_l2.weight = -1.0
        self.rewards.ang_vel_xy_l2.weight = -0.02
        self.rewards.flat_orientation_l2.weight = -15.0
        self.rewards.base_height_l2.weight = -0.0
        self.rewards.base_height_l2.params["target_height"] = 0.55
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.body_lin_acc_l2.weight = 0
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [self.base_link_name]

        # Joint penalties — lighter so legs can flex on steps
        self.rewards.joint_torques_l2.weight = -2.0e-5
        self.rewards.joint_torques_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_torques_wheel_l2.weight = 0
        self.rewards.joint_torques_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_vel_l2.weight = 0
        self.rewards.joint_vel_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_vel_wheel_l2.weight = 0
        self.rewards.joint_vel_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_acc_l2.weight = -2e-7
        self.rewards.joint_acc_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_acc_wheel_l2.weight = -1e-7
        self.rewards.joint_acc_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_pos_limits.weight = -0.0
        self.rewards.joint_pos_limits.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_vel_limits.weight = 0
        self.rewards.joint_vel_limits.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_power.weight = -0.0
        self.rewards.joint_power.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.stand_still.weight = -1.0
        self.rewards.stand_still.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.hipx_joint_pos_penalty.weight = -1.5
        self.rewards.hipx_joint_pos_penalty.params["asset_cfg"].joint_names = self.hipx_joint_names
        self.rewards.hipy_joint_pos_penalty.weight = -0.5
        self.rewards.hipy_joint_pos_penalty.params["asset_cfg"].joint_names = self.hipy_joint_names
        self.rewards.hipy_joint_pos_penalty.func = mdp.joint_pos_penalty_except_turn_side_cmd
        self.rewards.hipy_joint_pos_penalty.params["ang_cmd_threshold"] = 0.1
        self.rewards.hipy_joint_pos_penalty.params["y_cmd_threshold"] = 0.1
        self.rewards.hipy_joint_pos_penalty.params["xy_norm_max"] = 0.1
        self.rewards.hipy_joint_pos_penalty.params["xz_norm_max"] = 0.1
        self.rewards.hipy_joint_pos_penalty.params["sensor_cfg"] = SceneEntityCfg("height_scanner_base")
        self.rewards.hipy_joint_pos_penalty.params["terrain_height_threshold"] = 0.06
        self.rewards.hipy_joint_pos_penalty.params["high_terrain_penalty_scale"] = 0.05
        self.rewards.knee_joint_pos_penalty.weight = -0.3
        self.rewards.knee_joint_pos_penalty.params["asset_cfg"].joint_names = self.knee_joint_names
        self.rewards.knee_joint_pos_penalty.func = mdp.joint_pos_penalty_except_turn_side_cmd
        self.rewards.knee_joint_pos_penalty.params["ang_cmd_threshold"] = 0.1
        self.rewards.knee_joint_pos_penalty.params["y_cmd_threshold"] = 0.1
        self.rewards.knee_joint_pos_penalty.params["xy_norm_max"] = 0.1
        self.rewards.knee_joint_pos_penalty.params["xz_norm_max"] = 0.1
        self.rewards.knee_joint_pos_penalty.params["sensor_cfg"] = SceneEntityCfg("height_scanner_base")
        self.rewards.knee_joint_pos_penalty.params["terrain_height_threshold"] = 0.06
        self.rewards.knee_joint_pos_penalty.params["high_terrain_penalty_scale"] = 0.05
        self.rewards.wheel_vel_penalty.weight = 0
        self.rewards.wheel_vel_penalty.params["sensor_cfg"].body_names = self.foot_link_name
        self.rewards.wheel_vel_penalty.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_mirror.weight = 0.0
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["FL_(H1|H2|K)_L_j.*", "RR_(H1|H2|K)_L_j.*"],
            ["FR_(H1|H2|K)_L_j.*", "RL_(H1|H2|K)_L_j.*"],
        ]

        # Action penalties
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.action_smooth_l2.weight = -0.02

        # Contact sensor
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.rewards.contact_forces.weight = -1.5e-4
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        # Velocity-tracking — stronger so agents walk farther → terrain level up
        self.rewards.track_lin_vel_xy_exp.weight = 6.0
        self.rewards.track_lin_vel_xy_exp.params["std"] = math.sqrt(0.15)
        self.rewards.track_ang_vel_z_exp.weight = 2.0
        self.rewards.track_ang_vel_z_exp.params["std"] = math.sqrt(0.25)

        # Others
        self.rewards.feet_air_time.weight = 0
        self.rewards.feet_air_time.params["threshold"] = 0.5
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact.weight = 0
        self.rewards.feet_contact.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact_without_cmd.weight = 0.0
        self.rewards.feet_contact_without_cmd.params["sensor_cfg"].body_names = [self.foot_link_name]
        # Disable flat-turn air-time (conflicts with stair climbing)
        self.rewards.feet_air_time_ang_z_M20.weight = 0.0
        self.rewards.feet_air_time_ang_z_M20.params["threshold"] = 0.2
        self.rewards.feet_air_time_ang_z_M20.params["foot_height_threshold"] = 0.00
        self.rewards.feet_air_time_ang_z_M20.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_air_time_ang_z_M20.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_stumble.weight = -0.1
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.weight = 0
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide_ang_z_cmd.weight = -1.0
        self.rewards.feet_slide_ang_z_cmd.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide_ang_z_cmd.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height.weight = 0
        self.rewards.feet_height.params["target_height"] = 0.1
        self.rewards.feet_height.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height_body.weight = 0
        self.rewards.feet_height_body.params["target_height"] = -0.4
        self.rewards.feet_height_body.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_gait.weight = 0
        self.rewards.feet_gait.params["synced_feet_pair_names"] = (("FL_W_L", "RR_W_L"), ("FR_W_L", "RL_W_L"))
        self.rewards.upward.weight = 0.0

        # Rotation gait — keep mild
        self.rewards.rotation_gait_status.weight = 1.0
        self.rewards.rotation_gait_status.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.rotation_gait_status.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.rotation_gait_status.params["group_a_body_names"] = ["FL_W_L", "RR_W_L"]
        self.rewards.rotation_gait_status.params["group_b_body_names"] = ["FR_W_L", "RL_W_L"]

        self.rewards.rotation_gait_symmetry.weight = 5.0
        self.rewards.rotation_gait_symmetry.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.rotation_gait_symmetry.params["group_a_body_names"] = ["FL_W_L", "RR_W_L"]
        self.rewards.rotation_gait_symmetry.params["group_b_body_names"] = ["FR_W_L", "RL_W_L"]

        self.rewards.bad_orientation_penalty.weight = -80.0
        if self.__class__.__name__ == "WheeledDogV2RoughEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------
        self.terminations.illegal_contact = None
        # Don't kill episodes for mild pitch on stairs
        self.terminations.bad_orientation_2 = None

        # ------------------------------Curriculums------------------------------
        self.curriculum.command_levels = None

        # ------------------------------Commands------------------------------
        self.commands.base_velocity.debug_vis = False
        # Favor forward locomotion on stairs (less pure yaw / zero-vel)
        self.commands.base_velocity.ranges.lin_vel_x = (-1.5, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.6, 0.6)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        self.commands.base_velocity.rel_standing_envs = 0.02
        self.commands.base_velocity.rel_zero_vel_envs = 0.05
        self.commands.base_velocity.rel_only_lin_y_envs = 0.05
        self.commands.base_velocity.rel_only_lin_x_envs = 0.25
        self.commands.base_velocity.rel_only_ang_z_envs = 0.05
