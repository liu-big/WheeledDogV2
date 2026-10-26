#!/usr/bin/env python3
"""WheeledDogV2 策略 MuJoCo 部署 — 默认锁轮点足爬楼。

默认:
  - 指令全 0，站立不动（你不按键就不会走）
  - 轮子锁死当足；策略只驱动 12 个腿关节
  - 小键盘发速度指令（同点足 DIAN）

键位:
  8/2  前进/后退 (vx)
  4/6  左转/右转 (yaw)
  7/9  左移/右移 (vy)
  0/5  停止（指令清零 → 站立）
  L    锁轮（默认）
  U    解锁轮（才把策略的轮速动作接上）
  R    重置
  P    暂停

用法:
  python run_policy_wheeldog.py --scene 15up
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
DEFAULT_CKPT = REPO / "checkpoints" / "model_38700.pt"
SCENES = {
    "15up": ROOT / "scenes" / "stairs_15cm_up.xml",
    "15down": ROOT / "scenes" / "stairs_15cm_down.xml",
    "20up": ROOT / "scenes" / "stairs_20cm_up.xml",
    "20down": ROOT / "scenes" / "stairs_20cm_down.xml",
    "all": ROOT / "scenes" / "stairs_15_20_all.xml",
}

LEG_JOINTS = [
    "FL_H1_L_j1", "FL_H2_L_j2", "FL_K_L_j3",
    "FR_H1_L_j1", "FR_H2_L_j2", "FR_K_L_j3",
    "RL_H1_L_j1", "RL_H2_L_j2", "RL_K_L_j3",
    "RR_H1_L_j1", "RR_H2_L_j2", "RR_K_L_j3",
]
WHEEL_JOINTS = ["FL_W_L_j4", "FR_W_L_j4", "RL_W_L_j4", "RR_W_L_j4"]
ALL_JOINTS = LEG_JOINTS + WHEEL_JOINTS

DEFAULT_Q = {
    "FL_H1_L_j1": 0.12, "FR_H1_L_j1": -0.12, "RL_H1_L_j1": -0.12, "RR_H1_L_j1": 0.12,
    "FL_H2_L_j2": 0.35, "FR_H2_L_j2": -0.35, "RL_H2_L_j2": -0.35, "RR_H2_L_j2": 0.35,
    "FL_K_L_j3": 0.70, "FR_K_L_j3": -0.70, "RL_K_L_j3": -0.70, "RR_K_L_j3": 0.70,
    "FL_W_L_j4": 0.0, "FR_W_L_j4": 0.0, "RL_W_L_j4": 0.0, "RR_W_L_j4": 0.0,
}

LEG_ACTION_SCALE = np.array([0.125 if "H1" in n else 0.25 for n in LEG_JOINTS], dtype=np.float64)
WHEEL_VEL_SCALE = 5.0

OBS_DIM, ACT_DIM = 57, 16
DECIMATION = 4
SIM_DT = 0.005
ROOT_Z = 0.55
SETTLE_STEPS = 1200
CMD_DEADZONE = 0.05  # |cmd|<此 → 纯站立，不跑策略

NUMPAD_STEP = 0.1
CMD_LIMITS = [(-1.5, 1.5), (-0.6, 0.6), (-1.0, 1.0)]
_KP = {i: 320 + i for i in range(10)}
_DIGIT = {i: ord("0") + i for i in range(10)}


def _codes(*ds: int) -> set[int]:
    out: set[int] = set()
    for d in ds:
        out.add(_KP[d])
        out.add(_DIGIT[d])
    return out


KEY_STOP = _codes(0, 5)
KEY_MAP = {
    **{c: (0, +1.0) for c in _codes(8)},
    **{c: (0, -1.0) for c in _codes(2)},
    **{c: (2, +1.0) for c in _codes(4)},
    **{c: (2, -1.0) for c in _codes(6)},
    **{c: (1, +1.0) for c in _codes(7)},
    **{c: (1, -1.0) for c in _codes(9)},
}


def quat_rotate_inverse(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, x, y, z = q_wxyz / max(np.linalg.norm(q_wxyz), 1e-12)
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
    return R.T @ v


class ActorMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(57, 512), nn.ELU(),
            nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 16),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(obs)


def load_actor(ckpt: Path) -> ActorMLP:
    blob = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    sd = blob["actor_state_dict"]
    actor = ActorMLP()
    actor.load_state_dict({k: v for k, v in sd.items() if k.startswith("mlp.")}, strict=True)
    actor.eval()
    print(f"[policy] {ckpt}  iter={blob.get('iter', '?')}", flush=True)
    return actor


class WheeledDogController:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, ckpt: Path):
        self.model = model
        self.data = data
        self.actor = load_actor(ckpt)
        self.command = np.zeros(3, dtype=np.float64)
        self.last_action = np.zeros(ACT_DIM, dtype=np.float32)
        self.wheel_locked = True  # 默认锁轮点足

        joint_to_act = {int(model.actuator_trnid[a, 0]): a for a in range(model.nu)}
        self.qpos_adr, self.qvel_adr, self.act_ids = [], [], []
        for name in ALL_JOINTS:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise RuntimeError(f"missing joint {name}")
            self.qpos_adr.append(int(model.jnt_qposadr[jid]))
            self.qvel_adr.append(int(model.jnt_dofadr[jid]))
            self.act_ids.append(joint_to_act[jid])
        self.qpos_adr = np.asarray(self.qpos_adr)
        self.qvel_adr = np.asarray(self.qvel_adr)
        self.act_ids = np.asarray(self.act_ids)
        self.default_q = np.asarray([DEFAULT_Q[n] for n in ALL_JOINTS], dtype=np.float64)

    def reset(self) -> None:
        self.last_action[:] = 0.0
        self.command[:] = 0.0
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:3] = [0.0, 0.0, ROOT_Z]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qpos[self.qpos_adr] = self.default_q
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.settle()
        # 保证进场零速
        self.data.qvel[:] = 0.0
        self.command[:] = 0.0
        self.last_action[:] = 0.0
        self.apply_actuators()

    def settle(self, steps: int = SETTLE_STEPS) -> None:
        self.last_action[:] = 0.0
        for _ in range(steps):
            self.apply_actuators()
            mujoco.mj_step(self.model, self.data)
            if self.wheel_locked:
                self._hard_lock_wheels()
        self.data.qvel[:] = 0.0

    def _hard_lock_wheels(self) -> None:
        """锁轮：速度指令 0 + 关节角速度清零。"""
        for i in range(4):
            self.data.qvel[self.qvel_adr[12 + i]] = 0.0
            # 角位置钉在 0，避免滚动累积
            self.data.qpos[self.qpos_adr[12 + i]] = 0.0

    def build_obs(self) -> np.ndarray:
        ang = self.data.qvel[3:6].astype(np.float32) * 0.25
        grav = quat_rotate_inverse(
            self.data.qpos[3:7].astype(np.float64), np.array([0.0, 0.0, -1.0])
        ).astype(np.float32)
        q = self.data.qpos[self.qpos_adr].astype(np.float64)
        dq = self.data.qvel[self.qvel_adr].astype(np.float64)
        if self.wheel_locked:
            dq[12:16] = 0.0
        q_rel = (q - self.default_q).astype(np.float32)
        q_rel[12:16] = 0.0
        # last_action 里轮维在锁轮时对策略反馈为 0（点足）
        last = self.last_action.copy()
        if self.wheel_locked:
            last[12:16] = 0.0
        obs = np.concatenate(
            [
                ang,
                grav,
                self.command.astype(np.float32),
                q_rel,
                (dq * 0.05).astype(np.float32),
                last,
            ]
        )
        if obs.shape != (OBS_DIM,):
            raise RuntimeError(f"obs {obs.shape} != {OBS_DIM}")
        return obs

    def apply_actuators(self) -> None:
        """腿接策略位置；轮默认锁死（vel=0），解锁后才用策略轮速。"""
        action = self.last_action.astype(np.float64)
        leg_tgt = self.default_q[:12] + LEG_ACTION_SCALE * action[:12]
        for i in range(12):
            aid = int(self.act_ids[i])
            lo, hi = self.model.actuator_ctrlrange[aid]
            self.data.ctrl[aid] = float(np.clip(leg_tgt[i], lo, hi))

        if self.wheel_locked:
            wheel_vel = np.zeros(4, dtype=np.float64)
        else:
            wheel_vel = WHEEL_VEL_SCALE * action[12:16]
        for i in range(4):
            aid = int(self.act_ids[12 + i])
            lo, hi = self.model.actuator_ctrlrange[aid]
            self.data.ctrl[aid] = float(np.clip(wheel_vel[i], lo, hi))

    def policy_step(self) -> None:
        # 无速度指令 → 站立，不调用策略（避免一进场就动）
        if float(np.linalg.norm(self.command)) < CMD_DEADZONE:
            self.last_action[:] = 0.0
            self.apply_actuators()
            return
        obs = self.build_obs()
        with torch.no_grad():
            action = self.actor(torch.from_numpy(obs).unsqueeze(0)).squeeze(0).numpy()
        action = np.asarray(action, dtype=np.float64)
        action[:12] = np.clip(action[:12], -5.0, 5.0)
        action[12:16] = np.clip(action[12:16], -2.0, 2.0)
        if self.wheel_locked:
            action[12:16] = 0.0
        self.last_action = action.astype(np.float32)
        self.apply_actuators()

    def status(self) -> str:
        lock = "锁轮" if self.wheel_locked else "轮解锁"
        return (
            f"[{lock}] cmd=({self.command[0]:+.2f},{self.command[1]:+.2f},{self.command[2]:+.2f}) "
            f"z={self.data.qpos[2]:.3f} vx={self.data.qvel[0]:+.2f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="15up", choices=list(SCENES.keys()))
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--duration", type=float, default=0.0)
    ap.add_argument("--unlock-wheels", action="store_true", help="启动时解锁轮（默认锁）")
    args = ap.parse_args()

    scene = SCENES[args.scene]
    print(f"[wheeldog] scene={scene}", flush=True)
    model = mujoco.MjModel.from_xml_path(str(scene))
    model.opt.timestep = SIM_DT
    data = mujoco.MjData(model)
    ctrl = WheeledDogController(model, data, args.ckpt.expanduser().resolve())
    ctrl.wheel_locked = not args.unlock_wheels
    ctrl.reset()
    print(f"[wheeldog] settled z={data.qpos[2]:.3f}  {ctrl.status()}", flush=True)
    print(
        "[wheeldog] 8/2=vx 4/6=yaw 7/9=vy 0/5=停 | L=锁轮 U=解锁轮 | R=重置 P=暂停",
        flush=True,
    )
    print("[wheeldog] 默认锁轮站立；按 8 等发指令后策略才动腿", flush=True)

    paused = {"v": False}

    def on_key(keycode: int) -> None:
        if keycode in (ord("P"), ord("p")):
            paused["v"] = not paused["v"]
            print("paused" if paused["v"] else "running", flush=True)
            return
        if keycode in (ord("L"), ord("l")):
            ctrl.wheel_locked = True
            ctrl.last_action[12:16] = 0.0
            print(f"锁轮 | {ctrl.status()}", flush=True)
            return
        if keycode in (ord("U"), ord("u")):
            ctrl.wheel_locked = False
            print(f"轮解锁（策略轮速生效）| {ctrl.status()}", flush=True)
            return
        if keycode in (ord("R"), ord("r")):
            locked = ctrl.wheel_locked
            ctrl.reset()
            ctrl.wheel_locked = locked
            print(f"reset | {ctrl.status()}", flush=True)
            return
        if keycode in KEY_STOP:
            ctrl.command[:] = 0.0
            ctrl.last_action[:] = 0.0
        elif keycode in KEY_MAP:
            axis, sign = KEY_MAP[keycode]
            lo, hi = CMD_LIMITS[axis]
            ctrl.command[axis] = float(
                np.clip(ctrl.command[axis] + sign * NUMPAD_STEP, lo, hi)
            )
        else:
            return
        print(ctrl.status(), flush=True)

    phys_i = 0
    t0 = time.time()
    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        viewer.cam.distance = 5.0
        while viewer.is_running():
            if args.duration > 0 and time.time() - t0 > args.duration:
                break
            step_t = time.time()
            if not paused["v"]:
                if phys_i % DECIMATION == 0:
                    ctrl.policy_step()
                else:
                    ctrl.apply_actuators()
                mujoco.mj_step(model, data)
                if ctrl.wheel_locked:
                    ctrl._hard_lock_wheels()
                phys_i += 1
            viewer.cam.lookat[:] = data.qpos[:3]
            viewer.sync()
            dt = model.opt.timestep - (time.time() - step_t)
            if dt > 0:
                time.sleep(dt)


if __name__ == "__main__":
    main()
