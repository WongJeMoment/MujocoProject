"""Gymnasium environment for learning GO2 stair traversal with PPO."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


MODEL_PATH = Path(__file__).parent / "official_mjcf" / "scene.xml"
LEG_NAMES = ("FR", "FL", "RR", "RL")
JOINT_NAMES = tuple(
    f"{leg}_{joint}_joint"
    for leg in LEG_NAMES
    for joint in ("hip", "thigh", "calf")
)
HOME = np.array((0.0, 0.9, -1.8) * 4, dtype=np.float32)
KP = np.array((55.0, 65.0, 78.0) * 4, dtype=np.float32)
KD = np.array((2.5, 3.2, 3.5) * 4, dtype=np.float32)
STEP_XS = np.array([1.2, 1.6, 2.3, 2.6, 2.8, 3.0, 3.2, 3.4], dtype=np.float32)
ACTION_SCALE = np.array((0.18, 0.28, 0.34) * 4, dtype=np.float32)


def _quat_rotate_inverse(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    w, x, y, z = quat
    q_xyz = np.array([x, y, z], dtype=np.float32)
    t = 2.0 * np.cross(q_xyz, vec)
    return vec - w * t + np.cross(q_xyz, t)


def _leg_cycle(
    phase: float, duty_factor: float, lift_bias: float = 0.0
) -> tuple[float, float]:
    if phase < duty_factor:
        stance = phase / duty_factor
        sweep = 1.0 - 2.0 * stance
        lift = 0.0
    else:
        swing = (phase - duty_factor) / (1.0 - duty_factor)
        sweep = -1.0 + 2.0 * swing
        lift = math.sin(math.pi * swing) ** (1.2 + lift_bias)
    return sweep, lift


def nominal_gait_targets(sim_time: float, obstacle_mode: bool) -> np.ndarray:
    if obstacle_mode:
        frequency = 1.00
        duty_factor = 0.80
        stride = 0.24
        lift_gain = 0.86
        hip_sway = 0.06
        phase_offsets = (0.00, 0.50, 0.75, 0.25)
    else:
        frequency = 1.45
        duty_factor = 0.68
        stride = 0.18
        lift_gain = 0.36
        hip_sway = 0.035
        phase_offsets = (0.00, 0.50, 0.50, 0.00)

    base_thigh = 0.92
    base_calf = -1.84
    targets: list[float] = []

    for leg, phase_offset in enumerate(phase_offsets):
        phase = (max(sim_time - 0.5, 0.0) * frequency + phase_offset) % 1.0
        sweep, step_lift = _leg_cycle(
            phase, duty_factor, lift_bias=0.25 if obstacle_mode else 0.0
        )
        side_sign = -1.0 if leg in (0, 2) else 1.0
        is_front = leg in (0, 1)
        front_reach = 0.10 if is_front else 0.04
        extra_lift = (
            0.18 if (obstacle_mode and is_front) else 0.08 if obstacle_mode else 0.0
        )

        hip = side_sign * hip_sway * step_lift
        thigh = base_thigh - stride * sweep - (0.16 + front_reach) * step_lift
        calf = base_calf + 0.16 * sweep - (lift_gain + extra_lift) * step_lift
        targets.extend((hip, thigh, calf))

    return np.asarray(targets, dtype=np.float32)


@dataclass(slots=True)
class StepResult:
    observation: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, float]


class Go2StairTask:
    """State-based stair traversal task for reinforcement learning."""

    def __init__(self, max_time: float = 12.0) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.data = mujoco.MjData(self.model)
        self.max_time = max_time
        self.base_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link"
        )
        self._joint_qposadr = np.zeros(len(JOINT_NAMES), dtype=np.int32)
        self._joint_dofadr = np.zeros(len(JOINT_NAMES), dtype=np.int32)
        for index, name in enumerate(JOINT_NAMES):
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self._joint_qposadr[index] = self.model.jnt_qposadr[joint_id]
            self._joint_dofadr[index] = self.model.jnt_dofadr[joint_id]

        self._last_x = 0.0
        self._passed_steps = 0
        self._rng = np.random.default_rng()

    @property
    def action_size(self) -> int:
        return len(JOINT_NAMES)

    @property
    def observation_size(self) -> int:
        # Joint state (24), base motion/orientation (9), height (1), step range (1).
        return 2 * len(JOINT_NAMES) + 3 + 3 + 3 + 1 + 1

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, float]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        qpos_noise = self._rng.uniform(-0.03, 0.03, size=len(JOINT_NAMES))
        self.data.qpos[self._joint_qposadr] = HOME + qpos_noise
        self.data.qvel[:] = 0.0
        self.data.qpos[0] = self._rng.uniform(-0.03, 0.03)
        self.data.qpos[1] = self._rng.uniform(-0.02, 0.02)
        self.data.qpos[2] = 0.445
        mujoco.mj_forward(self.model, self.data)

        self._last_x = float(self.data.qpos[0])
        self._passed_steps = 0
        return self.observation(), {}

    def observation(self) -> np.ndarray:
        quat = self.data.qpos[3:7].copy()
        joint_pos = self.data.qpos[self._joint_qposadr] - HOME
        joint_vel = self.data.qvel[self._joint_dofadr]
        world_lin_vel = self.data.qvel[:3]
        world_ang_vel = self.data.qvel[3:6]
        body_lin_vel = _quat_rotate_inverse(quat, world_lin_vel)
        body_gravity = _quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0], dtype=np.float32))
        base_height = np.array([self.data.qpos[2] - 0.42], dtype=np.float32)
        next_step_dx = self._next_step_distance()
        obs = np.concatenate(
            [
                joint_pos.astype(np.float32),
                joint_vel.astype(np.float32) * 0.15,
                body_lin_vel.astype(np.float32),
                world_ang_vel.astype(np.float32) * 0.2,
                body_gravity.astype(np.float32),
                base_height,
                np.array([next_step_dx], dtype=np.float32),
            ]
        )
        return np.clip(obs, -10.0, 10.0).astype(np.float32)

    def step(self, action: np.ndarray) -> StepResult:
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)
        obstacle_mode = self._next_step_distance() < 0.55
        nominal = nominal_gait_targets(float(self.data.time), obstacle_mode)
        targets = nominal + ACTION_SCALE * action

        for _ in range(5):
            torque = KP * (targets - self.data.qpos[self._joint_qposadr])
            torque -= KD * self.data.qvel[self._joint_dofadr]
            ctrl_low = self.model.actuator_ctrlrange[:, 0]
            ctrl_high = self.model.actuator_ctrlrange[:, 1]
            self.data.ctrl[:] = np.clip(torque, ctrl_low, ctrl_high)
            mujoco.mj_step(self.model, self.data)

        obs = self.observation()
        reward, info = self._reward(action)
        terminated = self._terminated()
        truncated = bool(self.data.time >= self.max_time)
        return StepResult(obs, reward, terminated, truncated, info)

    def _reward(self, action: np.ndarray) -> tuple[float, dict[str, float]]:
        x_pos = float(self.data.qpos[0])
        progress = x_pos - self._last_x
        self._last_x = x_pos

        quat = self.data.qpos[3:7]
        body_gravity = _quat_rotate_inverse(
            quat, np.array([0.0, 0.0, -1.0], dtype=np.float32)
        )
        upright_bonus = float(body_gravity[2])
        height = float(self.data.qpos[2])
        forward_vel = float(self.data.qvel[0])
        energy = float(np.mean(np.square(self.data.ctrl)))

        step_bonus = 0.0
        while self._passed_steps < len(STEP_XS) and x_pos > float(STEP_XS[self._passed_steps]) + 0.08:
            self._passed_steps += 1
            step_bonus += 2.0

        reward = (
            8.0 * progress
            + 0.35 * max(forward_vel, 0.0)
            + 0.25 * upright_bonus
            + step_bonus
            - 0.03 * float(np.mean(np.square(action)))
            - 0.0008 * energy
            - 1.5 * max(0.0, 0.30 - height)
        )
        info = {
            "x_position": x_pos,
            "forward_velocity": forward_vel,
            "step_bonus": step_bonus,
            "height": height,
        }
        return reward, info

    def _terminated(self) -> bool:
        quat = self.data.qpos[3:7]
        body_gravity = _quat_rotate_inverse(
            quat, np.array([0.0, 0.0, -1.0], dtype=np.float32)
        )
        too_low = self.data.qpos[2] < 0.20
        tipped = body_gravity[2] > -0.2
        too_far_sideways = abs(self.data.qpos[1]) > 0.55
        return bool(too_low or tipped or too_far_sideways)

    def _next_step_distance(self) -> float:
        x_pos = float(self.data.qpos[0])
        future_steps = STEP_XS[STEP_XS >= x_pos]
        if len(future_steps) == 0:
            return 2.0
        return float(np.clip(future_steps[0] - x_pos, 0.0, 2.0))


class Go2StairEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(self, max_time: float = 12.0) -> None:
        super().__init__()
        self.task = Go2StairTask(max_time=max_time)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.task.action_size,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(self.task.observation_size,),
            dtype=np.float32,
        )

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        del options
        super().reset(seed=seed)
        return self.task.reset(seed=seed)

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, float]]:
        result = self.task.step(action)
        return (
            result.observation,
            result.reward,
            result.terminated,
            result.truncated,
            result.info,
        )
