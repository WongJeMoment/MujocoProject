"""Gymnasium environment for learning GO2 stair traversal with PPO."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from policy_contract import (
    ACTION_SCALE,
    HOME,
    JOINT_NAMES,
    KD,
    KP,
    OBSERVATION_SIZE,
    build_observation,
    nominal_gait_targets,
    projected_gravity,
    quat_rotate_inverse,
)


MODEL_PATH = Path(__file__).parent / "official_mjcf" / "scene.xml"
STEP_XS = np.array([1.2, 1.6, 2.3, 2.6, 2.8, 3.0, 3.2, 3.4], dtype=np.float32)


@dataclass(slots=True)
class StepResult:
    observation: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, float]


class Go2StairTask:
    """State-based stair traversal task for reinforcement learning."""

    def __init__(
        self, max_time: float = 12.0, domain_randomization: bool = False
    ) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.data = mujoco.MjData(self.model)
        self.max_time = max_time
        self.domain_randomization = domain_randomization
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
        self._nominal_body_mass = self.model.body_mass.copy()
        self._nominal_body_inertia = self.model.body_inertia.copy()
        self._nominal_geom_friction = self.model.geom_friction.copy()
        self._nominal_dof_damping = self.model.dof_damping.copy()
        self._motor_strength = 1.0
        self._observation_noise = 0.0
        self._action_history: deque[np.ndarray] = deque(maxlen=1)

    @property
    def action_size(self) -> int:
        return len(JOINT_NAMES)

    @property
    def observation_size(self) -> int:
        return OBSERVATION_SIZE

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, float]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._randomize_model()
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
        for _ in range(self._action_history.maxlen or 1):
            self._action_history.append(np.zeros(self.action_size, dtype=np.float32))
        return self.observation(), {}

    def _randomize_model(self) -> None:
        self.model.body_mass[:] = self._nominal_body_mass
        self.model.body_inertia[:] = self._nominal_body_inertia
        self.model.geom_friction[:] = self._nominal_geom_friction
        self.model.dof_damping[:] = self._nominal_dof_damping
        self._motor_strength = 1.0
        self._observation_noise = 0.0
        action_delay = 0

        if self.domain_randomization:
            mass_scale = self._rng.uniform(0.88, 1.12, size=(self.model.nbody, 1))
            self.model.body_mass[:] *= mass_scale[:, 0]
            self.model.body_inertia[:] *= mass_scale
            friction_scale = self._rng.uniform(0.75, 1.45, size=(self.model.ngeom, 1))
            self.model.geom_friction[:] *= friction_scale
            self.model.dof_damping[:] *= self._rng.uniform(
                0.75, 1.35, size=self.model.nv
            )
            self._motor_strength = float(self._rng.uniform(0.82, 1.18))
            self._observation_noise = float(self._rng.uniform(0.002, 0.015))
            action_delay = int(self._rng.integers(0, 3))

        self._action_history = deque(maxlen=action_delay + 1)
        mujoco.mj_setConst(self.model, self.data)

    def observation(self) -> np.ndarray:
        quat = self.data.qpos[3:7].copy()
        joint_pos = self.data.qpos[self._joint_qposadr]
        joint_vel = self.data.qvel[self._joint_dofadr]
        world_lin_vel = self.data.qvel[:3]
        world_ang_vel = self.data.qvel[3:6]
        obs = build_observation(
            joint_position=joint_pos,
            joint_velocity=joint_vel,
            body_linear_velocity=quat_rotate_inverse(quat, world_lin_vel),
            body_angular_velocity=quat_rotate_inverse(quat, world_ang_vel),
            gravity_body=projected_gravity(quat),
            base_height=float(self.data.qpos[2]),
            step_distance=self._next_step_distance(),
        )
        if self._observation_noise > 0.0:
            obs += self._rng.normal(0.0, self._observation_noise, size=obs.shape)
        return np.clip(obs, -10.0, 10.0).astype(np.float32)

    def step(self, action: np.ndarray) -> StepResult:
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)
        self._action_history.append(action.copy())
        delayed_action = self._action_history[0]
        obstacle_mode = self._next_step_distance() < 0.55
        nominal = nominal_gait_targets(float(self.data.time), obstacle_mode)
        targets = nominal + ACTION_SCALE * delayed_action

        for _ in range(5):
            torque = KP * (targets - self.data.qpos[self._joint_qposadr])
            torque -= KD * self.data.qvel[self._joint_dofadr]
            ctrl_low = self.model.actuator_ctrlrange[:, 0]
            ctrl_high = self.model.actuator_ctrlrange[:, 1]
            self.data.ctrl[:] = np.clip(
                self._motor_strength * torque, ctrl_low, ctrl_high
            )
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
        body_gravity = projected_gravity(quat)
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
        body_gravity = projected_gravity(quat)
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

    def __init__(
        self, max_time: float = 12.0, domain_randomization: bool = True
    ) -> None:
        super().__init__()
        self.task = Go2StairTask(
            max_time=max_time, domain_randomization=domain_randomization
        )
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
