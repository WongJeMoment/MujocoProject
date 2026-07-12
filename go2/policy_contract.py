"""Shared observation and action contract for GO2 simulation and hardware."""

from __future__ import annotations

import math

import numpy as np


LEG_NAMES = ("FR", "FL", "RR", "RL")
JOINT_NAMES = tuple(
    f"{leg}_{joint}_joint"
    for leg in LEG_NAMES
    for joint in ("hip", "thigh", "calf")
)
HOME = np.array((0.0, 0.9, -1.8) * 4, dtype=np.float32)
KP = np.array((55.0, 65.0, 78.0) * 4, dtype=np.float32)
KD = np.array((2.5, 3.2, 3.5) * 4, dtype=np.float32)
ACTION_SCALE = np.array((0.18, 0.28, 0.34) * 4, dtype=np.float32)
NOMINAL_BASE_HEIGHT = 0.42
OBSERVATION_SIZE = 35

# Unitree and the MJCF both use FR, FL, RR, RL with hip, thigh, calf per leg.
JOINT_LIMIT_LOW = np.array(
    (-1.0472, -1.5708, -2.7227) * 2
    + (-1.0472, -0.5236, -2.7227) * 2,
    dtype=np.float32,
)
JOINT_LIMIT_HIGH = np.array(
    (1.0472, 3.4907, -0.83776) * 2
    + (1.0472, 4.5379, -0.83776) * 2,
    dtype=np.float32,
)


def quat_rotate_inverse(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate a world-frame vector into the body frame using a wxyz quaternion."""
    quat = np.asarray(quat, dtype=np.float32)
    quat = quat / max(float(np.linalg.norm(quat)), 1e-6)
    w, x, y, z = quat
    q_xyz = np.array([x, y, z], dtype=np.float32)
    vector = np.asarray(vec, dtype=np.float32)
    t = 2.0 * np.cross(q_xyz, vector)
    return vector - w * t + np.cross(q_xyz, t)


def projected_gravity(quat: np.ndarray) -> np.ndarray:
    return quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0], dtype=np.float32))


def build_observation(
    joint_position: np.ndarray,
    joint_velocity: np.ndarray,
    body_linear_velocity: np.ndarray,
    body_angular_velocity: np.ndarray,
    gravity_body: np.ndarray,
    base_height: float,
    step_distance: float,
) -> np.ndarray:
    """Build the exact 35-value observation consumed by the PPO policy."""
    observation = np.concatenate(
        (
            np.asarray(joint_position, dtype=np.float32) - HOME,
            np.asarray(joint_velocity, dtype=np.float32) * 0.15,
            np.asarray(body_linear_velocity, dtype=np.float32),
            np.asarray(body_angular_velocity, dtype=np.float32) * 0.2,
            np.asarray(gravity_body, dtype=np.float32),
            np.array([base_height - NOMINAL_BASE_HEIGHT], dtype=np.float32),
            np.array([np.clip(step_distance, 0.0, 2.0)], dtype=np.float32),
        )
    )
    if observation.shape != (OBSERVATION_SIZE,):
        raise ValueError(
            f"Policy observation must have shape ({OBSERVATION_SIZE},), "
            f"got {observation.shape}."
        )
    return np.clip(observation, -10.0, 10.0).astype(np.float32)


def _leg_cycle(
    phase: float, duty_factor: float, lift_bias: float = 0.0
) -> tuple[float, float]:
    if phase < duty_factor:
        stance = phase / duty_factor
        return 1.0 - 2.0 * stance, 0.0

    swing = (phase - duty_factor) / (1.0 - duty_factor)
    lift = math.sin(math.pi * swing) ** (1.2 + lift_bias)
    return -1.0 + 2.0 * swing, lift


def nominal_gait_targets(elapsed_time: float, obstacle_mode: bool) -> np.ndarray:
    """Generate the same gait prior on both MuJoCo and the real robot."""
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

    targets: list[float] = []
    for leg, phase_offset in enumerate(phase_offsets):
        phase = (max(elapsed_time - 0.5, 0.0) * frequency + phase_offset) % 1.0
        sweep, step_lift = _leg_cycle(
            phase, duty_factor, lift_bias=0.25 if obstacle_mode else 0.0
        )
        side_sign = -1.0 if leg in (0, 2) else 1.0
        is_front = leg in (0, 1)
        front_reach = 0.10 if is_front else 0.04
        extra_lift = (
            0.18 if (obstacle_mode and is_front) else 0.08 if obstacle_mode else 0.0
        )
        targets.extend(
            (
                side_sign * hip_sway * step_lift,
                0.92 - stride * sweep - (0.16 + front_reach) * step_lift,
                -1.84 + 0.16 * sweep - (lift_gain + extra_lift) * step_lift,
            )
        )
    return np.asarray(targets, dtype=np.float32)


def policy_action_to_target(
    action: np.ndarray,
    elapsed_time: float,
    step_distance: float,
    action_scale: float = 1.0,
) -> np.ndarray:
    action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
    obstacle_mode = step_distance < 0.55
    target = nominal_gait_targets(elapsed_time, obstacle_mode)
    target += ACTION_SCALE * float(action_scale) * action
    return np.clip(target, JOINT_LIMIT_LOW + 0.03, JOINT_LIMIT_HIGH - 0.03)
