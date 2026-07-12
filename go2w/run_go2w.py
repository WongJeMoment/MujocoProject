"""Run Unitree's official GO2-W MuJoCo model and mesh assets."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import mujoco


MODEL_PATH = Path(__file__).parent / "official_mjcf" / "scene.xml"
LEG_NAMES = ("FR", "FL", "RR", "RL")
LEG_JOINT_NAMES = tuple(
    f"{leg}_{joint}_joint"
    for leg in LEG_NAMES
    for joint in ("hip", "thigh", "calf")
)
WHEEL_JOINT_NAMES = tuple(f"{leg}_wheel_joint" for leg in LEG_NAMES)
HOME = (0.0, 0.9, -1.8) * 4
KP = (45.0, 55.0, 60.0) * 4
KD = (2.0, 2.5, 2.5) * 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speed", type=float, default=6.0, help="Wheel speed in rad/s.")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=float, default=0.0)
    return parser.parse_args()


def get_addresses(
    model: mujoco.MjModel, names: tuple[str, ...]
) -> tuple[tuple[int, int], ...]:
    addresses = []
    for name in names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        addresses.append((model.jnt_qposadr[joint_id], model.jnt_dofadr[joint_id]))
    return tuple(addresses)


def initialize_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    leg_addresses: tuple[tuple[int, int], ...],
) -> None:
    for (qpos_address, _), target in zip(leg_addresses, HOME):
        data.qpos[qpos_address] = target
    mujoco.mj_forward(model, data)


def simulate(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    leg_addresses: tuple[tuple[int, int], ...],
    wheel_addresses: tuple[tuple[int, int], ...],
    speed: float,
) -> None:
    for index, ((qpos_address, dof_address), target) in enumerate(
        zip(leg_addresses, HOME)
    ):
        torque = KP[index] * (target - data.qpos[qpos_address])
        torque -= KD[index] * data.qvel[dof_address]
        low, high = model.actuator_ctrlrange[index]
        data.ctrl[index] = min(high, max(low, torque))

    for wheel_index, (_, dof_address) in enumerate(wheel_addresses, start=12):
        torque = 1.8 * (speed - data.qvel[dof_address])
        low, high = model.actuator_ctrlrange[wheel_index]
        data.ctrl[wheel_index] = min(high, max(low, torque))

    mujoco.mj_step(model, data)


def run_headless(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    leg_addresses: tuple[tuple[int, int], ...],
    wheel_addresses: tuple[tuple[int, int], ...],
    speed: float,
    duration: float,
) -> None:
    end_time = duration if duration > 0.0 else 5.0
    while data.time < end_time:
        simulate(model, data, leg_addresses, wheel_addresses, speed)
    x, y, z = data.xpos[1]
    print(f"Official GO2-W simulation finished at t={data.time:.2f} s")
    print(f"Base position: x={x:.3f}, y={y:.3f}, z={z:.3f} m")


def run_viewer(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    leg_addresses: tuple[tuple[int, int], ...],
    wheel_addresses: tuple[tuple[int, int], ...],
    speed: float,
    duration: float,
) -> None:
    import mujoco.viewer

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = 1
        viewer.cam.distance = 2.2
        while viewer.is_running():
            if duration > 0.0 and data.time >= duration:
                break
            step_start = time.monotonic()
            simulate(model, data, leg_addresses, wheel_addresses, speed)
            viewer.sync()
            delay = model.opt.timestep - (time.monotonic() - step_start)
            if delay > 0.0:
                time.sleep(delay)
    print(f"Official GO2-W viewer closed at t={data.time:.2f} s", flush=True)


def main() -> None:
    args = parse_args()
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    leg_addresses = get_addresses(model, LEG_JOINT_NAMES)
    wheel_addresses = get_addresses(model, WHEEL_JOINT_NAMES)
    initialize_pose(model, data, leg_addresses)

    if args.headless:
        run_headless(
            model, data, leg_addresses, wheel_addresses, args.speed, args.duration
        )
    else:
        run_viewer(
            model, data, leg_addresses, wheel_addresses, args.speed, args.duration
        )
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
