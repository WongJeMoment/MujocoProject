"""Run a simple Franka Emika Panda simulation in MuJoCo."""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import mujoco


MODEL_PATH = Path(__file__).parent / "official_mjcf" / "scene.xml"
HOME = (0.0, -0.45, 0.0, -2.15, 0.0, 1.75, 0.78)
AMPLITUDE = (0.35, 0.18, 0.30, 0.20, 0.30, 0.18, 0.35)
FREQUENCY = (0.45, 0.32, 0.52, 0.27, 0.38, 0.48, 0.30)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without opening the MuJoCo viewer.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Simulation duration in seconds; 0 means run until the viewer closes.",
    )
    return parser.parse_args()


def set_controls(data: mujoco.MjData) -> None:
    """Generate smooth joint-space targets inside the Panda joint limits."""
    for index, (home, amplitude, frequency) in enumerate(
        zip(HOME, AMPLITUDE, FREQUENCY)
    ):
        phase = index * 0.55
        data.ctrl[index] = home + amplitude * math.sin(frequency * data.time + phase)

    # The official Panda MJCF maps gripper command 0..255 to 0..0.04 m.
    data.ctrl[7] = 135.0 + 120.0 * math.sin(0.7 * data.time)


def run_headless(model: mujoco.MjModel, data: mujoco.MjData, duration: float) -> None:
    end_time = duration if duration > 0.0 else 5.0
    while data.time < end_time:
        set_controls(data)
        mujoco.mj_step(model, data)

    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    x, y, z = data.xpos[hand_id]
    print(f"Simulation finished at t={data.time:.2f} s")
    print(f"End-effector position: x={x:.3f}, y={y:.3f}, z={z:.3f} m")


def run_viewer(model: mujoco.MjModel, data: mujoco.MjData, duration: float) -> None:
    import mujoco.viewer

    start_time = time.monotonic()
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            if duration > 0.0 and data.time >= duration:
                break

            step_start = time.monotonic()
            set_controls(data)
            mujoco.mj_step(model, data)
            viewer.sync()

            sleep_time = model.opt.timestep - (time.monotonic() - step_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

        wall_time = time.monotonic() - start_time
        print(
            f"Viewer closed after {data.time:.2f} simulated seconds "
            f"({wall_time:.2f} s wall time).",
            flush=True,
        )


def main() -> None:
    args = parse_args()
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    if args.headless:
        run_headless(model, data, args.duration)
    else:
        run_viewer(model, data, args.duration)
        # Avoid a GLFW shutdown crash seen with some Linux graphics drivers.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
