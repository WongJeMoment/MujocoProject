"""Run a trained PPO policy for the GO2 stair task in the MuJoCo viewer."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
from stable_baselines3 import PPO

from rl_stair_env import Go2StairTask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).parent / "checkpoints" / "go2_stair_ppo.zip",
    )
    parser.add_argument("--duration", type=float, default=20.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy = PPO.load(str(args.model))
    task = Go2StairTask(max_time=max(args.duration, 12.0))
    obs, _ = task.reset()

    import mujoco.viewer

    with mujoco.viewer.launch_passive(task.model, task.data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = task.base_body_id
        viewer.cam.distance = 2.0

        while viewer.is_running() and task.data.time < args.duration:
            step_start = time.monotonic()
            action, _ = policy.predict(obs, deterministic=True)
            result = task.step(action)
            obs = result.observation
            viewer.sync()
            delay = task.model.opt.timestep * 5 - (time.monotonic() - step_start)
            if delay > 0.0:
                time.sleep(delay)
            if result.terminated or result.truncated:
                obs, _ = task.reset()


if __name__ == "__main__":
    main()
