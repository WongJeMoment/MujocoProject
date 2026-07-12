"""Train a PPO policy for GO2 stair traversal."""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env

from rl_stair_env import Go2StairEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "auto"),
        default="cpu",
        help="PPO with an MLP policy is normally faster on CPU.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "checkpoints" / "go2_stair_ppo",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    check_env(Go2StairEnv(), warn=True)
    env = make_vec_env(Go2StairEnv, n_envs=args.num_envs)
    eval_env = Go2StairEnv()
    checkpoint_dir = args.output.parent / "ppo_logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=args.rollout_steps,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=str(checkpoint_dir / "tensorboard"),
        device=args.device,
        verbose=1,
    )

    callbacks = [
        CheckpointCallback(
            save_freq=max(args.rollout_steps // 2, 1),
            save_path=str(checkpoint_dir / "checkpoints"),
            name_prefix="go2_stair_ppo",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=str(checkpoint_dir / "best_model"),
            log_path=str(checkpoint_dir / "eval"),
            eval_freq=max(args.rollout_steps // 2, 1),
            deterministic=True,
            render=False,
        ),
    ]

    model.learn(total_timesteps=args.timesteps, callback=callbacks, progress_bar=True)
    model.save(str(args.output))
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
