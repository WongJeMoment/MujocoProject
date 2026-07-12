# MuJoCo robot examples

Each robot directory contains official MJCF/mesh assets and a Python demo.
GO2 and GO2-W also include Unitree's official URDF description packages.

```bash
conda activate mujoco
python frank/run_franka.py
python go2/run_go2.py
python go2w/run_go2w.py
```

Add `--headless --duration 5` to run without opening the viewer.
`python go2/run_go2.py` now opens the MuJoCo viewer and, by default, a second
event-camera window that follows the robot body camera. It defaults to an
EVK4-like rendering mode (`--event-style evk4`) with a 16:9 event image. Use
`--no-event-camera` to disable it, or tune it with `--event-width`,
`--event-height`, `--event-fps`, `--event-threshold`,
`--event-accumulation-ms`, and `--event-refractory-ms`.
The GO2 scene now uses a pure white floor with dark stairs for clearer event
edges. Use `--auto-climb --gait trot` (or the older `--auto-jump` alias) to let
the GO2 switch into an obstacle-aware high-step walking mode when the event
camera sees stairs.

For a reinforcement-learning solution, install `gymnasium` and
`stable-baselines3`, then train PPO with:
`python go2/train_ppo.py --timesteps 1000000`
After training, run the learned policy with:
`python go2/run_trained_policy.py --model go2/checkpoints/go2_stair_ppo.zip`

The original simplified XML files are kept at the top level of each robot
directory for comparison. The Python demos load `official_mjcf/scene.xml`.
