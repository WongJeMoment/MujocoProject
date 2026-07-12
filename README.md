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

## GO2 sim-to-real

The stair environment enables domain randomization during training (mass,
inertia, friction, damping, motor strength, observation noise, and action
delay). Because the deployable observation contract now uses body-frame IMU
angular velocity, retrain a new model before using the hardware runner:

```bash
python go2/train_ppo.py --timesteps 1000000 \
  --output go2/checkpoints/go2_sim2real_ppo
```

Test the complete policy/control loop without a robot:

```bash
python go2/run_unitree_policy.py --mock --duration 10 \
  --model go2/checkpoints/go2_sim2real_ppo.zip
```

For a real GO2, install Unitree's official `unitree_sdk2_python` package in the
same environment and connect the computer to the robot's wired network. The
runner subscribes to `rt/lowstate` and `rt/sportmodestate`. It remains in
dry-run mode unless motor control is explicitly confirmed:

```bash
# State/policy check only; no low-level command is published.
python go2/run_unitree_policy.py --network-interface eth0 \
  --model go2/checkpoints/go2_sim2real_ppo.zip

# Only after suspended-robot testing, emergency-stop preparation, and
# disabling the conflicting Unitree sport service.
python go2/run_unitree_policy.py --network-interface eth0 \
  --model go2/checkpoints/go2_sim2real_ppo.zip \
  --event-port 17001 --enable-motors --confirm REAL_GO2
```

An EVK4/Metavision detector can import `EventDistancePublisher` from
`go2/send_event_distance.py` and publish metric stair distance plus confidence.
The following command sends a synthetic 0.45 m detection for interface tests:

```bash
python go2/send_event_distance.py --distance 0.45 --confidence 0.9 \
  --duration 5 --port 17001
```

The real-robot runner uses a slow startup interpolation, reduced gains/action
scale, communication timeout, joint limits, velocity limits, tilt protection,
target-jump protection, and a damping stop. Keep the robot suspended for the
first motor-enabled test; these software checks do not replace a physical
emergency stop.

The original simplified XML files are kept at the top level of each robot
directory for comparison. The Python demos load `official_mjcf/scene.xml`.
