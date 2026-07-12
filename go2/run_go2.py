"""Run Unitree's official GO2 MuJoCo model and mesh assets."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import os
import sys
import time
from pathlib import Path

import mujoco

from event_camera import EventCameraConfig, EventCameraViewer


MODEL_PATH = Path(__file__).parent / "official_mjcf" / "scene.xml"
LEG_NAMES = ("FR", "FL", "RR", "RL")
JOINT_NAMES = tuple(
    f"{leg}_{joint}_joint"
    for leg in LEG_NAMES
    for joint in ("hip", "thigh", "calf")
)
HOME = (0.0, 0.9, -1.8) * 4
KP = (45.0, 55.0, 60.0) * 4
KD = (2.0, 2.5, 2.5) * 4
TRAVERSE_KP = (52.0, 62.0, 72.0) * 4
TRAVERSE_KD = (2.5, 3.0, 3.2) * 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gait", choices=("stand", "trot"), default="stand")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument(
        "--event-camera",
        dest="event_camera",
        action="store_true",
        default=True,
        help="Open a second window with an event-camera style view.",
    )
    parser.add_argument(
        "--no-event-camera",
        dest="event_camera",
        action="store_false",
        help="Disable the extra event-camera window.",
    )
    parser.add_argument("--event-width", type=int, default=640)
    parser.add_argument("--event-height", type=int, default=360)
    parser.add_argument("--event-fps", type=float, default=60.0)
    parser.add_argument("--event-threshold", type=float, default=0.22)
    parser.add_argument(
        "--event-style",
        choices=("evk4", "polarity"),
        default="evk4",
        help="Event visualization style. 'evk4' mimics the EVK4/Metavision look.",
    )
    parser.add_argument(
        "--event-accumulation-ms",
        type=float,
        default=10.0,
        help="How long events stay visible in the event frame.",
    )
    parser.add_argument(
        "--event-refractory-ms",
        type=float,
        default=1.0,
        help="Suppress repeated same-polarity events for this long per pixel.",
    )
    parser.add_argument(
        "--auto-jump",
        "--auto-climb",
        action="store_true",
        help="Use obstacle-aware high-step walking when the event camera sees stairs.",
    )
    return parser.parse_args()


def joint_addresses(model: mujoco.MjModel) -> tuple[tuple[int, int], ...]:
    addresses = []
    for name in JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        addresses.append((model.jnt_qposadr[joint_id], model.jnt_dofadr[joint_id]))
    return tuple(addresses)


def target_positions(sim_time: float, gait: str) -> list[float]:
    if gait == "stand":
        return list(HOME)
    return procedural_gait_targets(sim_time, obstacle_mode=False)


def leg_cycle(
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


def procedural_gait_targets(sim_time: float, obstacle_mode: bool) -> list[float]:
    if obstacle_mode:
        frequency = 1.05
        duty_factor = 0.80
        stride = 0.26
        lift_gain = 0.82
        hip_sway = 0.06
        phase_offsets = (0.00, 0.50, 0.75, 0.25)
    else:
        frequency = 1.55
        duty_factor = 0.68
        stride = 0.20
        lift_gain = 0.38
        hip_sway = 0.035
        phase_offsets = (0.00, 0.50, 0.50, 0.00)

    base_thigh = 0.92
    base_calf = -1.84
    targets: list[float] = []

    for leg, phase_offset in enumerate(phase_offsets):
        phase = (max(sim_time - 0.6, 0.0) * frequency + phase_offset) % 1.0
        sweep, step_lift = leg_cycle(
            phase, duty_factor, lift_bias=0.25 if obstacle_mode else 0.0
        )
        side_sign = -1.0 if leg in (0, 2) else 1.0
        is_front = leg in (0, 1)
        front_reach = 0.10 if is_front else 0.04
        extra_lift = 0.18 if (obstacle_mode and is_front) else 0.08 if obstacle_mode else 0.0

        hip = side_sign * hip_sway * step_lift
        # Negative stride*sweep makes the robot step forward instead of backpedaling.
        thigh = base_thigh - stride * sweep - (0.16 + front_reach) * step_lift
        calf = base_calf + 0.16 * sweep - (lift_gain + extra_lift) * step_lift
        targets.extend((hip, thigh, calf))

    return targets


@dataclass(slots=True)
class ControlCommand:
    targets: list[float]
    kp: tuple[float, ...]
    kd: tuple[float, ...]


class ObstacleAwareTrotController:
    """Generate a smoother trot and switch to a higher-stepping gait near obstacles."""

    def __init__(self, cruise_gait: str) -> None:
        self._cruise_gait = "trot" if cruise_gait == "stand" else cruise_gait
        self._obstacle_mode_until = 0.0
        self._last_detection_time = -10.0
        self._phase_time = 0.0

    @property
    def state(self) -> str:
        return "high_step" if self._obstacle_mode_until > self._phase_time else "cruise"

    def update(
        self,
        sim_time: float,
        base_x: float,
        forward_velocity: float,
        step_detected: bool,
        step_score: float,
    ) -> ControlCommand:
        del base_x, forward_velocity
        self._phase_time = sim_time
        if (
            sim_time > 0.8
            and (step_detected or step_score > 0.60)
            and sim_time - self._last_detection_time > 0.18
        ):
            self._obstacle_mode_until = max(self._obstacle_mode_until, sim_time + 1.0)
            self._last_detection_time = sim_time

        obstacle_mode = sim_time < self._obstacle_mode_until
        gait_targets = procedural_gait_targets(sim_time, obstacle_mode)
        return ControlCommand(
            gait_targets,
            TRAVERSE_KP if obstacle_mode else KP,
            TRAVERSE_KD if obstacle_mode else KD,
        )


def simulate(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    addresses: tuple[tuple[int, int], ...],
    command: ControlCommand,
) -> None:
    for index, ((qpos_address, dof_address), target) in enumerate(
        zip(addresses, command.targets)
    ):
        torque = command.kp[index] * (target - data.qpos[qpos_address])
        torque -= command.kd[index] * data.qvel[dof_address]
        low, high = model.actuator_ctrlrange[index]
        data.ctrl[index] = min(high, max(low, torque))
    mujoco.mj_step(model, data)


def run_headless(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    addresses: tuple[tuple[int, int], ...],
    gait: str,
    duration: float,
    traversal_controller: ObstacleAwareTrotController | None = None,
    event_camera_config: EventCameraConfig | None = None,
) -> None:
    end_time = duration if duration > 0.0 else 5.0
    event_viewer: EventCameraViewer | None = None
    if traversal_controller is not None and event_camera_config is not None:
        event_viewer = EventCameraViewer(model, event_camera_config)

    try:
        while data.time < end_time:
            step_detected = False
            step_score = 0.0
            if event_viewer is not None:
                result = event_viewer.render_if_due(data)
                if result is not None:
                    step_detected = result.step_detected
                    step_score = result.step_score
            if traversal_controller is None:
                command = ControlCommand(target_positions(data.time, gait), KP, KD)
            else:
                command = traversal_controller.update(
                    data.time,
                    data.qpos[0],
                    data.qvel[0],
                    step_detected,
                    step_score,
                )
            simulate(model, data, addresses, command)
    finally:
        if event_viewer is not None:
            event_viewer.close()
    x, y, z = data.xpos[1]
    print(f"Official GO2 simulation finished at t={data.time:.2f} s")
    print(f"Base position: x={x:.3f}, y={y:.3f}, z={z:.3f} m")


def run_viewer(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    addresses: tuple[tuple[int, int], ...],
    gait: str,
    duration: float,
    event_camera_enabled: bool,
    event_camera_config: EventCameraConfig,
    auto_jump: bool,
) -> None:
    import mujoco.viewer

    event_viewer: EventCameraViewer | None = None
    traversal_controller = ObstacleAwareTrotController(gait) if auto_jump else None
    last_step_report = -1.0
    last_mode_report = -1.0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = 1
        viewer.cam.distance = 2.0
        if event_camera_enabled or auto_jump:
            try:
                event_viewer = EventCameraViewer(model, event_camera_config)
                print(
                    f"Event camera window opened with {event_viewer.backend} backend.",
                    flush=True,
                )
            except Exception as exc:
                print(f"Event camera disabled: {exc}", flush=True)

        try:
            while viewer.is_running():
                if duration > 0.0 and data.time >= duration:
                    break
                step_start = time.monotonic()
                step_detected = False
                step_score = 0.0
                if event_viewer is not None:
                    result = event_viewer.render_if_due(data)
                    if result is not None:
                        step_detected = result.step_detected
                        step_score = result.step_score
                        if (
                            (step_detected or result.step_score > 0.62)
                            and data.time - last_step_report > 0.4
                        ):
                            print(
                                f"Event camera detected stairs at t={data.time:.2f} s "
                                f"(score={result.step_score:.2f}).",
                                flush=True,
                            )
                            last_step_report = data.time
                if traversal_controller is None:
                    command = ControlCommand(
                        target_positions(data.time, gait),
                        KP,
                        KD,
                    )
                else:
                    previous_state = traversal_controller.state
                    command = traversal_controller.update(
                        data.time,
                        data.qpos[0],
                        data.qvel[0],
                        step_detected,
                        step_score,
                    )
                    if (
                        previous_state == "cruise"
                        and traversal_controller.state != "cruise"
                        and data.time - last_mode_report > 0.5
                    ):
                        print(
                            f"High-step gait engaged at t={data.time:.2f} s "
                            f"(score={step_score:.2f}).",
                            flush=True,
                        )
                        last_mode_report = data.time
                simulate(model, data, addresses, command)
                viewer.sync()
                delay = model.opt.timestep - (time.monotonic() - step_start)
                if delay > 0.0:
                    time.sleep(delay)
        finally:
            if event_viewer is not None:
                event_viewer.close()
    print(f"Official GO2 viewer closed at t={data.time:.2f} s", flush=True)


def main() -> None:
    args = parse_args()
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    addresses = joint_addresses(model)
    event_camera_config = EventCameraConfig(
        width=args.event_width,
        height=args.event_height,
        fps=args.event_fps,
        threshold=args.event_threshold,
        accumulation_ms=args.event_accumulation_ms,
        refractory_ms=args.event_refractory_ms,
        style=args.event_style,
        display=args.event_camera and not args.headless,
    )

    if args.headless:
        run_headless(
            model,
            data,
            addresses,
            args.gait,
            args.duration,
            ObstacleAwareTrotController(args.gait) if args.auto_jump else None,
            event_camera_config if args.auto_jump else None,
        )
    else:
        run_viewer(
            model,
            data,
            addresses,
            args.gait,
            args.duration,
            args.event_camera,
            event_camera_config,
            args.auto_jump,
        )
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
