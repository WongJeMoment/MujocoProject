"""Deploy a trained GO2 PPO policy through Unitree SDK2 low-level DDS.

The program is dry-run by default. Motor commands are published only when both
--enable-motors and --confirm REAL_GO2 are supplied.
"""

from __future__ import annotations

import argparse
import json
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from policy_contract import (
    HOME,
    JOINT_LIMIT_HIGH,
    JOINT_LIMIT_LOW,
    KD,
    KP,
    build_observation,
    policy_action_to_target,
    projected_gravity,
)


@dataclass(slots=True)
class RobotSample:
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    quaternion: np.ndarray
    gyroscope: np.ndarray
    body_velocity: np.ndarray
    base_height: float
    state_age: float
    estimator_age: float


class EventDistanceReceiver:
    """Receive {"step_distance": metres, "confidence": 0..1} over UDP."""

    def __init__(self, host: str, port: int, stale_after: float = 0.25) -> None:
        self._socket: socket.socket | None = None
        self._distance = 2.0
        self._confidence = 0.0
        self._updated_at = -np.inf
        self._stale_after = stale_after
        if port > 0:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._socket.bind((host, port))
            self._socket.setblocking(False)

    def read(self) -> tuple[float, float]:
        if self._socket is not None:
            while True:
                try:
                    payload, _ = self._socket.recvfrom(2048)
                except BlockingIOError:
                    break
                try:
                    message = json.loads(payload.decode("utf-8"))
                    distance = float(message["step_distance"])
                    confidence = float(message.get("confidence", 1.0))
                except (KeyError, TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if np.isfinite(distance) and np.isfinite(confidence):
                    self._distance = float(np.clip(distance, 0.0, 2.0))
                    self._confidence = float(np.clip(confidence, 0.0, 1.0))
                    self._updated_at = time.monotonic()

        if time.monotonic() - self._updated_at > self._stale_after:
            return 2.0, 0.0
        if self._confidence < 0.5:
            return 2.0, self._confidence
        return self._distance, self._confidence

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()


class UnitreeGo2Bridge:
    """Small adapter around unitree_sdk2py DDS messages."""

    def __init__(self, network_interface: str, publish_commands: bool) -> None:
        try:
            from unitree_sdk2py.core.channel import (
                ChannelFactoryInitialize,
                ChannelPublisher,
                ChannelSubscriber,
            )
            from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import (
                LowCmd_,
                LowState_,
                SportModeState_,
            )
            from unitree_sdk2py.utils.crc import CRC
        except ImportError as exc:
            raise RuntimeError(
                "unitree_sdk2py is not installed. Install Unitree SDK2 Python "
                "before connecting to the real GO2."
            ) from exc

        ChannelFactoryInitialize(0, network_interface)
        self._lock = threading.Lock()
        self._low_state = None
        self._sport_state = None
        self._low_state_time = -np.inf
        self._sport_state_time = -np.inf
        self._publish_commands = publish_commands

        self._low_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self._low_subscriber.Init(self._on_low_state, 10)
        self._sport_subscriber = ChannelSubscriber("rt/sportmodestate", SportModeState_)
        self._sport_subscriber.Init(self._on_sport_state, 10)

        self._publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self._publisher.Init()
        self._command = unitree_go_msg_dds__LowCmd_()
        self._crc = CRC()
        self._command.head[0] = 0xFE
        self._command.head[1] = 0xEF
        self._command.level_flag = 0xFF
        self._command.gpio = 0

    def _on_low_state(self, message) -> None:
        with self._lock:
            self._low_state = message
            self._low_state_time = time.monotonic()

    def _on_sport_state(self, message) -> None:
        with self._lock:
            self._sport_state = message
            self._sport_state_time = time.monotonic()

    def wait_for_state(self, timeout: float = 5.0) -> RobotSample:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            sample = self.sample()
            if sample is not None:
                return sample
            time.sleep(0.01)
        raise RuntimeError("No rt/lowstate data received from the GO2 within 5 seconds.")

    def sample(self) -> RobotSample | None:
        with self._lock:
            low_state = self._low_state
            sport_state = self._sport_state
            low_time = self._low_state_time
            sport_time = self._sport_state_time
            if low_state is None:
                return None

            joint_position = np.array(
                [low_state.motor_state[index].q for index in range(12)],
                dtype=np.float32,
            )
            joint_velocity = np.array(
                [low_state.motor_state[index].dq for index in range(12)],
                dtype=np.float32,
            )
            quaternion = np.asarray(low_state.imu_state.quaternion, dtype=np.float32).copy()
            gyroscope = np.asarray(low_state.imu_state.gyroscope, dtype=np.float32).copy()
            if sport_state is None:
                body_velocity = np.zeros(3, dtype=np.float32)
                base_height = 0.42
            else:
                body_velocity = np.asarray(sport_state.velocity, dtype=np.float32).copy()
                base_height = float(sport_state.body_height)

        now = time.monotonic()
        return RobotSample(
            joint_position=joint_position,
            joint_velocity=joint_velocity,
            quaternion=quaternion,
            gyroscope=gyroscope,
            body_velocity=body_velocity,
            base_height=base_height,
            state_age=now - low_time,
            estimator_age=now - sport_time,
        )

    def send_position_target(
        self, target: np.ndarray, kp: np.ndarray, kd: np.ndarray
    ) -> None:
        if not self._publish_commands:
            return
        for index in range(12):
            motor = self._command.motor_cmd[index]
            motor.mode = 0x01
            motor.q = float(target[index])
            motor.dq = 0.0
            motor.kp = float(kp[index])
            motor.kd = float(kd[index])
            motor.tau = 0.0
        self._command.crc = self._crc.Crc(self._command)
        self._publisher.Write(self._command)

    def damping_stop(self, joint_position: np.ndarray, duration: float = 0.3) -> None:
        if not self._publish_commands:
            return
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            self.send_position_target(
                joint_position,
                np.zeros(12, dtype=np.float32),
                np.full(12, 2.0, dtype=np.float32),
            )
            time.sleep(0.002)


def mock_sample(joint_position: np.ndarray, joint_velocity: np.ndarray) -> RobotSample:
    return RobotSample(
        joint_position=joint_position,
        joint_velocity=joint_velocity,
        quaternion=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        gyroscope=np.zeros(3, dtype=np.float32),
        body_velocity=np.array([0.2, 0.0, 0.0], dtype=np.float32),
        base_height=0.42,
        state_age=0.0,
        estimator_age=0.0,
    )


def validate_sample(sample: RobotSample, require_estimator: bool) -> None:
    arrays = (
        sample.joint_position,
        sample.joint_velocity,
        sample.quaternion,
        sample.gyroscope,
        sample.body_velocity,
    )
    if not all(np.all(np.isfinite(value)) for value in arrays):
        raise RuntimeError("Robot state contains NaN or infinity.")
    if sample.state_age > 0.10:
        raise RuntimeError(f"Low-level state timeout: {sample.state_age:.3f} s.")
    if require_estimator and sample.estimator_age > 0.20:
        raise RuntimeError("Sport state estimator is missing or stale.")
    if np.max(np.abs(sample.joint_velocity)) > 35.0:
        raise RuntimeError("Joint velocity safety limit exceeded.")
    gravity = projected_gravity(sample.quaternion)
    if gravity[2] > -0.55:
        raise RuntimeError("Robot tilt safety limit exceeded.")
    if np.any(sample.joint_position < JOINT_LIMIT_LOW - 0.15) or np.any(
        sample.joint_position > JOINT_LIMIT_HIGH + 0.15
    ):
        raise RuntimeError("Measured joint position is outside the GO2 safety limits.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).parent / "checkpoints" / "go2_stair_ppo.zip",
    )
    parser.add_argument("--network-interface", default="eth0")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--policy-hz", type=float, default=100.0)
    parser.add_argument("--command-hz", type=float, default=500.0)
    parser.add_argument("--startup-seconds", type=float, default=4.0)
    parser.add_argument("--gain-scale", type=float, default=0.35)
    parser.add_argument("--action-scale", type=float, default=0.50)
    parser.add_argument("--event-bind", default="127.0.0.1")
    parser.add_argument("--event-port", type=int, default=0)
    parser.add_argument("--allow-missing-estimator", action="store_true")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--enable-motors", action="store_true")
    parser.add_argument("--confirm", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.enable_motors and args.confirm != "REAL_GO2":
        raise SystemExit("Motor control requires: --enable-motors --confirm REAL_GO2")
    if args.mock and args.enable_motors:
        raise SystemExit("--mock and --enable-motors cannot be used together.")
    if args.policy_hz <= 0.0 or args.command_hz < args.policy_hz:
        raise SystemExit("command-hz must be greater than or equal to policy-hz.")

    policy = PPO.load(str(args.model), device=args.device)
    event_receiver = EventDistanceReceiver(args.event_bind, args.event_port)
    bridge = None if args.mock else UnitreeGo2Bridge(
        args.network_interface, publish_commands=args.enable_motors
    )

    if bridge is None:
        joint_position = HOME.copy()
        joint_velocity = np.zeros(12, dtype=np.float32)
        first_sample = mock_sample(joint_position, joint_velocity)
    else:
        first_sample = bridge.wait_for_state()
        joint_position = first_sample.joint_position.copy()
        joint_velocity = first_sample.joint_velocity.copy()

    mode = "MOTOR ENABLED" if args.enable_motors else "DRY RUN"
    print(f"Starting Unitree policy runner in {mode} mode.")
    start = time.monotonic()
    next_command = start
    next_policy = start
    last_report = start - 1.0
    initial_position = first_sample.joint_position.copy()
    target = initial_position.copy()
    step_distance = 2.0
    confidence = 0.0
    require_estimator = args.enable_motors and not args.allow_missing_estimator

    try:
        while args.duration <= 0.0 or time.monotonic() - start < args.duration:
            now = time.monotonic()
            if now < next_command:
                time.sleep(min(next_command - now, 0.001))
                continue
            next_command += 1.0 / args.command_hz

            if bridge is None:
                sample = mock_sample(joint_position, joint_velocity)
            else:
                sample = bridge.sample()
                if sample is None:
                    raise RuntimeError("No low-level robot state is available.")
            validate_sample(sample, require_estimator=require_estimator)

            elapsed = now - start
            if elapsed < args.startup_seconds:
                alpha = np.clip(elapsed / max(args.startup_seconds, 1e-3), 0.0, 1.0)
                alpha = alpha * alpha * (3.0 - 2.0 * alpha)
                target = (1.0 - alpha) * initial_position + alpha * HOME
                gain_ramp = max(float(alpha), 0.05)
            else:
                policy_elapsed = elapsed - args.startup_seconds
                gain_ramp = 1.0
                if now >= next_policy:
                    next_policy = now + 1.0 / args.policy_hz
                    step_distance, confidence = event_receiver.read()
                    observation = build_observation(
                        joint_position=sample.joint_position,
                        joint_velocity=sample.joint_velocity,
                        body_linear_velocity=sample.body_velocity,
                        body_angular_velocity=sample.gyroscope,
                        gravity_body=projected_gravity(sample.quaternion),
                        base_height=sample.base_height,
                        step_distance=step_distance,
                    )
                    action, _ = policy.predict(observation, deterministic=True)
                    policy_target = policy_action_to_target(
                        action,
                        elapsed_time=policy_elapsed,
                        step_distance=step_distance,
                        action_scale=args.action_scale,
                    )
                    motion_blend = min(policy_elapsed / 1.0, 1.0)
                    target = (1.0 - motion_blend) * HOME + motion_blend * policy_target

            if np.max(np.abs(target - sample.joint_position)) > 1.20:
                raise RuntimeError("Policy target jump exceeded 1.20 rad.")

            kp = KP * float(np.clip(args.gain_scale, 0.05, 1.0)) * gain_ramp
            kd = KD * float(np.clip(args.gain_scale, 0.05, 1.0))
            if bridge is not None:
                bridge.send_position_target(target, kp, kd)
            else:
                old_position = joint_position.copy()
                joint_position += 0.04 * (target - joint_position)
                joint_velocity = (joint_position - old_position) * args.command_hz

            if now - last_report >= 1.0:
                last_report = now
                print(
                    f"t={elapsed:6.1f}s step={step_distance:.2f}m "
                    f"confidence={confidence:.2f} max_target={np.max(np.abs(target)):.2f}rad"
                )
    except KeyboardInterrupt:
        print("Stopping on keyboard interrupt.")
    finally:
        event_receiver.close()
        if bridge is not None:
            latest = bridge.sample()
            stop_position = (
                latest.joint_position if latest is not None else first_sample.joint_position
            )
            bridge.damping_stop(stop_position)


if __name__ == "__main__":
    main()
