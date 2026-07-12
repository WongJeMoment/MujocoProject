"""Send an event-camera stair distance estimate to run_unitree_policy.py."""

from __future__ import annotations

import argparse
import json
import socket
import time


class EventDistancePublisher:
    """Publisher that can be imported by an EVK4/Metavision detector process."""

    def __init__(self, host: str = "127.0.0.1", port: int = 17001) -> None:
        self._address = (host, port)
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def publish(self, step_distance: float, confidence: float) -> None:
        message = json.dumps(
            {
                "step_distance": float(step_distance),
                "confidence": float(confidence),
            },
            separators=(",", ":"),
        ).encode("utf-8")
        self._socket.sendto(message, self._address)

    def close(self) -> None:
        self._socket.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=17001)
    parser.add_argument("--distance", type=float, required=True)
    parser.add_argument("--confidence", type=float, default=1.0)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--hz", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    publisher = EventDistancePublisher(args.host, args.port)
    deadline = time.monotonic() + max(args.duration, 0.0)
    try:
        while time.monotonic() < deadline:
            publisher.publish(args.distance, args.confidence)
            time.sleep(1.0 / max(args.hz, 1.0))
    finally:
        publisher.close()


if __name__ == "__main__":
    main()
