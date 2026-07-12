"""Render a MuJoCo camera stream as an event-camera style view."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


def _rgb(*values: int) -> np.ndarray:
    return np.array(values, dtype=np.uint8)


@dataclass(slots=True)
class EventCameraConfig:
    width: int = 640
    height: int = 360
    fps: float = 60.0
    threshold: float = 0.22
    accumulation_ms: float = 10.0
    refractory_ms: float = 1.0
    style: str = "evk4"
    camera_name: str = "event_camera"
    window_title: str = "GO2 Event Camera"
    display: bool = True


@dataclass(slots=True)
class EventFrameResult:
    image: np.ndarray
    step_detected: bool
    step_score: float


class EventCameraSensor:
    """Convert consecutive rendered frames into an event visualization."""

    _STYLES = {
        "evk4": {
            "background": _rgb(0, 0, 0),
            "positive": _rgb(255, 255, 255),
            "negative": _rgb(80, 170, 255),
        },
        "polarity": {
            "background": _rgb(0, 0, 0),
            "positive": _rgb(0, 0, 255),
            "negative": _rgb(255, 0, 0),
        },
    }

    def __init__(
        self,
        threshold: float,
        accumulation_ms: float,
        refractory_ms: float,
        style: str,
    ) -> None:
        if style not in self._STYLES:
            supported = ", ".join(sorted(self._STYLES))
            raise ValueError(f"Unsupported event style {style!r}. Choose from {supported}.")

        self._threshold = max(threshold, 1e-4)
        self._accumulation_s = max(accumulation_ms, 1.0) / 1000.0
        self._refractory_s = max(refractory_ms, 0.0) / 1000.0
        self._style = self._STYLES[style]
        self._previous_log_intensity: np.ndarray | None = None
        self._last_event_time: np.ndarray | None = None
        self._last_polarity: np.ndarray | None = None

    def process(self, frame: np.ndarray, frame_time: float) -> np.ndarray:
        grayscale = (
            0.299 * frame[..., 0] + 0.587 * frame[..., 1] + 0.114 * frame[..., 2]
        ).astype(np.float32)
        log_intensity = np.log1p(grayscale)

        if self._previous_log_intensity is None:
            self._previous_log_intensity = log_intensity
            self._last_event_time = np.full(frame.shape[:2], -np.inf, dtype=np.float32)
            self._last_polarity = np.zeros(frame.shape[:2], dtype=np.int8)
            return np.broadcast_to(
                self._style["background"], (*frame.shape[:2], 3)
            ).copy()

        if self._last_event_time is None or self._last_event_time.shape != frame.shape[:2]:
            self._last_event_time = np.full(frame.shape[:2], -np.inf, dtype=np.float32)
            self._last_polarity = np.zeros(frame.shape[:2], dtype=np.int8)

        delta = log_intensity - self._previous_log_intensity
        self._previous_log_intensity = log_intensity

        positive = delta >= self._threshold
        negative = delta <= -self._threshold

        if self._refractory_s > 0.0 and self._last_polarity is not None:
            recent = (frame_time - self._last_event_time) < self._refractory_s
            positive &= ~(recent & (self._last_polarity == 1))
            negative &= ~(recent & (self._last_polarity == -1))

        self._last_event_time[positive | negative] = frame_time
        self._last_polarity[positive] = 1
        self._last_polarity[negative] = -1

        active = (frame_time - self._last_event_time) <= self._accumulation_s
        event_frame = np.broadcast_to(
            self._style["background"], (*frame.shape[:2], 3)
        ).copy()

        positive_active = active & (self._last_polarity == 1)
        negative_active = active & (self._last_polarity == -1)
        event_frame[positive_active] = self._style["positive"]
        event_frame[negative_active] = self._style["negative"]
        return event_frame


class StepDetector:
    """Detect step-like horizontal event bands in the lower center view."""

    def __init__(self, min_consecutive_frames: int = 3) -> None:
        self._min_consecutive_frames = max(min_consecutive_frames, 1)
        self._consecutive = 0

    def detect(self, event_frame: np.ndarray) -> tuple[bool, float]:
        active = np.any(event_frame > 0, axis=2)
        height, width = active.shape
        y0 = int(height * 0.28)
        y1 = int(height * 0.92)
        x0 = int(width * 0.18)
        x1 = int(width * 0.82)
        roi = active[y0:y1, x0:x1]

        if roi.size == 0:
            self._consecutive = 0
            return False, 0.0

        row_ratio = roi.mean(axis=1)
        coverage = float(roi.mean())
        strong_rows = row_ratio > 0.18
        groups = 0
        run_length = 0
        for is_strong in strong_rows:
            if is_strong:
                run_length += 1
            elif run_length >= 2:
                groups += 1
                run_length = 0
            else:
                run_length = 0
        if run_length >= 2:
            groups += 1

        top_band = float(np.argmax(strong_rows) / max(len(strong_rows), 1)) if np.any(strong_rows) else 1.0
        score = min(
            1.0,
            coverage * 7.0 + groups * 0.22 + max(0.0, 0.75 - top_band) * 0.4,
        )
        detected_now = groups >= 2 and coverage > 0.03 and top_band < 0.78
        if detected_now:
            self._consecutive += 1
        else:
            self._consecutive = 0
        return self._consecutive >= self._min_consecutive_frames, score


def _resize_image_nearest(
    image: np.ndarray, target_width: int, target_height: int
) -> np.ndarray:
    if target_width <= 0 or target_height <= 0:
        return image

    source_height, source_width = image.shape[:2]
    if source_width == target_width and source_height == target_height:
        return image

    x_idx = np.linspace(0, source_width - 1, target_width).astype(np.int32)
    y_idx = np.linspace(0, source_height - 1, target_height).astype(np.int32)
    return image[y_idx][:, x_idx]


class _Cv2Window:
    def __init__(self, title: str) -> None:
        import cv2

        self._cv2 = cv2
        self._title = title
        self._closed = False
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title, 960, 540)

    @property
    def closed(self) -> bool:
        return self._closed

    def show(self, image: np.ndarray) -> None:
        if self._closed:
            return
        self._cv2.imshow(self._title, image[..., ::-1])
        self._cv2.waitKey(1)
        visible = self._cv2.getWindowProperty(self._title, self._cv2.WND_PROP_VISIBLE)
        self._closed = visible < 1

    def close(self) -> None:
        if self._closed:
            return
        self._cv2.destroyWindow(self._title)
        self._closed = True


class _TkWindow:
    def __init__(self, title: str) -> None:
        import tkinter as tk

        self._tk = tk
        self._root = tk.Tk()
        self._root.title(title)
        self._root.geometry("960x540")
        self._root.minsize(480, 270)
        self._root.protocol("WM_DELETE_WINDOW", self.close)
        self._canvas = tk.Canvas(
            self._root, bd=0, highlightthickness=0, background="black"
        )
        self._canvas.pack(fill="both", expand=True)
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def show(self, image: np.ndarray) -> None:
        if self._closed:
            return

        self._root.update_idletasks()
        canvas_width = max(self._canvas.winfo_width(), 1)
        canvas_height = max(self._canvas.winfo_height(), 1)
        height, width, _ = image.shape
        scale = min(canvas_width / width, canvas_height / height)
        scaled_width = max(int(width * scale), 1)
        scaled_height = max(int(height * scale), 1)
        scaled = _resize_image_nearest(image, scaled_width, scaled_height)
        ppm = (
            f"P6\n{scaled_width} {scaled_height}\n255\n".encode("ascii")
            + scaled.tobytes()
        )
        photo = self._tk.PhotoImage(data=ppm, format="PPM")
        self._canvas.delete("all")
        self._canvas.create_image(canvas_width // 2, canvas_height // 2, image=photo)
        self._canvas.image = photo
        try:
            self._root.update()
        except self._tk.TclError:
            self._closed = True

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._root.destroy()
        except self._tk.TclError:
            pass


class EventCameraViewer:
    """Render a MuJoCo camera and show the event image in a second window."""

    def __init__(self, model: mujoco.MjModel, config: EventCameraConfig) -> None:
        if (
            mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_CAMERA, config.camera_name
            )
            < 0
        ):
            raise ValueError(
                f"Camera {config.camera_name!r} was not found in the loaded model."
            )

        self._renderer = mujoco.Renderer(model, height=config.height, width=config.width)
        self._sensor = EventCameraSensor(
            threshold=config.threshold,
            accumulation_ms=config.accumulation_ms,
            refractory_ms=config.refractory_ms,
            style=config.style,
        )
        self._step_detector = StepDetector()
        self._camera_name = config.camera_name
        self._frame_interval = 1.0 / max(config.fps, 1.0)
        self._next_frame_time = 0.0
        self._window: _Cv2Window | _TkWindow | None = None
        self._backend = "offscreen"

        if config.display:
            try:
                self._window = _Cv2Window(config.window_title)
                self._backend = "cv2"
            except Exception:
                self._window = _TkWindow(config.window_title)
                self._backend = "tk"

    @property
    def backend(self) -> str:
        return self._backend

    def render_if_due(self, data: mujoco.MjData) -> EventFrameResult | None:
        if self._window is not None and self._window.closed:
            return None
        if data.time + 1e-9 < self._next_frame_time:
            return None

        self._renderer.update_scene(data, camera=self._camera_name)
        rgb = self._renderer.render()
        events = self._sensor.process(rgb, data.time)
        step_detected, step_score = self._step_detector.detect(events)
        if self._window is not None:
            self._window.show(events)
        self._next_frame_time = data.time + self._frame_interval
        return EventFrameResult(
            image=events, step_detected=step_detected, step_score=step_score
        )

    def close(self) -> None:
        if self._window is not None:
            self._window.close()
        self._renderer.close()
