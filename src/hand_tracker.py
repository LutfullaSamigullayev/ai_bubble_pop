"""MediaPipe 0.10.21 legacy Hands API wrapper for CPU-friendly tracking."""
from __future__ import annotations

import threading
import time
from typing import Optional

import cv2
import mediapipe as mp


CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
MAX_HANDS = 2
MODEL_COMPLEXITY = 0
MIN_DETECTION_CONFIDENCE = 0.45
MIN_TRACKING_CONFIDENCE = 0.45


class HandTrackingWorker:
    """Read webcam frames and run MediaPipe inference in a background thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="HandTrackingWorker",
            daemon=True,
        )
        self._frame = None
        self._hands: list[list[tuple[float, float]]] = []
        self._frame_id = 0
        self._ai_fps = 0.0
        self._error: Optional[str] = None

    def start(self) -> None:
        self._thread.start()
        if not self._ready_event.wait(timeout=15):
            raise RuntimeError(
                "Kamera va MediaPipe 15 soniya ichida ishga tushmadi."
            )
        if self._error:
            raise RuntimeError(self._error)

    def snapshot(self, last_frame_id: int = -1):
        """Return the latest frame only when it has changed."""
        with self._lock:
            frame = None
            if self._frame is not None and self._frame_id != last_frame_id:
                frame = self._frame.copy()
            return (
                frame,
                self._hands,
                self._frame_id,
                self._ai_fps,
                self._error,
            )

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=3.0)

    @staticmethod
    def _open_camera():
        # DirectShow is usually more reliable on Windows. Fall back to OpenCV's
        # default backend if DirectShow cannot open the camera.
        camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        if not camera.isOpened():
            camera.release()
            camera = cv2.VideoCapture(CAMERA_INDEX)
        return camera

    def _run(self) -> None:
        camera = None
        try:
            camera = self._open_camera()
            if not camera.isOpened():
                raise RuntimeError(
                    "Webkamera ochilmadi. Boshqa dastur kameradan foydalanmayotganini tekshiring."
                )

            camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # IMPORTANT: Use the exact legacy API tested successfully by the user
            # with mediapipe==0.10.21. Do not use mp.tasks in this project version.
            hands_api = mp.solutions.hands
            with hands_api.Hands(
                static_image_mode=False,
                max_num_hands=MAX_HANDS,
                model_complexity=MODEL_COMPLEXITY,
                min_detection_confidence=MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
            ) as detector:
                # Signal readiness only after camera and detector are initialized.
                self._ready_event.set()
                previous_frame_time = time.perf_counter()
                smoothed_fps = 0.0

                while not self._stop_event.is_set():
                    ok, frame = camera.read()
                    if not ok:
                        time.sleep(0.01)
                        continue

                    # Mirror the image exactly as in the successful hand_test_lite.py.
                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    rgb.flags.writeable = False
                    result = detector.process(rgb)

                    detected = result.multi_hand_landmarks or []
                    hands = [
                        [
                            (float(point.x), float(point.y))
                            for point in hand.landmark
                        ]
                        for hand in detected[:MAX_HANDS]
                    ]

                    now = time.perf_counter()
                    elapsed = max(now - previous_frame_time, 1e-6)
                    previous_frame_time = now
                    instant_fps = 1.0 / elapsed
                    smoothed_fps = (
                        instant_fps
                        if smoothed_fps == 0.0
                        else smoothed_fps * 0.80 + instant_fps * 0.20
                    )

                    with self._lock:
                        self._frame = frame
                        self._hands = hands
                        self._frame_id += 1
                        self._ai_fps = smoothed_fps

        except Exception as exc:
            with self._lock:
                self._error = f"Hand tracking xatosi: {exc}"
            self._ready_event.set()
        finally:
            if camera is not None:
                camera.release()
            self._ready_event.set()


class _HandTrack:
    def __init__(self, track_id: int, wrist: tuple[float, float]) -> None:
        self.track_id = track_id
        self.wrist = wrist
        self.is_pinching = False
        self.last_seen = time.perf_counter()


class PinchTracker:
    """Detect a new thumb-index pinch and report its midpoint in normalized XY."""

    PINCH_START_RATIO = 0.30
    PINCH_RELEASE_RATIO = 0.46
    HAND_MATCH_DISTANCE = 0.28
    HAND_TRACK_TIMEOUT = 0.80

    def __init__(self) -> None:
        self.tracks: list[_HandTrack] = []
        self.next_id = 1

    def update(self, hands: list[list[tuple[float, float]]]):
        now = time.perf_counter()
        used_track_ids: set[int] = set()
        pinch_events: list[tuple[float, float]] = []
        active_pinches: list[tuple[float, float]] = []

        for landmarks in hands:
            if len(landmarks) < 21:
                continue

            wrist = landmarks[0]
            best_track = None
            best_distance = self.HAND_MATCH_DISTANCE

            for track in self.tracks:
                if track.track_id in used_track_ids:
                    continue
                dx = wrist[0] - track.wrist[0]
                dy = wrist[1] - track.wrist[1]
                distance = (dx * dx + dy * dy) ** 0.5
                if distance < best_distance:
                    best_distance = distance
                    best_track = track

            if best_track is None:
                best_track = _HandTrack(self.next_id, wrist)
                self.next_id += 1
                self.tracks.append(best_track)

            used_track_ids.add(best_track.track_id)
            best_track.wrist = wrist
            best_track.last_seen = now

            thumb_tip = landmarks[4]
            index_tip = landmarks[8]
            index_mcp = landmarks[5]
            pinky_mcp = landmarks[17]

            fingertip_distance = (
                (thumb_tip[0] - index_tip[0]) ** 2
                + (thumb_tip[1] - index_tip[1]) ** 2
            ) ** 0.5
            palm_width = (
                (index_mcp[0] - pinky_mcp[0]) ** 2
                + (index_mcp[1] - pinky_mcp[1]) ** 2
            ) ** 0.5
            pinch_ratio = fingertip_distance / max(palm_width, 1e-5)

            pinch_point = (
                (thumb_tip[0] + index_tip[0]) / 2.0,
                (thumb_tip[1] + index_tip[1]) / 2.0,
            )

            # Hysteresis: separate close/release thresholds avoid flicker.
            if (
                not best_track.is_pinching
                and pinch_ratio < self.PINCH_START_RATIO
            ):
                best_track.is_pinching = True
                pinch_events.append(pinch_point)
            elif (
                best_track.is_pinching
                and pinch_ratio > self.PINCH_RELEASE_RATIO
            ):
                best_track.is_pinching = False

            if best_track.is_pinching:
                active_pinches.append(pinch_point)

        self.tracks = [
            track
            for track in self.tracks
            if now - track.last_seen <= self.HAND_TRACK_TIMEOUT
        ]
        return pinch_events, active_pinches
