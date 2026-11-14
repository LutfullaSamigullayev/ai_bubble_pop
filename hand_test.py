"""Standalone MediaPipe Hand Landmarker diagnostic (no Pygame, no threads)."""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import mediapipe as mp

MODEL_PATH = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"
CAMERA_INDEX = 0
MAX_HANDS = 2

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]
FINGERTIPS = {4, 8, 12, 16, 20}


def main() -> None:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model topilmadi: {MODEL_PATH}\n"
            "hand_landmarker.task faylini models/ papkasiga joylashtiring."
        )

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    Options = mp.tasks.vision.HandLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = Options(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=RunningMode.VIDEO,
        num_hands=MAX_HANDS,
        min_hand_detection_confidence=0.40,
        min_hand_presence_confidence=0.40,
        min_tracking_confidence=0.40,
    )

    camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        raise RuntimeError("Webkamera ochilmadi. CAMERA_INDEX qiymatini tekshiring.")

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    started = time.perf_counter()
    prev = started
    fps_ema = 0.0
    last_timestamp_ms = -1

    try:
        with HandLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, frame = camera.read()
                if not ok:
                    print("Kameradan frame olinmadi.")
                    break

                frame = cv2.flip(frame, 1)
                height, width = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                timestamp_ms = max(
                    last_timestamp_ms + 1,
                    int((time.perf_counter() - started) * 1000),
                )
                last_timestamp_ms = timestamp_ms
                result = landmarker.detect_for_video(mp_image, timestamp_ms)
                hands = result.hand_landmarks or []

                for hand_index, hand in enumerate(hands):
                    pts = [
                        (max(0, min(width - 1, int(p.x * width))),
                         max(0, min(height - 1, int(p.y * height))))
                        for p in hand
                    ]
                    for a, b in CONNECTIONS:
                        cv2.line(frame, pts[a], pts[b], (60, 220, 90), 2)
                    for index, point in enumerate(pts):
                        color = (0, 80, 255) if index in FINGERTIPS else (255, 220, 0)
                        cv2.circle(frame, point, 5 if index in FINGERTIPS else 3, color, -1)

                    wrist_x, wrist_y = pts[0]
                    cv2.putText(
                        frame, f"Hand {hand_index + 1}",
                        (wrist_x + 8, max(25, wrist_y - 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2,
                    )

                now = time.perf_counter()
                instant_fps = 1.0 / max(now - prev, 1e-6)
                prev = now
                fps_ema = instant_fps if fps_ema == 0 else fps_ema * 0.8 + instant_fps * 0.2

                cv2.putText(frame, f"HANDS: {len(hands)}/{MAX_HANDS}", (18, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"AI FPS: {fps_ema:.1f}", (18, 66),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"Camera: {width}x{height}", (18, 98),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, "Show one/two open hands. Press Q to quit.",
                            (18, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (255, 255, 255), 1)

                cv2.imshow("MediaPipe Hand Detection Diagnostic", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
