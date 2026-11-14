"""Lightweight MediaPipe Hands diagnostic for CPU-only Windows PCs.

This test uses the legacy MediaPipe Hands API with model_complexity=0
(the lightweight model), up to two hands, and no Pygame or background threads.
"""
from __future__ import annotations

import time

import cv2
import mediapipe as mp

CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
MAX_HANDS = 2


def main() -> None:
    # Prefer the Windows DirectShow backend; fall back to the default backend.
    camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        raise RuntimeError(
            "Webkamera ochilmadi. CAMERA_INDEX = 0 qiymatini tekshiring."
        )

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    hands_api = mp.solutions.hands
    drawing = mp.solutions.drawing_utils
    drawing_styles = mp.solutions.drawing_styles

    # model_complexity=0 selects the lightweight model.
    # static_image_mode=False uses video tracking between detections.
    with hands_api.Hands(
        static_image_mode=False,
        max_num_hands=MAX_HANDS,
        model_complexity=0,
        min_detection_confidence=0.45,
        min_tracking_confidence=0.45,
    ) as hands:
        previous_time = time.perf_counter()
        fps_smooth = 0.0

        try:
            while True:
                ok, frame = camera.read()
                if not ok:
                    print("Kameradan tasvir olinmadi.")
                    break

                # Mirror the webcam view for a more natural interaction.
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                result = hands.process(rgb)
                rgb.flags.writeable = True

                detected = result.multi_hand_landmarks or []
                handedness = result.multi_handedness or []

                h, w = frame.shape[:2]
                for index, hand_landmarks in enumerate(detected):
                    drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        hands_api.HAND_CONNECTIONS,
                        drawing_styles.get_default_hand_landmarks_style(),
                        drawing_styles.get_default_hand_connections_style(),
                    )

                    wrist = hand_landmarks.landmark[0]
                    x = max(0, min(w - 1, int(wrist.x * w)))
                    y = max(25, min(h - 1, int(wrist.y * h)))

                    label = f"Hand {index + 1}"
                    if index < len(handedness) and handedness[index].classification:
                        category = handedness[index].classification[0]
                        label = f"{category.label} {category.score:.2f}"

                    cv2.putText(
                        frame, label, (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2, cv2.LINE_AA,
                    )

                now = time.perf_counter()
                instant_fps = 1.0 / max(now - previous_time, 1e-6)
                previous_time = now
                fps_smooth = instant_fps if fps_smooth == 0 else (
                    0.85 * fps_smooth + 0.15 * instant_fps
                )

                cv2.putText(
                    frame, f"HANDS: {len(detected)}/{MAX_HANDS}",
                    (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (0, 255, 0), 2, cv2.LINE_AA,
                )
                cv2.putText(
                    frame, f"FPS: {fps_smooth:.1f} | MODEL: LITE",
                    (16, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 255, 255), 2, cv2.LINE_AA,
                )
                cv2.putText(
                    frame, f"CAMERA: {w}x{h} | Press Q to quit",
                    (16, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA,
                )

                cv2.imshow("MediaPipe Hands - Lightweight CPU Test", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
        finally:
            camera.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
