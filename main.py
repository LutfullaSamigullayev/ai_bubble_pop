import math
import random
import time
from pathlib import Path

import cv2
import mediapipe as mp
import pygame


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
MODEL_PATH = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"
MAX_HANDS = 6
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
TARGET_FPS = 60
BUBBLE_SPAWN_INTERVAL = 0.75
MAX_BUBBLES = 16
FINGERTIPS = [4, 8, 12, 16, 20]

BUBBLE_COLORS = [
    (75, 190, 255),
    (255, 105, 170),
    (120, 225, 155),
    (255, 190, 75),
    (185, 135, 255),
    (80, 225, 220),
]


class Bubble:
    """A bubble that falls from the top of the screen."""

    def __init__(self, screen_width: int, screen_height: int, initial: bool = False):
        self.radius = random.randint(27, 48)
        self.x = random.randint(self.radius, max(self.radius, screen_width - self.radius))
        self.y = (
            random.randint(50, max(80, screen_height // 3))
            if initial
            else -self.radius
        )
        self.speed = random.uniform(115.0, 205.0)
        self.drift = random.uniform(-24.0, 24.0)
        self.color = random.choice(BUBBLE_COLORS)

    def update(self, dt: float, screen_width: int) -> None:
        self.y += self.speed * dt
        self.x += self.drift * dt

        if self.x < self.radius:
            self.x = self.radius
            self.drift = abs(self.drift)
        elif self.x > screen_width - self.radius:
            self.x = screen_width - self.radius
            self.drift = -abs(self.drift)

    def draw(self, surface: pygame.Surface) -> None:
        center = (int(self.x), int(self.y))
        pygame.draw.circle(surface, (*self.color, 105), center, self.radius)
        pygame.draw.circle(surface, (*self.color, 245), center, self.radius, 3)

        # Small white shine makes the bubble look glossy.
        shine_center = (
            int(self.x - self.radius * 0.32),
            int(self.y - self.radius * 0.32),
        )
        pygame.draw.circle(
            surface,
            (255, 255, 255, 225),
            shine_center,
            max(3, self.radius // 7),
        )
        pygame.draw.arc(
            surface,
            (255, 255, 255, 150),
            pygame.Rect(
                int(self.x - self.radius * 0.65),
                int(self.y - self.radius * 0.65),
                int(self.radius * 1.3),
                int(self.radius * 1.3),
            ),
            math.radians(205),
            math.radians(315),
            2,
        )


class PopParticle:
    """A small particle used for the bubble-pop effect."""

    def __init__(self, x: float, y: float, color: tuple[int, int, int]):
        angle = random.uniform(0.0, math.tau)
        speed = random.uniform(55.0, 175.0)
        self.x = x
        self.y = y
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.color = color
        self.life = 0.48
        self.max_life = self.life

    def update(self, dt: float) -> None:
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.vy += 90.0 * dt
        self.life -= dt

    def draw(self, surface: pygame.Surface) -> None:
        ratio = max(0.0, self.life / self.max_life)
        alpha = int(255 * ratio)
        radius = max(1, int(5 * ratio))
        pygame.draw.circle(
            surface,
            (*self.color, alpha),
            (int(self.x), int(self.y)),
            radius,
        )


def prepare_fullscreen_camera_frame(frame, screen_size):
    """Scale camera frame to fill the display and return crop offsets."""
    screen_width, screen_height = screen_size
    frame_height, frame_width = frame.shape[:2]

    # Fill the whole screen while preserving the camera image proportions.
    scale = max(screen_width / frame_width, screen_height / frame_height)
    scaled_width = math.ceil(frame_width * scale)
    scaled_height = math.ceil(frame_height * scale)

    resized = cv2.resize(
        frame,
        (scaled_width, scaled_height),
        interpolation=cv2.INTER_LINEAR,
    )

    crop_x = max(0, (scaled_width - screen_width) // 2)
    crop_y = max(0, (scaled_height - screen_height) // 2)
    cropped = resized[
        crop_y : crop_y + screen_height,
        crop_x : crop_x + screen_width,
    ]

    return cropped, scaled_width, scaled_height, crop_x, crop_y


def create_pop_particles(particles, x, y, color):
    for _ in range(14):
        particles.append(PopParticle(x, y, color))


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"MediaPipe modeli topilmadi: {MODEL_PATH}\n"
            "Modelni models/hand_landmarker.task manziliga joylashtiring."
        )

    pygame.init()
    pygame.display.set_caption("AI Bubble Pop Game")

    # Open the game directly in fullscreen mode.
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    screen_width, screen_height = screen.get_size()

    clock = pygame.time.Clock()
    score_font = pygame.font.SysFont("arial", 38, bold=True)
    info_font = pygame.font.SysFont("arial", 23, bold=True)
    help_font = pygame.font.SysFont("arial", 22)

    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    if not camera.isOpened():
        camera.release()
        pygame.quit()
        raise RuntimeError("Webkamera ochilmadi. Kamerani tekshiring.")

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=RunningMode.VIDEO,
        num_hands=MAX_HANDS,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    bubbles = [Bubble(screen_width, screen_height, initial=True) for _ in range(4)]
    particles = []
    score = 0
    spawn_timer = 0.0
    last_timestamp_ms = -1
    start_time = time.perf_counter()
    running = True

    # Reusable transparent layers.
    effects_layer = pygame.Surface((screen_width, screen_height), pygame.SRCALPHA)
    shade_layer = pygame.Surface((screen_width, screen_height), pygame.SRCALPHA)
    shade_layer.fill((0, 0, 0, 48))
    hud_layer = pygame.Surface((390, 112), pygame.SRCALPHA)
    pygame.draw.rect(
        hud_layer,
        (12, 20, 38, 185),
        hud_layer.get_rect(),
        border_radius=20,
    )

    try:
        with HandLandmarker.create_from_options(options) as landmarker:
            while running:
                dt = min(clock.tick(TARGET_FPS) / 1000.0, 0.05)

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key in (pygame.K_ESCAPE, pygame.K_q):
                            running = False

                if not running:
                    break

                success, frame = camera.read()
                if not success:
                    print("Kameradan tasvir olinmadi.")
                    break

                # Mirror the camera so movement feels natural to players.
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame,
                )

                # MediaPipe VIDEO mode requires strictly increasing timestamps.
                timestamp_ms = max(
                    last_timestamp_ms + 1,
                    int((time.perf_counter() - start_time) * 1000),
                )
                last_timestamp_ms = timestamp_ms
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                # Prepare a full-screen camera background and matching coordinates.
                camera_frame, scaled_w, scaled_h, crop_x, crop_y = (
                    prepare_fullscreen_camera_frame(frame, (screen_width, screen_height))
                )
                camera_rgb = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
                camera_surface = pygame.image.frombuffer(
                    camera_rgb.tobytes(),
                    (screen_width, screen_height),
                    "RGB",
                ).copy()
                screen.blit(camera_surface, (0, 0))
                screen.blit(shade_layer, (0, 0))

                # Convert all fingertips from normalized camera coordinates
                # into the same cropped fullscreen coordinate system.
                fingertip_points = []
                detected_hands = result.hand_landmarks

                for hand_landmarks in detected_hands:
                    for landmark_index in FINGERTIPS:
                        landmark = hand_landmarks[landmark_index]
                        point_x = int(landmark.x * scaled_w - crop_x)
                        point_y = int(landmark.y * scaled_h - crop_y)

                        if 0 <= point_x < screen_width and 0 <= point_y < screen_height:
                            fingertip_points.append((point_x, point_y))

                # Spawn bubbles continuously.
                spawn_timer += dt
                while spawn_timer >= BUBBLE_SPAWN_INTERVAL:
                    spawn_timer -= BUBBLE_SPAWN_INTERVAL
                    if len(bubbles) < MAX_BUBBLES:
                        bubbles.append(Bubble(screen_width, screen_height))

                # Move bubbles and check collisions against every detected fingertip.
                remaining_bubbles = []
                for bubble in bubbles:
                    bubble.update(dt, screen_width)
                    popped = False
                    hit_position = (bubble.x, bubble.y)
                    hit_distance = (bubble.radius + 10) ** 2

                    for finger_x, finger_y in fingertip_points:
                        distance_squared = (
                            (finger_x - bubble.x) ** 2
                            + (finger_y - bubble.y) ** 2
                        )
                        if distance_squared <= hit_distance:
                            popped = True
                            hit_position = (finger_x, finger_y)
                            break

                    if popped:
                        score += 10
                        create_pop_particles(
                            particles,
                            hit_position[0],
                            hit_position[1],
                            bubble.color,
                        )
                    elif bubble.y - bubble.radius <= screen_height:
                        remaining_bubbles.append(bubble)

                bubbles = remaining_bubbles

                # Draw bubbles and pop particles on a transparent layer.
                effects_layer.fill((0, 0, 0, 0))
                for bubble in bubbles:
                    bubble.draw(effects_layer)

                active_particles = []
                for particle in particles:
                    particle.update(dt)
                    if particle.life > 0:
                        particle.draw(effects_layer)
                        active_particles.append(particle)
                particles = active_particles
                screen.blit(effects_layer, (0, 0))

                # Draw fingertip markers above the bubbles.
                for finger_x, finger_y in fingertip_points:
                    pygame.draw.circle(screen, (255, 255, 255), (finger_x, finger_y), 12, 2)
                    pygame.draw.circle(screen, (70, 225, 255), (finger_x, finger_y), 5)

                # Score and status panel.
                screen.blit(hud_layer, (22, 22))
                score_surface = score_font.render(f"BALL: {score}", True, (255, 255, 255))
                status_surface = info_font.render(
                    f"QO'LLAR: {len(detected_hands)}   FPS: {clock.get_fps():.0f}",
                    True,
                    (190, 235, 255),
                )
                screen.blit(score_surface, (43, 30))
                screen.blit(status_surface, (46, 77))

                help_surface = help_font.render(
                    "Barmoqlaringiz bilan pufaklarni yoriting  |  ESC - chiqish",
                    True,
                    (255, 255, 255),
                )
                help_rect = help_surface.get_rect(
                    center=(screen_width // 2, screen_height - 36)
                )
                screen.blit(help_surface, help_rect)

                pygame.display.flip()

    finally:
        camera.release()
        pygame.quit()


if __name__ == "__main__":
    main()
