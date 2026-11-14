"""AI Bubble Pop Game using MediaPipe 0.10.21 lightweight Hands API."""
from __future__ import annotations

import math
import random
from pathlib import Path

import cv2
import pygame

from src.hand_tracker import HandTrackingWorker, PinchTracker, MAX_HANDS


TARGET_GAME_FPS = 60
BALLOON_SPAWN_INTERVAL = 0.95
MAX_BALLOONS = 12

BALLOON_COLORS = [
    (245, 75, 100),
    (75, 165, 245),
    (180, 85, 225),
    (245, 155, 55),
    (65, 190, 145),
    (245, 105, 175),
    (80, 190, 220),
]


def blend_color(a, b, amount: float):
    return tuple(
        max(0, min(255, int(a[i] + (b[i] - a[i]) * amount)))
        for i in range(3)
    )


def make_balloon_sprite(radius: int, base_color: tuple[int, int, int]):
    """Pre-render a glossy balloon; no gradient work is done per game frame."""
    body_width = int(radius * 1.48)
    body_height = int(radius * 1.88)
    padding = 12
    sprite = pygame.Surface(
        (body_width + padding * 2, body_height + padding * 2 + 12),
        pygame.SRCALPHA,
    )
    body_rect = pygame.Rect(padding, padding, body_width, body_height)
    dark = tuple(int(channel * 0.48) for channel in base_color)
    light = tuple(min(255, int(channel * 1.25 + 42)) for channel in base_color)

    pygame.draw.ellipse(sprite, (*dark, 255), body_rect)
    for step in range(30):
        t = step / 29.0
        inset_x = int((1.0 - t) * body_width * 0.13)
        inset_y = int((1.0 - t) * body_height * 0.09)
        rect = body_rect.inflate(-2 * inset_x, -2 * inset_y)
        rect.x += int(body_width * 0.025)
        rect.y -= int(body_height * 0.015)
        color = blend_color(base_color, light, t * 0.72)
        pygame.draw.ellipse(sprite, (*color, 255), rect)

    lower_shade = pygame.Surface(
        (body_width, int(body_height * 0.55)), pygame.SRCALPHA
    )
    pygame.draw.ellipse(
        lower_shade,
        (*dark, 55),
        pygame.Rect(0, 0, body_width, int(body_height * 0.90)),
    )
    sprite.blit(lower_shade, (padding, padding + int(body_height * 0.48)))

    pygame.draw.ellipse(
        sprite,
        (255, 255, 255, 165),
        pygame.Rect(
            padding + int(body_width * 0.17),
            padding + int(body_height * 0.13),
            max(5, int(body_width * 0.18)),
            max(10, int(body_height * 0.31)),
        ),
    )
    pygame.draw.ellipse(
        sprite,
        (255, 255, 255, 100),
        pygame.Rect(
            padding + int(body_width * 0.38),
            padding + int(body_height * 0.10),
            max(4, int(body_width * 0.10)),
            max(5, int(body_height * 0.08)),
        ),
    )
    pygame.draw.ellipse(sprite, (*dark, 220), body_rect, width=2)

    neck_top = padding + body_height - 1
    center_x = padding + body_width // 2
    pygame.draw.polygon(
        sprite,
        (*base_color, 255),
        [
            (center_x - 6, neck_top - 1),
            (center_x + 6, neck_top - 1),
            (center_x + 4, neck_top + 8),
            (center_x - 4, neck_top + 8),
        ],
    )
    pygame.draw.polygon(
        sprite,
        (*dark, 255),
        [
            (center_x - 5, neck_top + 6),
            (center_x + 5, neck_top + 6),
            (center_x, neck_top + 14),
        ],
    )
    return sprite, body_width, body_height, center_x, neck_top + 12


class Balloon:
    def __init__(self, screen_width: int, screen_height: int, initial=False):
        self.radius = random.randint(32, 48)
        self.color = random.choice(BALLOON_COLORS)
        (
            self.sprite,
            self.body_width,
            self.body_height,
            self.string_x,
            self.string_start_y,
        ) = make_balloon_sprite(self.radius, self.color)
        margin = self.body_width // 2 + 8
        self.x = random.randint(margin, max(margin, screen_width - margin))
        self.y = (
            random.randint(60, max(100, screen_height // 3))
            if initial
            else -self.body_height
        )
        self.speed = random.uniform(95.0, 165.0)
        self.drift = random.uniform(-22.0, 22.0)
        self.string_length = random.randint(34, 58)
        self.phase = random.uniform(0.0, math.tau)

    def update(self, dt: float, screen_width: int):
        self.y += self.speed * dt
        self.x += self.drift * dt
        self.phase += dt * 2.0
        half_width = self.body_width / 2
        if self.x < half_width:
            self.x = half_width
            self.drift = abs(self.drift)
        elif self.x > screen_width - half_width:
            self.x = screen_width - half_width
            self.drift = -abs(self.drift)

    def contains(self, point_x: float, point_y: float) -> bool:
        # Only the inflated body is a hit target; knot and string are excluded.
        dx = (point_x - self.x) / max(self.body_width * 0.50, 1.0)
        dy = (point_y - self.y) / max(self.body_height * 0.50, 1.0)
        return dx * dx + dy * dy <= 1.0

    def draw(self, surface: pygame.Surface):
        left = int(self.x - self.sprite.get_width() / 2)
        top = int(self.y - self.body_height / 2 - 12)
        start_x = self.x
        start_y = top + self.string_start_y
        string_points = []
        for index in range(9):
            t = index / 8.0
            sway = math.sin(self.phase + t * 3.2) * (4.0 + 5.0 * t)
            string_points.append(
                (int(start_x + sway), int(start_y + t * self.string_length))
            )
        pygame.draw.lines(surface, (235, 235, 245), False, string_points, 2)
        pygame.draw.lines(surface, (90, 100, 120), False, string_points, 1)
        surface.blit(self.sprite, (left, top))


class PopParticle:
    def __init__(self, x: float, y: float, color: tuple[int, int, int]):
        angle = random.uniform(0.0, math.tau)
        speed = random.uniform(65.0, 190.0)
        self.x = x
        self.y = y
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.color = color
        self.life = random.uniform(0.28, 0.48)
        self.max_life = self.life
        self.radius = random.randint(2, 5)

    def update(self, dt: float):
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.vy += 100.0 * dt
        self.life -= dt

    def draw(self, surface: pygame.Surface):
        ratio = max(0.0, self.life / self.max_life)
        pygame.draw.circle(
            surface,
            self.color,
            (int(self.x), int(self.y)),
            max(1, int(self.radius * ratio)),
        )


def build_fullscreen_camera(frame, screen_size):
    """Resize/crop only when a fresh camera frame arrives."""
    screen_width, screen_height = screen_size
    frame_height, frame_width = frame.shape[:2]
    scale = max(screen_width / frame_width, screen_height / frame_height)
    scaled_width = max(screen_width, int(frame_width * scale))
    scaled_height = max(screen_height, int(frame_height * scale))
    crop_x = (scaled_width - screen_width) // 2
    crop_y = (scaled_height - screen_height) // 2

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    camera_surface = pygame.image.frombuffer(
        rgb.tobytes(), (frame_width, frame_height), "RGB"
    )
    scaled_surface = pygame.transform.scale(
        camera_surface, (scaled_width, scaled_height)
    )
    background = scaled_surface.subsurface(
        pygame.Rect(crop_x, crop_y, screen_width, screen_height)
    ).copy()
    return background, (scaled_width, scaled_height, crop_x, crop_y)


def normalized_to_screen(point, transform, screen_size):
    scaled_width, scaled_height, crop_x, crop_y = transform
    screen_width, screen_height = screen_size
    x = int(point[0] * scaled_width - crop_x)
    y = int(point[1] * scaled_height - crop_y)
    if 0 <= x < screen_width and 0 <= y < screen_height:
        return x, y
    return None


def main() -> None:
    pygame.init()
    pygame.display.set_caption("AI Bubble Pop Game")
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    screen_width, screen_height = screen.get_size()
    clock = pygame.time.Clock()

    score_font = pygame.font.SysFont("arial", 38, bold=True)
    info_font = pygame.font.SysFont("arial", 23, bold=True)
    help_font = pygame.font.SysFont("arial", 22)

    tracker_worker = HandTrackingWorker()
    try:
        tracker_worker.start()
    except Exception:
        tracker_worker.stop()
        pygame.quit()
        raise

    balloons = [
        Balloon(screen_width, screen_height, initial=True) for _ in range(5)
    ]
    particles: list[PopParticle] = []
    pinch_tracker = PinchTracker()
    score = 0
    spawn_timer = 0.0
    last_frame_id = -1
    camera_background = None
    camera_transform = None
    latest_hands: list[list[tuple[float, float]]] = []
    ai_fps = 0.0
    running = True

    shade_layer = pygame.Surface((screen_width, screen_height), pygame.SRCALPHA)
    shade_layer.fill((0, 0, 0, 42))
    hud_width = min(500, screen_width - 40)
    hud_layer = pygame.Surface((hud_width, 118), pygame.SRCALPHA)
    pygame.draw.rect(
        hud_layer,
        (10, 18, 34, 190),
        hud_layer.get_rect(),
        border_radius=20,
    )

    try:
        while running:
            dt = min(clock.tick(TARGET_GAME_FPS) / 1000.0, 0.05)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
            if not running:
                break

            (
                frame,
                hands,
                frame_id,
                current_ai_fps,
                worker_error,
            ) = tracker_worker.snapshot(last_frame_id)
            ai_fps = current_ai_fps
            latest_hands = hands
            if worker_error:
                raise RuntimeError(worker_error)

            if frame is not None and frame_id != last_frame_id:
                camera_background, camera_transform = build_fullscreen_camera(
                    frame, (screen_width, screen_height)
                )
                last_frame_id = frame_id

            if camera_background is not None:
                screen.blit(camera_background, (0, 0))
            else:
                screen.fill((18, 25, 45))
            screen.blit(shade_layer, (0, 0))

            pinch_events_norm, active_pinches_norm = pinch_tracker.update(
                latest_hands
            )
            pinch_events = []
            active_pinches = []
            if camera_transform is not None:
                for point in pinch_events_norm:
                    mapped = normalized_to_screen(
                        point, camera_transform, (screen_width, screen_height)
                    )
                    if mapped is not None:
                        pinch_events.append(mapped)
                for point in active_pinches_norm:
                    mapped = normalized_to_screen(
                        point, camera_transform, (screen_width, screen_height)
                    )
                    if mapped is not None:
                        active_pinches.append(mapped)

            spawn_timer += dt
            while spawn_timer >= BALLOON_SPAWN_INTERVAL:
                spawn_timer -= BALLOON_SPAWN_INTERVAL
                if len(balloons) < MAX_BALLOONS:
                    balloons.append(Balloon(screen_width, screen_height))

            for balloon in balloons:
                balloon.update(dt, screen_width)

            remaining_balloons = []
            for balloon in balloons:
                hit_point = None
                for pinch_x, pinch_y in pinch_events:
                    if balloon.contains(pinch_x, pinch_y):
                        hit_point = (pinch_x, pinch_y)
                        break
                if hit_point is not None:
                    score += 10
                    particles.extend(
                        PopParticle(hit_point[0], hit_point[1], balloon.color)
                        for _ in range(15)
                    )
                elif balloon.y - balloon.body_height / 2 < screen_height:
                    remaining_balloons.append(balloon)
            balloons = remaining_balloons

            for balloon in balloons:
                balloon.draw(screen)
            alive_particles = []
            for particle in particles:
                particle.update(dt)
                if particle.life > 0:
                    particle.draw(screen)
                    alive_particles.append(particle)
            particles = alive_particles

            # Visual cursor: the midpoint between thumb and index finger.
            for pinch_x, pinch_y in active_pinches:
                pygame.draw.circle(
                    screen, (255, 255, 255), (pinch_x, pinch_y), 17, 3
                )
                pygame.draw.circle(
                    screen, (255, 220, 80), (pinch_x, pinch_y), 6
                )

            screen.blit(hud_layer, (22, 22))
            score_surface = score_font.render(
                f"BALL: {score}", True, (255, 255, 255)
            )
            status_surface = info_font.render(
                f"QO'LLAR: {len(latest_hands)}/{MAX_HANDS}   "
                f"O'YIN FPS: {clock.get_fps():.0f}   AI FPS: {ai_fps:.0f}",
                True,
                (195, 235, 255),
            )
            screen.blit(score_surface, (43, 30))
            screen.blit(status_surface, (46, 82))

            help_surface = help_font.render(
                "Bosh va kursatkich barmog'ingizni qisib sharni yoriting  |  ESC - chiqish",
                True,
                (255, 255, 255),
            )
            help_rect = help_surface.get_rect(
                center=(screen_width // 2, screen_height - 36)
            )
            screen.blit(help_surface, help_rect)
            pygame.display.flip()

    finally:
        tracker_worker.stop()
        pygame.quit()


if __name__ == "__main__":
    main()
