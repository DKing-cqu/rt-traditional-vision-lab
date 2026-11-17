from dataclasses import dataclass

import cv2
import numpy as np


BASELINE_FILL = 0.58
RELAXED_FILL = 0.40
CLOSING_KERNEL_SIZE = 5


@dataclass(frozen=True)
class Candidate:
    contour: np.ndarray
    center: tuple[float, float]
    radius: float
    area: float

    @property
    def fill_ratio(self) -> float:
        return float(self.area / (np.pi * self.radius**2)) if self.radius else 0.0


def color_mask(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, (4, 105, 85), (28, 255, 255))


def close_mask(mask: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (CLOSING_KERNEL_SIZE,) * 2)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)


def find_candidates(mask: np.ndarray) -> list[Candidate]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        center, radius = cv2.minEnclosingCircle(contour)
        candidates.append(Candidate(contour, center, radius, cv2.contourArea(contour)))
    return candidates


def select_ball(
    candidates: list[Candidate], image_shape: tuple[int, ...], min_fill: float = BASELINE_FILL
) -> Candidate | None:
    max_radius = min(image_shape[:2]) / 4
    eligible = (
        item for item in candidates
        if 4 <= item.radius <= max_radius and item.fill_ratio >= min_fill
    )
    return max(eligible, key=lambda item: item.area, default=None)
