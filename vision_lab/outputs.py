import csv
from pathlib import Path

import cv2
import numpy as np

from .detection import Candidate


def write_csv(path: Path, columns: tuple[str, ...], rows: list[tuple]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        writer.writerows(rows)


def _label(image: np.ndarray, title: str) -> np.ndarray:
    panel = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image
    panel = cv2.copyMakeBorder(panel, 39, 0, 0, 0, cv2.BORDER_CONSTANT, value=(35, 29, 23))
    cv2.putText(panel, title, (9, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def write_panels(path: Path, *panels: tuple[str, np.ndarray]) -> None:
    sheet = cv2.hconcat([_label(image, title) for title, image in panels])
    if not cv2.imwrite(str(path), sheet):
        raise OSError(f"Cannot write image: {path}")


def candidate_view(frame: np.ndarray, candidates: list[Candidate]) -> np.ndarray:
    view = frame.copy()
    for number, item in enumerate(candidates, start=1):
        center = tuple(round(value) for value in item.center)
        cv2.drawContours(view, [item.contour], -1, (230, 75, 235), 2)
        cv2.putText(view, str(number), (center[0] + 4, max(15, center[1] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
    return view


def detection_view(frame: np.ndarray, item: Candidate | None) -> np.ndarray:
    view = frame.copy()
    if item is not None:
        center = tuple(round(value) for value in item.center)
        cv2.circle(view, center, round(item.radius), (85, 255, 95), 2, cv2.LINE_AA)
        cv2.circle(view, center, 3, (85, 255, 95), -1, cv2.LINE_AA)
    return view


def detection_label(prefix: str, item: Candidate | None) -> str:
    return f"{prefix} / {'found' if item is not None else 'none'}"


def detection_coordinates(item: Candidate | None) -> tuple[int | str, ...]:
    if item is None:
        return ("", "", "")
    return (round(item.center[0]), round(item.center[1]), round(item.radius))
