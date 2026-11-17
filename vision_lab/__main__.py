import argparse
from pathlib import Path

import cv2
import numpy as np

from .detection import BASELINE_FILL, RELAXED_FILL, close_mask, color_mask, find_candidates, select_ball
from .outputs import candidate_view, detection_coordinates, detection_label, detection_view, write_csv, write_panels


SAMPLE_VIDEO = Path(__file__).resolve().parents[1] / "samples" / "basketball.mp4"
COMPARISON_FRAMES = (60, 145, 180, 214, 225)
RECHECK_FRAMES = (100, 150, 200, 213, 265)


def read_frames(indices: tuple[int, ...]) -> dict[int, np.ndarray]:
    video = cv2.VideoCapture(str(SAMPLE_VIDEO))
    if not video.isOpened():
        video.release()
        raise OSError(f"Cannot open input video: {SAMPLE_VIDEO}")
    requested = set(indices)
    frames = {}
    try:
        for index in range(max(indices) + 1):
            ok, frame = video.read()
            if not ok:
                raise OSError(f"Could not read frame {index} from {SAMPLE_VIDEO.name}")
            if index in requested:
                frames[index] = frame
    finally:
        video.release()
    return frames


def task1(output: Path) -> None:
    frame = read_frames((100,))[100]
    write_panels(output / "frame-100.png",
                 ("Frame 100 / Original", frame),
                 ("Grayscale", cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)),
                 ("R channel", frame[:, :, 2]),
                 ("HSV mask", color_mask(frame)))


def task2(output: Path) -> None:
    rows = []
    for index, frame in read_frames((145, 180)).items():
        mask = color_mask(frame)
        candidates = find_candidates(mask)
        ball = select_ball(candidates, frame.shape)
        write_panels(output / f"frame-{index}.png",
                     (f"Frame {index} / Original", frame),
                     ("HSV mask", mask),
                     (f"Candidates / {len(candidates)}", candidate_view(frame, candidates)),
                     (detection_label("Baseline 0.58", ball), detection_view(frame, ball)))
        for number, item in enumerate(candidates, start=1):
            rows.append((index, number, round(item.radius, 3), round(item.fill_ratio, 4)))
    write_csv(output / "candidates.csv", ("frame", "candidate", "radius_px", "area_ratio"), rows)


def task3(output: Path) -> None:
    frame = read_frames((150,))[150]
    mask = color_mask(frame)
    closed = close_mask(mask)
    before = select_ball(find_candidates(mask), frame.shape)
    after = select_ball(find_candidates(closed), frame.shape)
    write_panels(output / "frame-150.png",
                 ("Frame 150 / HSV mask", mask),
                 ("After 5x5 closing", closed),
                 (detection_label("Before", before), detection_view(frame, before)),
                 (detection_label("After", after), detection_view(frame, after)))
    difference = cv2.absdiff(mask, closed)
    x, y, width, height = cv2.boundingRect(difference)
    left, top = max(0, x - 12), max(0, y - 12)
    crop = np.s_[top:top + max(height + 24, 64), left:left + max(width + 24, 72)]
    details = [
        cv2.resize(image[crop], None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
        for image in (mask, closed, difference)
    ]
    write_panels(output / "mask-detail.png",
                 ("Before / enlarged", details[0]),
                 ("After / enlarged", details[1]),
                 (f"Changed pixels: {cv2.countNonZero(difference)}", details[2]))


def task4(output: Path) -> None:
    frames = read_frames(COMPARISON_FRAMES + RECHECK_FRAMES)
    predictions = []
    blank = []
    for group, indices in (("comparison", COMPARISON_FRAMES), ("recheck", RECHECK_FRAMES)):
        for index in indices:
            frame = frames[index]
            candidates = find_candidates(color_mask(frame))
            baseline = select_ball(candidates, frame.shape, BASELINE_FILL)
            relaxed = select_ball(candidates, frame.shape, RELAXED_FILL)
            write_panels(output / f"frame-{index}.png",
                         (f"Frame {index} / Original", frame),
                         (detection_label("0.58", baseline), detection_view(frame, baseline)),
                         (detection_label("0.40", relaxed), detection_view(frame, relaxed)))
            predictions.append((
                group, index, int(baseline is not None), *detection_coordinates(baseline),
                int(relaxed is not None), *detection_coordinates(relaxed),
            ))
            blank.append((group, index, "", "", ""))
    write_csv(output / "predictions.csv",
              ("group", "frame", "found_058", "x_058", "y_058", "radius_058",
               "found_040", "x_040", "y_040", "radius_040"), predictions)
    template = output / "submission.csv"
    if not template.exists():
        write_csv(template, ("帧组", "帧号", "原图判断", "0.58 的结果", "0.40 的结果"), blank)


TASKS = {"task1": task1, "task2": task2, "task3": task3, "task4": task4}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RT traditional vision exercises.")
    parser.add_argument("task", choices=TASKS)
    parser.add_argument("--output-dir", type=Path, help="Defaults to runs/<task>.")
    args = parser.parse_args()
    output = args.output_dir if args.output_dir is not None else Path("runs") / args.task
    try:
        output.mkdir(parents=True, exist_ok=True)
        TASKS[args.task](output)
    except (OSError, cv2.error) as exc:
        parser.exit(1, f"{exc}\n")
    print(f"Output: {output.resolve()}")


if __name__ == "__main__":
    main()
