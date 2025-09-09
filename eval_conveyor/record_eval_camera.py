import argparse
import os
import signal
import sys
import time
from datetime import datetime
from typing import Optional, Tuple, Union

import cv2


def parse_index_or_path(value: str) -> Union[int, str]:
    """Return camera index as int if numeric, else return as path string."""
    try:
        return int(value)
    except ValueError:
        return value


def open_capture(
    index_or_path: Union[int, str], width: int, height: int, fps: int
) -> cv2.VideoCapture:
    """Create and configure a VideoCapture.

    OpenCV may not honor all requested properties depending on the backend and
    camera driver. We still set them and later report the actual values.
    """
    cap = cv2.VideoCapture(index_or_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera: {index_or_path}")

    # Request properties
    if width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    if height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    if fps > 0:
        cap.set(cv2.CAP_PROP_FPS, float(fps))

    return cap


def get_actual_capture_props(cap: cv2.VideoCapture) -> Tuple[int, int, float]:
    """Read back capture properties actually in effect."""
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = float(cap.get(cv2.CAP_PROP_FPS))
    return actual_width, actual_height, actual_fps


def build_output_path(
    output: Optional[str], index_or_path: Union[int, str], ext: str
) -> str:
    """Build an output path if not provided using timestamp and source name."""
    if output:
        return output
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    source_name = (
        f"cam{index_or_path}"
        if isinstance(index_or_path, int)
        else os.path.splitext(os.path.basename(str(index_or_path)))[0]
    )
    return f"record_{source_name}_{timestamp}.{ext.lstrip('.')}"


def create_writer(
    path: str, fourcc_str: str, fps: float, frame_size: Tuple[int, int]
) -> cv2.VideoWriter:
    """Create VideoWriter for the given path and parameters."""
    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    writer = cv2.VideoWriter(path, fourcc, fps if fps > 0 else 30.0, frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for: {path}")
    return writer


def record(
    index_or_path: Union[int, str],
    width: int,
    height: int,
    fps: int,
    duration: Optional[float],
    output: Optional[str],
    fourcc: str,
    ext: str,
    preview: bool,
) -> None:
    cap = open_capture(index_or_path, width, height, fps)
    actual_width, actual_height, actual_fps = get_actual_capture_props(cap)

    output_path = build_output_path(output, index_or_path, ext)
    writer = create_writer(
        output_path, fourcc, actual_fps if actual_fps > 0 else float(fps), (actual_width, actual_height)
    )

    print(
        f"Recording from {index_or_path} -> {output_path}\n"
        f"Requested: {width}x{height}@{fps} | Actual: {actual_width}x{actual_height}@{actual_fps:.2f}"
    )

    frames_written = 0
    start_time = time.time()
    end_time = start_time + duration if duration and duration > 0 else None

    interrupted = False

    def handle_sigint(_sig, _frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, handle_sigint)

    while True:
        if interrupted:
            print("Interrupted by user (SIGINT). Stopping...")
            break
        if end_time is not None and time.time() >= end_time:
            print("Reached requested duration. Stopping...")
            break

        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame from camera. Stopping...")
            break

        if frame.shape[1] != actual_width or frame.shape[0] != actual_height:
            # Resize to writer frame size if capture fluctuates
            frame = cv2.resize(frame, (actual_width, actual_height))

        writer.write(frame)
        frames_written += 1

        if preview:
            cv2.imshow("Recording - press q to stop", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("'q' pressed. Stopping...")
                break

    elapsed = max(time.time() - start_time, 1e-6)
    approx_fps = frames_written / elapsed
    print(f"Saved {frames_written} frames in {elapsed:.2f}s (≈{approx_fps:.2f} FPS)")

    writer.release()
    cap.release()
    if preview:
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Record video from a single camera (OpenCV). "
            "Example matching your config: --source 8 --width 640 --height 480 --fps 30"
        )
    )
    parser.add_argument(
        "--source",
        type=str,
        default="8",
        help=(
            "Camera index (e.g., '8') or video device/file path. "
            "If numeric, treated as index; otherwise as path."
        ),
    )
    parser.add_argument("--width", type=int, default=640, help="Requested capture width")
    parser.add_argument("--height", type=int, default=480, help="Requested capture height")
    parser.add_argument("--fps", type=int, default=30, help="Requested capture FPS")
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Optional duration in seconds. Omit to record until interrupted",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output video path. Defaults to record_{source}_{timestamp}.mp4",
    )
    parser.add_argument(
        "--fourcc",
        type=str,
        default="mp4v",
        help="FourCC codec (e.g., mp4v, avc1, XVID, MJPG). Default: mp4v",
    )
    parser.add_argument(
        "--ext",
        type=str,
        default="mp4",
        help="Output file extension (mp4 or avi). Default: mp4",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable preview window (saves CPU).",
    )

    args = parser.parse_args()

    source = parse_index_or_path(args.source)
    try:
        record(
            index_or_path=source,
            width=args.width,
            height=args.height,
            fps=args.fps,
            duration=args.duration,
            output=args.output,
            fourcc=args.fourcc,
            ext=args.ext,
            preview=not args.no_preview,
        )
    except Exception as exc:  # noqa: BLE001 - prefer explicit exit with message for CLI tool
        print(f"Error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()


