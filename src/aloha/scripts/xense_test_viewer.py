#!/home/robot/miniforge3/envs/lerobot/bin/python
"""
XENSE single-sensor test viewer (no ROS topics published).

Reads one XENSE sensor directly through xensesdk and shows its Rectify,
Difference and Depth outputs side by side in one OpenCV window:

    | rectify | difference | depth (JET colormap) |

All three come from a single ``selectSensorInfo(Rectify, Difference, Depth)``
call, so every displayed triple belongs to the same sensor frame — they are
synchronised by construction, not by matching timestamps afterwards.

Rectify and Difference are BGR straight from the SDK, which is what
cv2.imshow wants, so no channel swap is done. Depth is float32 in mm, shape
(700, 400); see ``~depth_max_mm`` for how it is scaled to colours.

Keys (with the window focused):
    r        reset the sensor's reference image (re-zeroes Difference/Depth)
    q / ESC  quit (closing the window also quits)

ROS params:
    ~serial        sensor serial number (default "OG000635")
    ~use_gpu       run the SDK's inference on GPU (default true)
    ~depth_max_mm  depth value shown as full red. 0 (default) = auto-scale to
                   each frame's max, but never below DEPTH_AUTO_FLOOR_MM so
                   resting-sensor noise (~0.006 mm) is not stretched to full
                   colour range.
"""

from __future__ import annotations

import cv2
import numpy as np
import rospy
from xensesdk import Sensor  # type: ignore[import-untyped]

# Lower bound of the auto depth range. At rest OG000635 reads ~0.006 mm max,
# so 0.1 mm keeps an untouched gel dark blue instead of noisy rainbow.
DEPTH_AUTO_FLOOR_MM: float = 0.1

WINDOW_TITLE_FMT: str = "XENSE {serial}  (r: reset reference, q/ESC: quit)"


def _depth_to_bgr(depth: np.ndarray, depth_max_mm: float) -> tuple[np.ndarray, float]:
    """Map a float mm depth map to a BGR JET image; also return the mm value used as full scale."""
    vmax = depth_max_mm if depth_max_mm > 0 else max(float(depth.max()), DEPTH_AUTO_FLOOR_MM)
    norm = np.clip(depth / vmax, 0.0, 1.0)
    return cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET), vmax


def _label(img: np.ndarray, lines: list[str]) -> np.ndarray:
    """Draw outlined text lines in the top-left corner of a copy of ``img``."""
    out = np.ascontiguousarray(img).copy()
    for i, text in enumerate(lines):
        org = (8, 24 + 24 * i)
        cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def compose_frame(
    rectify: np.ndarray, difference: np.ndarray, depth: np.ndarray, depth_max_mm: float
) -> np.ndarray:
    """Build the labelled [rectify | difference | depth] panel shown in the window."""
    depth_bgr, vmax = _depth_to_bgr(depth, depth_max_mm)
    scale_mode = "fixed" if depth_max_mm > 0 else "auto"
    return np.hstack([
        _label(rectify, ["rectify"]),
        _label(difference, ["difference"]),
        _label(depth_bgr, [
            f"depth  max {float(depth.max()):.3f} mm",
            f"scale 0-{vmax:.2f} mm ({scale_mode})",
        ]),
    ])


def main() -> None:
    rospy.init_node("xense_test_viewer")

    serial = str(rospy.get_param("~serial", "OG000635"))
    use_gpu = rospy.get_param("~use_gpu", True)
    depth_max_mm = float(rospy.get_param("~depth_max_mm", 0.0))

    found = Sensor.scanSerialNumber()
    if serial not in found:
        rospy.logfatal(f"XENSE sensor {serial} NOT found (connected: {sorted(found)})")
        return

    sensor = Sensor.create(cam_id=serial, use_gpu=use_gpu)
    if not sensor:
        rospy.logfatal(f"Sensor.create returned {sensor!r} for {serial}")
        return

    window = WINDOW_TITLE_FMT.format(serial=serial)
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    rospy.loginfo(f"Showing XENSE {serial}: rectify | difference | depth. r = reset reference, q/ESC = quit.")

    try:
        while not rospy.is_shutdown():
            rectify, difference, depth = sensor.selectSensorInfo(
                Sensor.OutputType.Rectify,
                Sensor.OutputType.Difference,
                Sensor.OutputType.Depth,
            )
            if rectify is None or difference is None or depth is None:
                rospy.logwarn_throttle(5.0, f"XENSE {serial}: incomplete frame, skipping")
                continue

            cv2.imshow(window, compose_frame(rectify, difference, depth, depth_max_mm))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                sensor.resetReferenceImage()
                rospy.loginfo(f"XENSE {serial}: reference image reset.")
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break  # window closed with the title-bar X
    finally:
        sensor.release()
        cv2.destroyAllWindows()
        rospy.loginfo("XENSE test viewer shut down.")


if __name__ == "__main__":
    main()
