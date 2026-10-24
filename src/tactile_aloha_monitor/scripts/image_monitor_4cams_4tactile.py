#!/usr/bin/env python
"""
Image monitor for 4 USB cameras + 4 XENSE tactile sensors.

Displays a real-time 2×4 grid:
  Top row:    cam_high | cam_low | cam_left_wrist | cam_right_wrist
  Bottom row: left_gripper_inner | left_gripper_outer | right_gripper_inner | right_gripper_outer

Subscribes to:
  /usb_cam_high/image_raw
  /usb_cam_low/image_raw
  /usb_cam_left_wrist/image_raw
  /usb_cam_right_wrist/image_raw
  /tactile/left_gripper_inner/rectify
  /tactile/left_gripper_outer/rectify
  /tactile/right_gripper_inner/rectify
  /tactile/right_gripper_outer/rectify

The tactile /rectify topics arrive at native XENSE resolution (700, 400, 3),
rgb8: xense_tactile_publisher converts the XENSE SDK's native BGR to RGB before
publishing (it does the same to "difference" inside /raw). So every feed here is
RGB, while cv2.imshow wants BGR — this node swaps the channels back on the way
in. Camera frames are resized to the display cell size (~display_width x
~display_height) and centred in 700 px wide columns. Tactile frames (native
700 (h) x 400 (w), portrait) are rotated 90° (~tactile_rotation: "cw" default,
or "ccw") and drawn landscape at 400 (h) x 700 (w) — never scaled or stretched.

Which tactile topic is shown follows xense_tactile_publisher's own
~primary_output param (default "rectify"; 4arms_tactile_teleop.launch sets it
from its primary_output arg): the bottom row subscribes to
/tactile/<name>/<primary_output>. difference arrives rgb8 like rectify and is
handled the same way. depth arrives 32FC1 in mm and is drawn with a JET
colormap (see ~depth_max_mm), each cell's label showing that sensor's current
max depth.

Press 'q' to quit.
"""

# Required: this node has no interpreter wrapper (unlike xense_tactile_publisher.py),
# so ROS Noetic's catkin runs it on the system python3 (3.8 here), which doesn't
# support `list[str] | None`-style annotations natively — this defers evaluation
# so they're never actually executed. Without it the process crashes on import
# with "TypeError: 'type' object is not subscriptable" before rospy even starts.
from __future__ import annotations

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


# ---------------------------------------------------------------------------
# Feed descriptor: (display_label, topic, callback_name)
# ---------------------------------------------------------------------------
CAMERA_FEEDS = [
    ("cam_high",          "/usb_cam_high/image_raw"),
    ("cam_low",           "/usb_cam_low/image_raw"),
    ("cam_left_wrist",    "/usb_cam_left_wrist/image_raw"),
    ("cam_right_wrist",   "/usb_cam_right_wrist/image_raw"),
]

# Tactile topics are /tactile/<name>/<tactile_output>, where tactile_output is
# xense_tactile_publisher's ~primary_output ("rectify", "difference" or "depth").
TACTILE_SENSOR_NAMES = [
    "left_gripper_inner",
    "left_gripper_outer",
    "right_gripper_inner",
    "right_gripper_outer",
]

# Height in px of each row's label bar (see get_display_grid). Also used in
# main() to size the window to the grid's actual aspect ratio.
LABEL_BAR_H = 50

# Size every tactile cell is drawn at: the native 700 (h) x 400 (w) XENSE frame
# rotated 90° to landscape, not scaled or stretched to the camera cells' 4:3.
# Also used in main() for the window's aspect ratio.
TACTILE_H = 400
TACTILE_W = 700

# ~tactile_rotation values → cv2.rotate codes.
TACTILE_ROTATIONS = {
    "cw": cv2.ROTATE_90_CLOCKWISE,
    "ccw": cv2.ROTATE_90_COUNTERCLOCKWISE,
}

# Lower bound of the auto depth colour range (~depth_max_mm = 0). A resting
# XENSE gel reads ~0.006 mm max, so 0.1 mm keeps it dark blue instead of
# stretching that noise across the whole colormap.
DEPTH_AUTO_FLOOR_MM = 0.1


def depth_to_bgr(depth: np.ndarray, depth_max_mm: float) -> np.ndarray:
    """Float mm depth map → BGR JET image. depth_max_mm <= 0 means auto-scale per frame."""
    vmax = depth_max_mm if depth_max_mm > 0 else max(float(depth.max()), DEPTH_AUTO_FLOOR_MM)
    norm = np.clip(depth / vmax, 0.0, 1.0)
    return cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)


def no_signal_placeholder(height: int, width: int) -> np.ndarray:
    """Black height x width cell with "NO SIGNAL" centred on it."""
    placeholder = np.zeros((height, width, 3), dtype=np.uint8)
    text = "NO SIGNAL"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, 2, 3)
    cv2.putText(placeholder, text, ((width - tw) // 2, (height + th) // 2), font, 2, (255, 255, 255), 3)
    return placeholder


def pad_to_width(img: np.ndarray, width: int) -> np.ndarray:
    """Centre img between black bars so it is `width` px wide (unchanged if already that wide)."""
    pad = width - img.shape[1]
    if pad <= 0:
        return img
    left = pad // 2
    return cv2.copyMakeBorder(img, 0, 0, left, pad - left, cv2.BORDER_CONSTANT, value=(0, 0, 0))


class MultiFeedMonitor:
    def __init__(
        self,
        display_width: int = 640,
        display_height: int = 480,
        extra_outputs: list[str] | None = None,
        tactile_output: str = "rectify",
        depth_max_mm: float = 0.0,
        tactile_rotation: str = "cw",
    ):
        self.bridge = CvBridge()
        self.display_w = display_width
        self.display_h = display_height
        self.extra_outputs = extra_outputs or []
        self.tactile_output = tactile_output
        self.depth_max_mm = depth_max_mm
        self.tactile_rotate_code = TACTILE_ROTATIONS[tactile_rotation]
        self.feeds = CAMERA_FEEDS + [
            (name, f"/tactile/{name}/{tactile_output}") for name in TACTILE_SENSOR_NAMES
        ]
        self.images: dict[str, np.ndarray | None] = {}
        self.timestamps: dict[str, rospy.Time] = {}
        # Latest max depth (mm) per tactile feed, only filled for 32FC1 depth feeds.
        self.depth_peak_mm: dict[str, float] = {}

        for name, _ in self.feeds:
            self.images[name] = None
            self.timestamps[name] = rospy.Time(0)

    def subscribe_all(self) -> None:
        for name, topic in self.feeds:
            # Use a closure factory to capture `name`.
            cb = self._make_callback(name)
            rospy.Subscriber(topic, Image, cb)
            rospy.loginfo(f"  Subscribed: {topic} → {name}")

    def _make_callback(self, name: str):
        def callback(msg: Image) -> None:
            try:
                img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                # cv2.imshow always expects BGR-ordered arrays, and both rows arrive
                # RGB: /usb_cam_*/image_raw publishes rgb8 (empirically: this swap is
                # required for the cameras to look correct), and /tactile/*/rectify is
                # rgb8 too since xense_tactile_publisher converts the XENSE SDK's
                # native BGR before publishing. So every feed gets the same swap back.
                # The exception is /tactile/*/depth: 32FC1 mm, colour-mapped here.
                if msg.encoding == "32FC1":
                    self.depth_peak_mm[name] = float(img.max())
                    img = depth_to_bgr(img, self.depth_max_mm)
                elif img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif img.ndim == 3 and img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                # Tactile: portrait 700x400 → landscape 400x700 (see TACTILE_H/W).
                if name in TACTILE_SENSOR_NAMES:
                    img = cv2.rotate(img, self.tactile_rotate_code)
                self.images[name] = img
                self.timestamps[name] = msg.header.stamp
            except Exception:
                pass
        return callback

    def get_display_grid(self) -> np.ndarray:
        """Build a 2×4 composite image.

        Camera cells are display_w x display_h. Tactile cells are TACTILE_W x
        TACTILE_H (the rotated native frame, so normally no resize at all).
        Whichever is narrower is padded with black bars to the wider one's width
        so the two rows' columns line up.
        """
        col_w = max(self.display_w, TACTILE_W)
        n_cams = len(CAMERA_FEEDS)
        cam_placeholder = no_signal_placeholder(self.display_h, self.display_w)
        tac_placeholder = no_signal_placeholder(TACTILE_H, TACTILE_W)

        cells: list[np.ndarray] = []
        labels: list[str] = []

        for i, (name, _) in enumerate(self.feeds):
            is_tactile = i >= n_cams
            # cv2 sizes are (w, h).
            size = (TACTILE_W, TACTILE_H) if is_tactile else (self.display_w, self.display_h)
            img = self.images.get(name)
            if img is None:
                cell = tac_placeholder if is_tactile else cam_placeholder
            else:
                # Tactile frames are already TACTILE_W x TACTILE_H after rotation,
                # so this only resizes cameras (or a tactile frame of an unexpected size).
                cell = img if img.shape[1::-1] == size else cv2.resize(img, size)
                if cell.ndim == 2:
                    cell = cv2.cvtColor(cell, cv2.COLOR_GRAY2RGB)
            cells.append(pad_to_width(cell, col_w))
            # Depth cells are auto-scaled per frame, so colours alone aren't
            # comparable between sensors — put the actual max depth in the label.
            peak = self.depth_peak_mm.get(name)
            labels.append(name if peak is None else f"{name}  max {peak:.2f}mm")

        # Layout: 2 rows × 4 columns
        row1 = np.hstack(cells[0:4])   # cameras
        row2 = np.hstack(cells[4:8])   # tactile sensors

        # Add labels on the left side of each row. Sized to be readable at
        # normal viewing distance, not just technically present.
        label_h = LABEL_BAR_H
        row_font_scale = 1.0
        row_thickness = 2
        text_y = label_h - 14

        # Camera row label
        cam_label = np.zeros((label_h, row1.shape[1], 3), dtype=np.uint8)
        cv2.putText(cam_label, "Cameras (top): high | low | left_wrist | right_wrist", (10, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, row_font_scale, (200, 200, 200), row_thickness, cv2.LINE_AA)

        # Tactile row label — includes which /raw extras this run is publishing
        # (the primary output itself is always on, so it's not listed as an extra).
        # Names the tactile output only when it isn't the default rectify.
        tac_title = "Tactile" if self.tactile_output == "rectify" else f"Tactile {self.tactile_output}"
        tac_label = np.zeros((label_h, row2.shape[1], 3), dtype=np.uint8)
        cv2.putText(tac_label, f"{tac_title} (bottom): left_gripper_inner | left_gripper_outer | right_gripper_inner | right_gripper_outer",
                    (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, row_font_scale, (200, 200, 200), row_thickness, cv2.LINE_AA)
        if self.extra_outputs:
            extras_text = f"extras: {', '.join(self.extra_outputs)}"
            extras_color = (0, 200, 255)  # amber — draws the eye when extras are on
        else:
            extras_text = f"extras: none ({self.tactile_output} only)"
            extras_color = (150, 150, 150)
        (extras_w, _), _ = cv2.getTextSize(extras_text, cv2.FONT_HERSHEY_SIMPLEX, row_font_scale, row_thickness)
        cv2.putText(tac_label, extras_text, (row2.shape[1] - extras_w - 10, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, row_font_scale, extras_color, row_thickness, cv2.LINE_AA)

        # Column labels on each cell — black outline behind the green text so it
        # stays readable regardless of what's under it.
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        font_scale_cell = 1.1
        cell_thickness = 2
        for i, (cell, label) in enumerate(zip(cells, labels)):
            overlay = cell.copy()
            cv2.putText(overlay, label, (10, 40), font_face, font_scale_cell,
                        (0, 0, 0), cell_thickness + 3, cv2.LINE_AA)
            cv2.putText(overlay, label, (10, 40), font_face, font_scale_cell,
                        (0, 255, 0), cell_thickness, cv2.LINE_AA)
            cells[i] = overlay

        row1_labeled = np.hstack(cells[0:4])
        row2_labeled = np.hstack(cells[4:8])

        grid = np.vstack((cam_label, row1_labeled, tac_label, row2_labeled))
        return grid


def main() -> None:
    # NOT anonymous: same reasoning as xense_tactile_publisher.py — anonymous=True
    # would make "~param" lookups resolve to a random suffixed name that the
    # launch file's nested <param> tags were never written to.
    rospy.init_node("image_monitor_4cams_4tactile")

    display_w = rospy.get_param("~display_width", 640)
    display_h = rospy.get_param("~display_height", 480)
    # Depth (mm) drawn as full red when showing /tactile/*/depth. 0 = auto-scale
    # each frame to its max, floored at DEPTH_AUTO_FLOOR_MM.
    depth_max_mm = float(rospy.get_param("~depth_max_mm", 0.0))
    # Which way to turn the portrait tactile frames to show them landscape.
    tactile_rotation = str(rospy.get_param("~tactile_rotation", "cw")).strip().lower()
    if tactile_rotation not in TACTILE_ROTATIONS:
        rospy.logwarn(f"Unknown ~tactile_rotation {tactile_rotation!r} (valid: cw, ccw), using cw")
        tactile_rotation = "cw"

    # Which tactile output xense_tactile_publisher is publishing as its preview
    # topic (and always-on /raw entry), and what it was configured to also
    # publish in /raw on top of that — both are that node's own private params,
    # so they're read here off its full path rather than "~".
    tactile_output = str(rospy.get_param("/xense_tactile_publisher/primary_output", "rectify")).strip().lower()
    extra_outputs_param = rospy.get_param("/xense_tactile_publisher/extra_outputs", "")
    extra_outputs = [s.strip() for s in extra_outputs_param.split(",") if s.strip()]
    if extra_outputs:
        rospy.loginfo(f"Tactile /raw extras this run: {', '.join(extra_outputs)}")
    else:
        rospy.loginfo(f"Tactile /raw extras this run: none ({tactile_output} only).")

    monitor = MultiFeedMonitor(
        display_w, display_h, extra_outputs=extra_outputs,
        tactile_output=tactile_output, depth_max_mm=depth_max_mm,
        tactile_rotation=tactile_rotation,
    )
    monitor.subscribe_all()

    rospy.loginfo("4-cam + 4-tactile monitor running. Press 'q' in the display window to quit.")

    # Give subscribers a moment to receive the first messages.
    rospy.sleep(0.5)

    # Match the actual grid's aspect ratio (4 columns of max(display_w, TACTILE_W)
    # wide, display_h + TACTILE_H + 2*label_h tall) so the window scales the grid
    # uniformly — tactile cells keep their 700:400 landscape shape on screen — rather than
    # stretching it and shrinking the text further (a fixed 1920x600 window against
    # a ~2560x1060 image once made everything look too small). Still resizable.
    grid_w = 4 * max(display_w, TACTILE_W)
    grid_h = display_h + TACTILE_H + 2 * LABEL_BAR_H
    window_w = 1920
    window_h = int(window_w * grid_h / grid_w)
    cv2.namedWindow("4 Cameras + 4 Tactile (4arms_tactile)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("4 Cameras + 4 Tactile (4arms_tactile)", window_w, window_h)

    # Cap the redraw rate. The loop below rebuilds the whole grid every pass —
    # 4 camera cv2.resize calls to display_w x display_h plus the hstack/vstack of
    # a ~2800x980 composite — and cv2.waitKey(1) alone bounds that at ~1000 Hz,
    # which measured at ~200% CPU for a window nobody can read that fast. This
    # node shares the box with xense_tactile_publisher's inference threads, and
    # those are what actually need the cores, so give the preview a rate it can
    # be honest about: 15Hz still looks live to a human.
    display_rate = rospy.Rate(rospy.get_param("~display_rate_hz", 15))

    while not rospy.is_shutdown():
        grid = monitor.get_display_grid()
        cv2.imshow("4 Cameras + 4 Tactile (4arms_tactile)", grid)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        display_rate.sleep()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
