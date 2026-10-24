#!/usr/bin/env python3
"""Image monitor for USB cameras + full-frame XENSE tactile topics."""

from __future__ import annotations

import pickle

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import UInt8MultiArray


CAMERA_FEEDS = [
    ("cam_high", "/usb_cam_high/image_raw"),
    ("cam_low", "/usb_cam_low/image_raw"),
    ("cam_left_wrist", "/usb_cam_left_wrist/image_raw"),
    ("cam_right_wrist", "/usb_cam_right_wrist/image_raw"),
]

CAMERA_TOPIC_BY_NAME = dict(CAMERA_FEEDS)

XENSE_FEEDS = [
    ("left_fingertip", "/xense/left_fingertip/frame"),
    ("left_knuckle", "/xense/left_knuckle/frame"),
    ("right_fingertip", "/xense/right_fingertip/frame"),
    ("right_knuckle", "/xense/right_knuckle/frame"),
]

XENSE_TOPIC_BY_NAME = dict(XENSE_FEEDS)
ALL_FEEDS = CAMERA_FEEDS + XENSE_FEEDS
DEFAULT_CAMERA_NAMES = tuple(name for name, _ in CAMERA_FEEDS)
DEFAULT_XENSE_NAMES = tuple(name for name, _ in XENSE_FEEDS)


def parse_name_list(value: str | list[str] | tuple[str, ...], default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return default
    if isinstance(value, str):
        names = tuple(name.strip() for name in value.split(",") if name.strip())
    else:
        names = tuple(str(name).strip() for name in value if str(name).strip())
    return names or default


def validate_names(names: tuple[str, ...], known_names: dict[str, str], kind: str) -> None:
    unknown = [name for name in names if name not in known_names]
    if unknown:
        raise ValueError(f"Unknown {kind} feed(s): {unknown}. Choices: {tuple(known_names)}")


class MultiFeedXenseMonitor:
    def __init__(
        self,
        display_width: int = 640,
        display_height: int = 480,
        camera_names: tuple[str, ...] = DEFAULT_CAMERA_NAMES,
        xense_names: tuple[str, ...] = DEFAULT_XENSE_NAMES,
        bottom_placeholders: int = 0,
    ):
        validate_names(camera_names, CAMERA_TOPIC_BY_NAME, "camera")
        validate_names(xense_names, XENSE_TOPIC_BY_NAME, "XENSE")
        self.bridge = CvBridge()
        self.display_w = int(display_width)
        self.display_h = int(display_height)
        self.camera_names = camera_names
        self.xense_names = xense_names
        self.bottom_placeholders = max(0, int(bottom_placeholders))
        self.images: dict[str, np.ndarray | None] = {name: None for name, _ in ALL_FEEDS}

    def subscribe_all(self) -> None:
        for name, topic in CAMERA_FEEDS:
            rospy.Subscriber(topic, Image, self._make_camera_callback(name), queue_size=1)
            rospy.loginfo("Subscribed camera: %s -> %s", topic, name)
        for name, topic in XENSE_FEEDS:
            rospy.Subscriber(topic, UInt8MultiArray, self._make_xense_callback(name), queue_size=1)
            rospy.loginfo("Subscribed XENSE: %s -> %s", topic, name)

    def _make_camera_callback(self, name: str):
        def callback(msg: Image) -> None:
            try:
                img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                if img.ndim == 3 and img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.images[name] = img
            except Exception as exc:  # noqa: BLE001
                rospy.logwarn_throttle(5.0, "Camera monitor failed for %s: %s", name, exc)

        return callback

    def _make_xense_callback(self, name: str):
        def callback(msg: UInt8MultiArray) -> None:
            try:
                payload = pickle.loads(bytes(msg.data))
                frame = payload.get("frames", {})
                self.images[name] = self._xense_frame_to_rgb(frame)
            except Exception as exc:  # noqa: BLE001
                rospy.logwarn_throttle(5.0, "XENSE monitor failed for %s: %s", name, exc)

        return callback

    @staticmethod
    def _as_uint8_image(data: np.ndarray) -> np.ndarray:
        arr = np.asarray(data)
        if arr.dtype == np.uint8:
            return arr
        arr = arr.astype(np.float32, copy=False)
        finite = arr[np.isfinite(arr)]
        if finite.size and float(np.nanmax(finite)) <= 1.5:
            arr = arr * 255.0
        return np.clip(np.rint(arr), 0, 255).astype(np.uint8)

    def _xense_frame_to_rgb(self, frame: dict) -> np.ndarray:
        img = frame.get("rectify_rgb")
        if img is None:
            img = frame.get("rectify")
        if img is None:
            raise KeyError("frame has no rectify_rgb or rectify")

        img = self._as_uint8_image(img)
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if img.ndim == 3 and img.shape[2] >= 3:
            return img[:, :, :3]
        raise ValueError(f"Unsupported XENSE display image shape: {img.shape}")

    def _placeholder(self, text: str = "NO SIGNAL") -> np.ndarray:
        placeholder = np.zeros((self.display_h, self.display_w, 3), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(text, font, 2, 3)
        cv2.putText(
            placeholder,
            text,
            ((self.display_w - tw) // 2, (self.display_h + th) // 2),
            font,
            2,
            (255, 255, 255),
            3,
        )
        return placeholder

    def _blank(self) -> np.ndarray:
        return np.zeros((self.display_h, self.display_w, 3), dtype=np.uint8)

    def _feed_cell(self, name: str, placeholder: np.ndarray, target: tuple[int, int]) -> np.ndarray:
        img = self.images.get(name)
        if img is None:
            cell = placeholder
        else:
            cell = cv2.resize(img, target)
            if cell.ndim == 2:
                cell = cv2.cvtColor(cell, cv2.COLOR_GRAY2RGB)
        overlay = cell.copy()
        cv2.putText(overlay, name, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        return overlay

    def get_display_grid(self) -> np.ndarray:
        target = (self.display_w, self.display_h)
        placeholder = self._placeholder()
        row1_cells = [self._feed_cell(name, placeholder, target) for name in self.camera_names]
        row2_cells = [self._feed_cell(name, placeholder, target) for name in self.xense_names]
        row2_cells.extend(self._blank() for _ in range(self.bottom_placeholders))

        if not row1_cells:
            row1_cells = [self._blank()]
        if not row2_cells:
            row2_cells = [self._blank()]

        max_cols = max(len(row1_cells), len(row2_cells))
        row1_cells.extend(self._blank() for _ in range(max_cols - len(row1_cells)))
        row2_cells.extend(self._blank() for _ in range(max_cols - len(row2_cells)))

        row1 = np.hstack(row1_cells)
        row2 = np.hstack(row2_cells)

        label_h = 30
        cam_label = np.zeros((label_h, row1.shape[1], 3), dtype=np.uint8)
        cv2.putText(
            cam_label,
            "Cameras (top): " + " | ".join(self.camera_names),
            (10, label_h - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )
        xense_label = np.zeros((label_h, row2.shape[1], 3), dtype=np.uint8)
        cv2.putText(
            xense_label,
            "XENSE rectify_rgb (bottom): " + " | ".join(self.xense_names),
            (10, label_h - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )
        return np.vstack((cam_label, row1, xense_label, row2))

    def window_name(self) -> str:
        return f"{len(self.camera_names)} Cameras + {len(self.xense_names)} XENSE"


def main() -> None:
    rospy.init_node("image_monitor_4cams_4xense", anonymous=True)
    monitor = MultiFeedXenseMonitor(
        display_width=rospy.get_param("~display_width", 640),
        display_height=rospy.get_param("~display_height", 480),
        camera_names=parse_name_list(rospy.get_param("~camera_names", ",".join(DEFAULT_CAMERA_NAMES)), DEFAULT_CAMERA_NAMES),
        xense_names=parse_name_list(rospy.get_param("~xense_names", ",".join(DEFAULT_XENSE_NAMES)), DEFAULT_XENSE_NAMES),
        bottom_placeholders=rospy.get_param("~bottom_placeholders", 0),
    )
    monitor.subscribe_all()
    rospy.sleep(0.5)

    window_name = monitor.window_name()
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1920, 600)
    while not rospy.is_shutdown():
        cv2.imshow(window_name, monitor.get_display_grid())
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
