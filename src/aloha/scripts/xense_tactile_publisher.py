#!/home/robot/miniforge3/envs/lerobot/bin/python
"""
XENSE Tactile Publisher for ROS (4-arm ALOHA with tactile).

Reads 4 XENSE G1-WS tactile sensors and publishes two things per sensor:

  1. The full, canonical data for that frame, bundled into a single pickled
     payload (frames dict + host-side read timing), mirroring the approach
     used by act_tactile's xense_publisher.py. This is the one topic to
     subscribe to for anything downstream (recording, training, analysis) —
     it always publishes every frame_rate cycle, real-time:

       /tactile/left_gripper_inner/raw
       /tactile/left_gripper_outer/raw
       /tactile/right_gripper_inner/raw
       /tactile/right_gripper_outer/raw

     std_msgs/UInt8MultiArray whose ``data`` is
     ``pickle.dumps({"frames": {...}, "meta": {...}})``. ``frames`` always has
     a ``"rectify"`` entry; the other nine SDK output types — Difference,
     Depth, Force, ForceNorm, ForceResultant, Marker2D, Mesh3D, Mesh3DInit,
     Mesh3DFlow, TimeStamp (see RAW_OUTPUT_TYPES) — are opt-in via the
     ``~extra_outputs`` param (comma-separated keys, e.g. "depth,force").
     Default is empty: only rectify is requested from the SDK and published —
     nothing else is read at all, not just "not published". Everything in one
     message (rectify plus every extra) comes from a single
     ``selectSensorInfo()`` call, so it is all the same sensor frame.
     Whatever is selected is at native XENSE SDK
     resolution/dtype (no resize). The two colour images — Rectify and
     Difference — are converted from the SDK's native BGR to RGB before
     publishing; every other output is byte-for-byte what the SDK returned,
     including the 3-channel ones (Force, ForceNorm, Mesh3D*), whose channels
     are x/y/z components rather than colours. ``meta`` holds the host-side
     read_start/read_end times for that snapshot.

  2. A convenience preview of just the rectified image, for live monitoring
     only — not meant as a data source, so it's throttled (see
     ``~preview_every_n_frames``) instead of publishing every frame:

       /tactile/left_gripper_inner/rectify
       /tactile/left_gripper_outer/rectify
       /tactile/right_gripper_inner/rectify
       /tactile/right_gripper_outer/rectify

     sensor_msgs/Image, shape (700, 400, 3), encoding rgb8 — same native
     resolution and same RGB order as the ``rectify`` entry inside ``/raw``,
     just a duplicate copy published less often and as a standard Image message
     so tools like image_monitor/rqt_image_view can subscribe directly.

Everything above describes the default ``~primary_output: rectify``. Setting
it to ``depth`` or ``difference`` (4arms_tactile_teleop.launch exposes it as the
``primary_output`` arg) swaps rectify for that output as the always-on one:
``/raw``'s ``frames`` then always has that key instead of ``"rectify"``, and the
preview topic becomes ``/tactile/<name>/<primary_output>``:
  depth:      (700, 400) float32 mm; preview sensor_msgs/Image 32FC1
  difference: (700, 400, 3) uint8, BGR->RGB like rectify; preview rgb8
Rectify is then not read at all unless listed in ``~extra_outputs``.
"""

from __future__ import annotations

import pickle
import threading
import time
from typing import Any

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import UInt8MultiArray

# ---------------------------------------------------------------------------
# Per‑robot hardware — serial numbers from act_tactile / xense_recorder.py
# ---------------------------------------------------------------------------
TACTILE_SENSORS: dict[str, str] = {
    "left_gripper_inner": "OG000707",
    "left_gripper_outer": "OG000635",
    "right_gripper_inner": "OG000614",
    "right_gripper_outer": "OG000706",
}

# The always-on output, chosen with ~primary_output: read from the SDK every
# cycle, always bundled into /raw under its key, and also published as the
# /tactile/<name>/<key> preview Image.
#   key -> (Sensor.OutputType attr, preview Image encoding)
#   "rectify":    (700, 400, 3) uint8, BGR from the SDK, published as RGB
#   "difference": (700, 400, 3) uint8, BGR from the SDK, published as RGB
#   "depth":      (700, 400) float32, mm, published as-is
PRIMARY_OUTPUTS: dict[str, tuple[str, str]] = {
    "rectify": ("Rectify", "rgb8"),
    "difference": ("Difference", "rgb8"),
    "depth": ("Depth", "32FC1"),
}

# Default ~primary_output, so launch files that don't set it behave as before.
DEFAULT_PRIMARY_OUTPUT: str = "rectify"

# Peak depth of the synthetic press blob in --mock mode with primary_output=depth.
MOCK_DEPTH_PEAK_MM: float = 1.0

# ROS topic suffix for the pickled bundle of the remaining raw output types.
RAW_TOPIC_SUFFIX: str = "raw"

# Shape used only by --mock mode to fabricate a synthetic rectify frame.
# The real sensor path publishes whatever shape selectSensorInfo() returns,
# unresized — this just matches the documented native (700, 400, 3) so mock
# testing exercises the same shape.
MOCK_H: int = 700
MOCK_W: int = 400

# Every Sensor.OutputType that ~extra_outputs can add to the "raw" topic,
# mapped to the key used inside the pickled ``frames`` dict. Whichever one is
# the primary output (see PRIMARY_OUTPUTS) is always in "raw" already — it's
# read once in _reader_loop and reused for both the "raw" entry and the
# separate preview Image — so _parse_extra_outputs drops it from the extras.
#   Sensor.OutputType.Rectify         -> (700, 400, 3), BGR from the SDK,
#                                        published as RGB (see COLOUR_IMAGE_KEYS)
#   Sensor.OutputType.Difference      -> (700, 400, 3), BGR from the SDK,
#                                        published as RGB (see COLOUR_IMAGE_KEYS)
#   Sensor.OutputType.Depth           -> (700, 400), mm
#   Sensor.OutputType.Force           -> (35, 20, 3)
#   Sensor.OutputType.ForceNorm       -> (35, 20, 3)
#   Sensor.OutputType.ForceResultant  -> (6,)
#   Sensor.OutputType.Marker2D        -> (35, 20, 2)
#   Sensor.OutputType.Mesh3D          -> (35, 20, 3)
#   Sensor.OutputType.Mesh3DInit      -> (35, 20, 3)
#   Sensor.OutputType.Mesh3DFlow      -> (35, 20, 3)
#   Sensor.OutputType.TimeStamp       -> sensor-side capture time, seconds
RAW_OUTPUT_TYPES: dict[str, str] = {
    "Rectify": "rectify",
    "Difference": "difference",
    "Depth": "depth",
    "Force": "force",
    "ForceNorm": "force_norm",
    "ForceResultant": "force_resultant",
    "Marker2D": "marker2d",
    "Mesh3D": "mesh3d",
    "Mesh3DInit": "mesh3d_init",
    "Mesh3DFlow": "mesh3d_flow",
    "TimeStamp": "timestamp",
}

# The keys in RAW_OUTPUT_TYPES that are colour images, and so get the BGR->RGB
# swap on the way out. Every other 3-channel output (force, force_norm, mesh3d,
# ...) carries x/y/z components rather than colours and must be handed on exactly
# as the SDK returned it. Whichever of these is the primary output is read
# outside the extras loop and swapped there instead (see _convert_primary).
COLOUR_IMAGE_KEYS: frozenset = frozenset({"rectify", "difference"})


def _parse_extra_outputs(raw_param: str, primary_output: str) -> dict[str, str]:
    """Parse the ``~extra_outputs`` param into an {attr: key} subset of RAW_OUTPUT_TYPES.

    ``raw_param`` is a comma-separated list of the snake_case keys above (e.g.
    "depth,force"). Empty/unset means "primary output only" — nothing extra is
    read from the SDK or bundled into /raw. Unknown names are dropped with a
    warning rather than failing the whole node, and so is ``primary_output``
    itself, which is always in /raw already.
    """
    key_to_attr = {key: attr for attr, key in RAW_OUTPUT_TYPES.items()}
    requested = [s.strip().lower() for s in raw_param.split(",") if s.strip()]
    selected: dict[str, str] = {}
    for key in requested:
        if key == primary_output:
            rospy.logwarn(
                f"xense_tactile_publisher: ~extra_outputs entry {key!r} is already the "
                f"primary output (always in /raw), ignoring"
            )
            continue
        attr = key_to_attr.get(key)
        if attr is None:
            rospy.logwarn(
                f"xense_tactile_publisher: unknown ~extra_outputs entry {key!r}, ignoring "
                f"(valid: {', '.join(sorted(key_to_attr))})"
            )
            continue
        selected[attr] = key
    return selected

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_uint8(gray: np.ndarray) -> np.ndarray:
    """Float32 [0, 1] → uint8 [0, 255]."""
    return np.clip(gray * 255.0, 0, 255).astype(np.uint8)


def _to_uint8_image(data: np.ndarray) -> np.ndarray:
    """Defensively coerce a raw XENSE colour image to a contiguous uint8 array.

    No resize, no colour-order change (that's _bgr_to_rgb's job) — this only
    normalises dtype in case the SDK hands back float32 (either [0, 1] or
    [0, 255] range) instead of uint8.
    """
    arr = np.asarray(data)
    if arr.dtype != np.uint8:
        if arr.dtype.kind == "f" and arr.max() <= 1.5:
            arr = _to_uint8(arr.astype(np.float32))
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _bgr_to_rgb(arr: np.ndarray) -> np.ndarray:
    """Swap channel order on a 3-channel image; pass anything else through.

    The XENSE SDK returns Rectify and Difference as BGR, but everything this
    node publishes is RGB, so both go through here on the way out. Done with a
    numpy reverse-slice rather than cv2.cvtColor so it stays dtype-agnostic —
    Difference is stored verbatim from the SDK and isn't guaranteed uint8.

    Only call this on actual colour images: Force/ForceNorm/Mesh3D are 3-channel
    too, but those channels are x/y/z components and swapping them corrupts the
    data. COLOUR_IMAGE_KEYS is the list of extras this is valid for.
    """
    if arr.ndim == 3 and arr.shape[2] == 3:
        return np.ascontiguousarray(arr[..., ::-1])
    return arr


def _convert_primary(primary_output: str, data: np.ndarray) -> np.ndarray:
    """Turn the SDK's primary output into what gets published (see PRIMARY_OUTPUTS).

    rectify / difference: uint8 (rgb8 preview needs it), BGR -> RGB.
    depth: contiguous float32 mm, values untouched.
    """
    if primary_output in COLOUR_IMAGE_KEYS:
        return _bgr_to_rgb(_to_uint8_image(data))
    return np.ascontiguousarray(data, dtype=np.float32)


def _build_raw_payload(raw: dict[str, np.ndarray | None], read_start: float, read_end: float) -> bytes:
    payload = {
        "frames": raw,
        "meta": {
            "local_read_start": read_start,
            "local_read_end": read_end,
        },
    }
    return pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)


# ---------------------------------------------------------------------------
# Async background reader (one thread per sensor)
# ---------------------------------------------------------------------------


class AsyncXenseReader:
    """Continuously reads one XENSE sensor in a background thread.

    The main thread calls ``snapshot()`` to grab the latest frame + metadata
    without blocking on I/O.
    """

    def __init__(
        self,
        name: str,
        serial: str,
        use_gpu: bool = True,
        poll_interval_s: float = 0.0,
        max_frame_age_s: float = 0.25,
        extra_output_types: dict[str, str] | None = None,
        primary_output: str = DEFAULT_PRIMARY_OUTPUT,
    ):
        self._name = name
        self._serial = serial
        self._use_gpu = use_gpu
        self._poll_interval_s = max(0.0, poll_interval_s)
        self._max_frame_age_s = max_frame_age_s
        # {Sensor.OutputType attr name: raw dict key} for whatever the user opted
        # into via ~extra_outputs — empty means "primary output only", the default.
        self._extra_output_types = extra_output_types or {}
        # Key of the always-on output (a PRIMARY_OUTPUTS key) and its SDK attr.
        self._primary_output = primary_output
        self._primary_attr = PRIMARY_OUTPUTS[primary_output][0]

        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._latest_raw: dict[str, np.ndarray | None] | None = None
        self._latest_read_start: float = 0.0
        self._latest_sec: float = 0.0
        self._latest_read_end: float = 0.0
        self._error: BaseException | None = None
        self._thread: threading.Thread | None = None
        self._sensor: Any = None

    # -- connect / disconnect -------------------------------------------------

    def connect(self) -> None:
        from xensesdk import Sensor  # type: ignore[import-untyped]

        max_retries = 8
        retry_delay = 1.5  # seconds between retries
        last_error: Exception | None = None

        for attempt in range(1, max_retries + 1):
            try:
                self._sensor = Sensor.create(cam_id=self._serial, use_gpu=self._use_gpu)
                if self._sensor is None or self._sensor is False:
                    raise RuntimeError(
                        f"Sensor.create returned {self._sensor!r} for {self._name}"
                    )
                break  # success
            except Exception as exc:
                last_error = exc
                if attempt < max_retries:
                    rospy.logwarn(
                        f"XENSE {self._name} connection attempt {attempt}/{max_retries} "
                        f"failed ({exc}), retrying in {retry_delay}s..."
                    )
                    time.sleep(retry_delay)
                else:
                    raise RuntimeError(
                        f"Failed to connect XENSE {self._name} after {max_retries} attempts"
                    ) from last_error

        self._stop.clear()
        self._latest_frame = None
        self._error = None
        self._thread = threading.Thread(
            target=self._reader_loop,
            name=f"xense-{self._name}",
            daemon=True,
        )
        self._thread.start()

        # Block until we have at least one frame (or an error).
        deadline = time.time() + 10.0
        while self._latest_frame is None and self._error is None:
            if time.time() > deadline:
                raise TimeoutError(f"Timed out waiting for first frame from {self._name}")
            time.sleep(0.02)

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._sensor is not None:
            try:
                self._sensor.release()
            except Exception:
                pass
            self._sensor = None

    # -- snapshot -------------------------------------------------------------

    def snapshot(
        self, include_image: bool = True
    ) -> tuple[np.ndarray | None, dict[str, np.ndarray | None] | None, float, float, float]:
        """Return (primary_output_image, raw_frames, read_start, read_end, frame_age_secs).

        ``primary_output_image`` is what _convert_primary produced: the RGB uint8
        rectify frame by default, the RGB uint8 difference frame with
        primary_output=difference, or the float32 mm depth map with primary_output=depth.

        Returns (None, None, 0.0, 0.0, inf) when no frame is available yet (checked via
        the raw dict, not the image, since ``include_image=False`` intentionally skips
        the image copy below while the sensor is otherwise ready).

        ``include_image=False`` skips copying the (700, 400[, 3]) primary frame — the
        caller passes this on frames where it won't publish the preview, to avoid
        paying for an ~1MB memcpy it doesn't need. ``raw`` is returned either way
        since it's cheap and the caller needs it every frame to stay real-time.
        """
        with self._lock:
            if self._error is not None:
                raise self._error
            if self._latest_raw is None:
                return None, None, 0.0, 0.0, float("inf")
            age = time.time() - self._latest_read_end
            if age > self._max_frame_age_s:
                raise RuntimeError(
                    f"XENSE {self._name} frame is stale: age={age:.3f}s "
                    f"> {self._max_frame_age_s:.3f}s"
                )
            return (
                self._latest_frame.copy() if include_image else None,
                self._latest_raw,
                self._latest_read_start,
                self._latest_read_end,
                age,
            )

    # -- internals ------------------------------------------------------------

    def _reader_loop(self) -> None:
        from xensesdk import Sensor  # type: ignore[import-untyped]

        try:
            # [(attr, raw key)] for the extras, in the order they're requested.
            extras = list(self._extra_output_types.items())
            request = [getattr(Sensor.OutputType, self._primary_attr)] + [
                getattr(Sensor.OutputType, attr) for attr, _ in extras
            ]
            while not self._stop.is_set():
                read_start = time.time()
                # ONE selectSensorInfo() call for the primary output and every
                # extra, so everything in one /raw message (and the preview) is
                # from the same sensor frame. Separate calls each take ~10 ms and
                # land on different frames — measured: force read right after
                # depth was always the next frame, ~11 ms later by the sensor's
                # own TimeStamp. One requested output comes back as the bare
                # value, several as a list in request order.
                values = self._sensor.selectSensorInfo(*request)
                values = [values] if len(request) == 1 else list(values)
                if len(values) != len(request):
                    raise RuntimeError(
                        f"XENSE {self._name}: selectSensorInfo returned {len(values)} "
                        f"outputs for {len(request)} requested"
                    )

                # Rectify / difference: the SDK hands back BGR; everything this
                # node publishes is RGB. Depth: float32 mm, passed through.
                img = _convert_primary(self._primary_output, values[0])

                # Independent copy for "raw" — self._latest_frame (below) is a
                # separate reference the preview path copies-on-demand; this one
                # must not alias it, since "raw" always gets published this frame
                # regardless of whether the preview does.
                raw: dict[str, np.ndarray | None] = {self._primary_output: np.array(img, copy=True)}
                for (_, key), value in zip(extras, values[1:]):
                    if value is None:
                        raw[key] = None
                        continue
                    value = np.array(value, copy=True)
                    # Only the colour images (rectify / difference, whichever
                    # isn't the primary) get the BGR->RGB swap.
                    raw[key] = _bgr_to_rgb(value) if key in COLOUR_IMAGE_KEYS else value

                read_end = time.time()

                with self._lock:
                    self._latest_frame = img
                    self._latest_raw = raw
                    self._latest_read_start = read_start
                    self._latest_sec = read_end
                    self._latest_read_end = read_end
        except BaseException as exc:
            with self._lock:
                self._error = exc
            return

        # Respect poll interval (0 = read as fast as possible).
        if self._poll_interval_s > 0:
            self._stop.wait(self._poll_interval_s)


# ---------------------------------------------------------------------------
# ROS node
# ---------------------------------------------------------------------------


def _mock_raw_frame(rng: np.random.Generator, t: float) -> dict[str, np.ndarray]:
    """Synthetic stand-ins for every output type except rectify (the mock loop
    supplies that from its own blob), for --mock testing."""
    return {
        "difference": rng.integers(0, 8, size=(700, 400, 3), dtype=np.uint8),
        "depth": (5.0 + rng.normal(0, 0.05, (700, 400))).astype(np.float32),
        "force": rng.normal(0, 0.1, (35, 20, 3)).astype(np.float32),
        "force_norm": rng.normal(0, 0.1, (35, 20, 3)).astype(np.float32),
        "force_resultant": rng.normal(0, 0.1, (6,)).astype(np.float32),
        "marker2d": rng.normal(0, 0.5, (35, 20, 2)).astype(np.float32),
        "mesh3d": rng.normal(0, 0.1, (35, 20, 3)).astype(np.float32),
        "mesh3d_init": np.zeros((35, 20, 3), dtype=np.float32),
        "mesh3d_flow": rng.normal(0, 0.05, (35, 20, 3)).astype(np.float32),
        "timestamp": np.array([t], dtype=np.float64),
    }


def main() -> None:
    # NOT anonymous: this is a singleton hardware-driver node, and anonymous=True
    # would make rospy.get_name() (and therefore every "~param" lookup below)
    # resolve to a random suffixed name that the launch file's nested <param>
    # tags were never written to — silently discarding every launch-file override.
    rospy.init_node("xense_tactile_publisher")

    # Optional ROS params (can be set in launch file). -----------------------
    use_gpu = rospy.get_param("~use_gpu", True)
    frame_rate = rospy.get_param("~frame_rate", 60)
    poll_interval = rospy.get_param("~async_poll_interval_s", 0.0)
    max_frame_age = rospy.get_param("~async_max_frame_age_s", 0.25)
    mock = rospy.get_param("~mock", False)
    # The always-on output (read every cycle, always in /raw, and the preview
    # topic's content): "rectify" (default), "difference" or "depth". See PRIMARY_OUTPUTS.
    primary_output = str(rospy.get_param("~primary_output", DEFAULT_PRIMARY_OUTPUT)).strip().lower()
    if primary_output not in PRIMARY_OUTPUTS:
        rospy.logfatal(
            f"xense_tactile_publisher: unknown ~primary_output {primary_output!r} "
            f"(valid: {', '.join(PRIMARY_OUTPUTS)})"
        )
        return
    preview_encoding = PRIMARY_OUTPUTS[primary_output][1]
    # /raw publishes every loop iteration (must stay real-time). The
    # /<primary_output> preview is only for human monitoring, so it's fine to lag
    # — only publish it (and only pay for the ~1MB image copy) every Nth iteration.
    preview_every_n_frames = max(1, int(rospy.get_param("~preview_every_n_frames", 6)))
    # Comma-separated subset of RAW_OUTPUT_TYPES' keys to also read from the SDK
    # and bundle into /raw, e.g. "depth,force". Empty (default) = primary output
    # only — nothing else is requested from the SDK, not just "not published".
    # Primary + extras are read in one selectSensorInfo() call (same frame).
    extra_output_types = _parse_extra_outputs(
        str(rospy.get_param("~extra_outputs", "")), primary_output
    )
    if extra_output_types:
        rospy.loginfo(f"Extra /raw outputs enabled: {', '.join(sorted(extra_output_types.values()))}")
    else:
        rospy.loginfo(
            f"No extra /raw outputs enabled — publishing {primary_output} only (see ~extra_outputs)."
        )

    bridge = CvBridge()

    if mock:
        rospy.logwarn("XENSE tactile publisher running in MOCK mode — no real sensors.")
        # Publish synthetic frames so the pipeline can be tested without hardware.
        publishers: dict[str, rospy.Publisher] = {}
        raw_publishers: dict[str, rospy.Publisher] = {}
        for name in TACTILE_SENSORS:
            publishers[name] = rospy.Publisher(
                f"/tactile/{name}/{primary_output}", Image, queue_size=10
            )
            raw_publishers[name] = rospy.Publisher(
                f"/tactile/{name}/{RAW_TOPIC_SUFFIX}", UInt8MultiArray, queue_size=1
            )

        rate = rospy.Rate(frame_rate)
        rng = np.random.default_rng(seed=42)
        ys, xs = np.ogrid[:MOCK_H, :MOCK_W]
        selected_keys = set(extra_output_types.values())
        frame_idx = 0
        while not rospy.is_shutdown():
            stamp = rospy.Time.now()
            read_start = time.time()
            publish_preview = frame_idx % preview_every_n_frames == 0
            for idx, name in enumerate(TACTILE_SENSORS):
                # The primary output is generated every frame — "raw" needs it
                # real-time regardless of whether the preview Image is published
                # this cycle.
                cy = MOCK_H * (0.35 + 0.1 * idx)
                cx = MOCK_W * (0.35 + 0.08 * idx)
                blob = np.exp(-((ys - cy) ** 2 + (xs - cx) ** 2) / (2 * 18**2))
                noisy = np.clip(blob + rng.normal(0, 0.02, (MOCK_H, MOCK_W)), 0, 1)
                gray = _to_uint8(noisy.astype(np.float32))
                # RGB, to match what the real sensor path publishes (grey, so the
                # channel order is academic here — it keeps the encoding honest).
                img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
                # rectify / difference: the grey blob image. depth: same blob as a
                # press MOCK_DEPTH_PEAK_MM deep.
                primary = img if primary_output != "depth" else (
                    (noisy * MOCK_DEPTH_PEAK_MM).astype(np.float32)
                )

                if publish_preview:
                    msg = bridge.cv2_to_imgmsg(primary, encoding=preview_encoding)
                    msg.header.stamp = stamp
                    msg.header.frame_id = f"tactile_{name}"
                    publishers[name].publish(msg)

                mock_frame = _mock_raw_frame(rng, time.time())
                # Only used when rectify is an extra, i.e. not the primary output
                # (selected_keys never contains the primary).
                mock_frame["rectify"] = img
                raw = {
                    k: (_bgr_to_rgb(v) if k in COLOUR_IMAGE_KEYS else v)
                    for k, v in mock_frame.items()
                    if k in selected_keys
                }
                raw[primary_output] = primary
                read_end = time.time()
                raw_msg = UInt8MultiArray()
                raw_msg.data = _build_raw_payload(raw, read_start, read_end)
                raw_publishers[name].publish(raw_msg)
            frame_idx += 1
            rate.sleep()
        return

    # Connect to real sensors. -----------------------------------------------
    sensors: dict[str, AsyncXenseReader] = {}
    publishers: dict[str, rospy.Publisher] = {}
    raw_publishers: dict[str, rospy.Publisher] = {}

    from xensesdk import Sensor  # type: ignore[import-untyped]

    found = set(Sensor.scanSerialNumber())
    rospy.loginfo(f"Found XENSE serial numbers: {sorted(found)}")

    for name, serial in TACTILE_SENSORS.items():
        if serial not in found:
            rospy.logerr(f"XENSE sensor {name} (serial {serial}) NOT found!")
            continue

        publishers[name] = rospy.Publisher(
            f"/tactile/{name}/{primary_output}", Image, queue_size=10
        )
        raw_publishers[name] = rospy.Publisher(
            f"/tactile/{name}/{RAW_TOPIC_SUFFIX}", UInt8MultiArray, queue_size=1
        )

        reader = AsyncXenseReader(
            name=name,
            serial=serial,
            use_gpu=use_gpu,
            poll_interval_s=poll_interval,
            max_frame_age_s=max_frame_age,
            extra_output_types=extra_output_types,
            primary_output=primary_output,
        )
        reader.connect()
        sensors[name] = reader
        raw_contents = primary_output + (
            "+" + "+".join(sorted(extra_output_types.values())) if extra_output_types else ""
        )
        rospy.loginfo(
            f"Connected XENSE sensor: {name} ({serial}) → "
            f"/tactile/{name}/{primary_output} ({primary_output}), "
            f"/tactile/{name}/{RAW_TOPIC_SUFFIX} ({raw_contents})"
        )

    if not sensors:
        rospy.logerr("No XENSE sensors connected! Check serial numbers.")
        return

    rospy.loginfo(
        f"Publishing {RAW_TOPIC_SUFFIX} at ~{frame_rate} Hz, {primary_output} preview "
        f"every {preview_every_n_frames} frames (~{frame_rate / preview_every_n_frames:.1f} Hz), "
        f"on {len(sensors)} sensors."
    )
    rate = rospy.Rate(frame_rate)
    frame_idx = 0

    try:
        while not rospy.is_shutdown():
            stamp = rospy.Time.now()
            publish_preview = frame_idx % preview_every_n_frames == 0
            for name, reader in sensors.items():
                try:
                    img, raw, read_start, read_end, _ = reader.snapshot(include_image=publish_preview)
                    if raw is None:
                        continue  # sensor not ready yet

                    if publish_preview and img is not None:
                        msg = bridge.cv2_to_imgmsg(img, encoding=preview_encoding)
                        msg.header.stamp = stamp
                        msg.header.frame_id = f"tactile_{name}"
                        publishers[name].publish(msg)

                    raw_msg = UInt8MultiArray()
                    raw_msg.data = _build_raw_payload(raw, read_start, read_end)
                    raw_publishers[name].publish(raw_msg)
                except Exception as exc:
                    rospy.logerr_throttle(5.0, f"Error reading {name}: {exc}")
            frame_idx += 1
            rate.sleep()
    finally:
        for reader in sensors.values():
            reader.close()
        rospy.loginfo("XENSE tactile publisher shut down.")


if __name__ == "__main__":
    main()
