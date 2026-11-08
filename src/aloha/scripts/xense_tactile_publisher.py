#!/home/robot/miniforge3/envs/lerobot/bin/python
"""
XENSE Tactile Publisher for ROS (4-arm ALOHA with tactile).

Reads 4 XENSE G1-WS tactile sensors and publishes, per sensor:

  1. The full, canonical data for each frame, bundled into one pickled payload:

       /tactile/left_gripper_inner/raw
       /tactile/left_gripper_outer/raw
       /tactile/right_gripper_inner/raw
       /tactile/right_gripper_outer/raw

     std_msgs/UInt8MultiArray whose ``data`` is
     ``pickle.dumps({"frames": {...}, "meta": {...}})``. This is the one topic to
     subscribe to for anything downstream (recording, training, analysis).

  2. A throttled preview of the primary output, for live monitoring only
     (``~preview_every_n_frames``): ``/tactile/<name>/<primary_output>``,
     sensor_msgs/Image (rgb8 for rectify/difference, 32FC1 mm for depth).

and, for the whole node:

  3. ``/tactile/runtime_config`` (latched, UInt8MultiArray, pickled dict): every
     sensor's SDK runtime config exported right after the last zeroing, plus the
     SDK version and this node's settings. Recorders store it with each episode.
  4. ``/tactile/calibrate`` (std_srvs/Trigger): re-zero every sensor. Nothing may
     touch the gels while it runs. The readers pause around it, so subscribers see
     no frames for that moment.

WHEN /raw IS PUBLISHED (~publish_mode)
  on_new_frame (default): each sensor's own reader thread pickles and publishes a
     frame right after the SDK returns it -- no timer, no waiting, no duplicate
     frames -- at most ~max_rate_hz (default 60, 0 = no cap). Frames beyond the cap
     are dropped, never delayed (token bucket, so a 59 Hz sensor still sends ~59 Hz
     and a 100 Hz one ~60 Hz). Measured 2026-11-08 with the old timer design: the
     SDK-to-subscriber delay was ~11-12 ms (timer hold) instead of ~3 ms, and with
     several native-resolution outputs the main thread's pickling cut new frames
     from ~60/s to ~26/s.
  timer: the original design, kept as a fallback. Readers store their newest frame
     and the main thread publishes the latest one of each sensor every 1/frame_rate
     s, re-sending the same frame when the sensor is slower than the timer.

FRAMES
  ``frames`` always has the primary output (``~primary_output``: rectify
  (default), difference or depth) plus whatever ``~extra_outputs`` selects
  (comma-separated keys, e.g. "depth,force"; see RAW_OUTPUT_TYPES). Nothing else
  is requested from the SDK. The primary output and every extra come from ONE
  ``selectSensorInfo()`` call, so everything in one message is the same sensor
  frame. Everything is at native XENSE resolution (rectify/difference
  (700, 400, 3), depth (700, 400) float32 mm): no resize.

  With ``~image_codec: raw`` (default) rectify and difference are uint8 RGB arrays
  (the SDK returns BGR; swapped here), every other output is what the SDK
  returned, including the 3-channel ones (force, force_norm, mesh3d*), whose
  channels are x/y/z rather than colours.
  With ``~image_codec: jpeg`` rectify and difference are instead 1-D uint8 arrays
  holding a standard true-colour JPEG (``~jpeg_quality``, default 90; decode with
  cv2.imdecode -> BGR), and depth is zlib-compressed float32 bytes (lossless);
  ``meta["encoding"]`` says which keys are encoded and how to restore them. The
  other outputs are unchanged.

META
  local_read_start / local_read_end   host time.time() around the SDK call; the
                                      end is when the frame was handed over, so
                                      ``now - local_read_end`` is the data's age
  timestamp, timestamp_source         the SDK's own TimeStamp for this frame, read in
                                      the same call ("xense"); measured to equal the
                                      moment the call returns. "local_read_end" if
                                      the SDK has no TimeStamp.
  seq                                 per-sensor count of frames read from the SDK
                                      (gaps = frames dropped by the rate cap)
  image_codec, encoding               "raw" / "jpeg", and {key: {...}} for encoded keys

Missing sensors are skipped with an error log; the others publish as usual.
"""

from __future__ import annotations

import io
import pickle
import tarfile
import tempfile
import threading
import time
import zlib
from collections import deque
from typing import Any

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import UInt8MultiArray
from std_srvs.srv import Trigger, TriggerResponse

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

PUBLISH_MODES: tuple[str, ...] = ("on_new_frame", "timer")
IMAGE_CODECS: tuple[str, ...] = ("raw", "jpeg")

# Token-bucket capacity for ~max_rate_hz: up to this many frames of credit, so the
# cap keeps the fractional credit of a fast sensor (100 Hz -> 60 Hz, not 50 Hz)
# without allowing more than a 2-frame burst after a pause.
RATE_BUCKET_CAPACITY: float = 2.0

RUNTIME_CONFIG_TOPIC: str = "/tactile/runtime_config"
CALIBRATE_SERVICE: str = "/tactile/calibrate"

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
# the primary output (see PRIMARY_OUTPUTS) is always in "raw" already, so
# _parse_extra_outputs drops it from the extras.
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
#   Sensor.OutputType.TimeStamp       -> the SDK's time for this frame, seconds
#                                        (always in meta["timestamp"] as well)
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
# swap on the way out (or, with image_codec=jpeg, are JPEG-encoded). Every other
# 3-channel output (force, force_norm, mesh3d, ...) carries x/y/z components
# rather than colours and must be handed on exactly as the SDK returned it.
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
    """Swap channel order on a 3-channel image (always a new array); pass anything else through.

    The XENSE SDK returns Rectify and Difference as BGR, but everything this
    node publishes raw is RGB, so both go through here on the way out.

    Only call this on actual colour images: Force/ForceNorm/Mesh3D are 3-channel
    too, but those channels are x/y/z components and swapping them corrupts the
    data. COLOUR_IMAGE_KEYS is the list of keys this is valid for.
    """
    if arr.ndim == 3 and arr.shape[2] == 3:
        return np.ascontiguousarray(arr[..., ::-1])
    return arr


def _encode_bundle(native: dict[str, Any], codec: str, jpeg_quality: int) -> tuple[dict[str, Any], dict[str, dict]]:
    """Turn the SDK's outputs for one frame into the published ``frames`` dict.

    ``native`` holds the values exactly as the SDK returned them (colour images
    BGR). Returns (frames, encoding); ``encoding`` is empty for codec "raw".
    Every returned array is independent of the SDK's buffers.
    """
    frames: dict[str, Any] = {}
    encoding: dict[str, dict] = {}
    for key, value in native.items():
        if value is None:
            frames[key] = None
        elif key in COLOUR_IMAGE_KEYS:
            bgr = _to_uint8_image(value)
            if codec == "jpeg":
                ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
                if not ok:
                    raise RuntimeError(f"JPEG encode failed for {key}")
                frames[key] = buf.reshape(-1)
                encoding[key] = {"codec": "jpeg", "quality": int(jpeg_quality), "shape": list(bgr.shape)}
            else:
                frames[key] = _bgr_to_rgb(bgr)
        elif key == "depth":
            depth = np.ascontiguousarray(value, dtype=np.float32)
            if codec == "jpeg":
                frames[key] = np.frombuffer(zlib.compress(depth.tobytes(), 1), dtype=np.uint8)
                encoding[key] = {"codec": "zlib", "dtype": "float32", "shape": list(depth.shape)}
            else:
                frames[key] = depth.copy()
        else:
            frames[key] = np.array(value, copy=True)
    return frames, encoding


def _preview_image(primary_output: str, native_primary: Any) -> np.ndarray:
    """The preview Image's array: RGB uint8 for colour outputs, float32 mm for depth."""
    if primary_output in COLOUR_IMAGE_KEYS:
        return _bgr_to_rgb(_to_uint8_image(native_primary))
    return np.ascontiguousarray(native_primary, dtype=np.float32)


def _build_raw_payload(frames: dict[str, Any], meta: dict[str, Any]) -> bytes:
    return pickle.dumps({"frames": frames, "meta": meta}, protocol=pickle.HIGHEST_PROTOCOL)


def _sdk_version() -> str:
    try:
        from importlib.metadata import version

        return version("xensesdk")
    except Exception:
        return ""


def _export_runtime_config(sensor: Any) -> bytes:
    """The sensor's SDK runtime config (what Sensor.createSolver needs), as tar.gz bytes.

    xensesdk 1.7.0 writes one file, runtime_<serial>, into the given directory;
    it is archived as "./runtime_<serial>", the same layout tactile_acuity stores,
    so its offline tools (verify_xense_offline_recompute.py) can read it.
    """
    with tempfile.TemporaryDirectory() as tmp:
        sensor.exportRuntimeConfig(tmp)
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            tar.add(tmp, arcname=".")
        return buf.getvalue()


class RateLimiter:
    """Token bucket: allow() at most ``rate_hz`` times per second on average, never waiting."""

    def __init__(self, rate_hz: float):
        self.rate_hz = float(rate_hz)
        self._tokens = 1.0
        self._last: float | None = None

    def allow(self, now: float) -> bool:
        if self.rate_hz <= 0:
            return True
        if self._last is not None:
            self._tokens = min(RATE_BUCKET_CAPACITY, self._tokens + (now - self._last) * self.rate_hz)
        self._last = now
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        return False


class PauseGate:
    """Lets the calibrate service stop every reader between two SDK calls.

    Readers call wait_if_paused() at the top of each loop; the service calls
    pause() (which returns once every reader is parked), uses the sensor handles,
    then resume(). The hot path only checks an Event.
    """

    def __init__(self):
        self._pause = threading.Event()
        self._resume = threading.Event()
        self._resume.set()
        self._parked: dict[str, threading.Event] = {}

    def register(self, name: str) -> None:
        self._parked[name] = threading.Event()

    def wait_if_paused(self, name: str, stop: threading.Event) -> None:
        if not self._pause.is_set():
            return
        self._parked[name].set()
        while not self._resume.wait(0.1):
            if stop.is_set():
                break
        self._parked[name].clear()

    def pause(self, names: list[str], timeout_s: float = 2.0) -> list[str]:
        """Park the named readers; returns the ones that did not park in time."""
        self._resume.clear()
        self._pause.set()
        return [n for n in names if not self._parked[n].wait(timeout_s)]

    def resume(self) -> None:
        self._pause.clear()
        self._resume.set()


# ---------------------------------------------------------------------------
# Background reader (one thread per sensor)
# ---------------------------------------------------------------------------


class AsyncXenseReader:
    """Continuously reads one XENSE sensor in a background thread.

    publish_mode "on_new_frame": the thread itself publishes /raw (and the
    preview) for every new frame the rate limiter lets through.
    publish_mode "timer": the thread only keeps the newest frame; the main thread
    calls ``snapshot()`` and publishes it.
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
        publish_mode: str = "on_new_frame",
        max_rate_hz: float = 0.0,
        image_codec: str = "raw",
        jpeg_quality: int = 90,
        preview_every_n_frames: int = 6,
        raw_publisher: rospy.Publisher | None = None,
        preview_publisher: rospy.Publisher | None = None,
        gate: PauseGate | None = None,
    ):
        self.name = name
        self.serial = serial
        self._use_gpu = use_gpu
        self._poll_interval_s = max(0.0, poll_interval_s)
        self._max_frame_age_s = max_frame_age_s
        # {Sensor.OutputType attr name: raw dict key} for whatever the user opted
        # into via ~extra_outputs — empty means "primary output only", the default.
        self._extra_output_types = extra_output_types or {}
        # Key of the always-on output (a PRIMARY_OUTPUTS key) and its SDK attr.
        self._primary_output = primary_output
        self._primary_attr, self._preview_encoding = PRIMARY_OUTPUTS[primary_output]
        self._publish_mode = publish_mode
        self._limiter = RateLimiter(max_rate_hz)
        self._codec = image_codec
        self._jpeg_quality = int(jpeg_quality)
        self._preview_every = max(1, int(preview_every_n_frames))
        self._raw_pub = raw_publisher
        self._preview_pub = preview_publisher
        self._gate = gate
        if gate is not None:
            gate.register(name)
        self._bridge = CvBridge()

        self._stop = threading.Event()
        self._first_frame = threading.Event()
        self._lock = threading.Lock()
        # timer mode: the newest frame, its meta and the source of its preview
        self._latest_frames: dict[str, Any] | None = None
        self._latest_meta: dict[str, Any] | None = None
        self._latest_preview_src: Any = None
        self._latest_read_end: float = 0.0
        self._error: BaseException | None = None
        self._thread: threading.Thread | None = None
        self.sensor: Any = None
        self._seq = 0
        self._sent = 0
        # heartbeat counters, reset by take_stats()
        self._stats_lock = threading.Lock()
        self._n_read = self._n_sent = self._n_dropped = 0
        self._read_ms: deque = deque(maxlen=1000)
        self._send_ms: deque = deque(maxlen=1000)

    # -- connect / start / close ------------------------------------------------

    def connect(self) -> None:
        """Open the sensor (no reading yet; see start())."""
        from xensesdk import Sensor  # type: ignore[import-untyped]

        max_retries = 8
        retry_delay = 1.5  # seconds between retries
        last_error: Exception | None = None

        for attempt in range(1, max_retries + 1):
            try:
                self.sensor = Sensor.create(cam_id=self.serial, use_gpu=self._use_gpu)
                if self.sensor is None or self.sensor is False:
                    raise RuntimeError(
                        f"Sensor.create returned {self.sensor!r} for {self.name}"
                    )
                return
            except Exception as exc:
                last_error = exc
                if attempt < max_retries:
                    rospy.logwarn(
                        f"XENSE {self.name} connection attempt {attempt}/{max_retries} "
                        f"failed ({exc}), retrying in {retry_delay}s..."
                    )
                    time.sleep(retry_delay)
        raise RuntimeError(
            f"Failed to connect XENSE {self.name} after {max_retries} attempts"
        ) from last_error

    def start(self, timeout_s: float = 10.0) -> None:
        """Start the reader thread and wait for its first frame.

        A reader that fails on its first frame is logged and left stopped while the
        other sensors carry on (as before); no first frame at all within the
        timeout raises.
        """
        self._stop.clear()
        self._error = None
        self._thread = threading.Thread(target=self._reader_loop, name=f"xense-{self.name}", daemon=True)
        self._thread.start()
        if not self._first_frame.wait(timeout_s):
            raise TimeoutError(f"Timed out waiting for first frame from {self.name}")
        if self._error is not None:
            rospy.logerr(f"XENSE {self.name} reader failed on its first frame: {self._error!r}")

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and self._error is None

    @property
    def error(self) -> BaseException | None:
        return self._error

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self.sensor is not None:
            try:
                self.sensor.release()
            except Exception:
                pass
            self.sensor = None

    # -- timer mode ---------------------------------------------------------------

    def snapshot(self, include_image: bool = True) -> tuple[np.ndarray | None, dict | None, dict | None, float]:
        """timer mode: return (preview_image, frames, meta, frame_age_secs) of the newest frame.

        Returns (None, None, None, inf) when no frame is available yet. Raises if the
        reader failed, or if its newest frame is older than ~async_max_frame_age_s, so
        a stalled sensor is never re-sent as if it were live.
        """
        with self._lock:
            if self._error is not None:
                raise self._error
            if self._latest_frames is None:
                return None, None, None, float("inf")
            age = time.time() - self._latest_read_end
            if age > self._max_frame_age_s:
                raise RuntimeError(
                    f"XENSE {self.name} frame is stale: age={age:.3f}s > {self._max_frame_age_s:.3f}s"
                )
            frames, meta, src = self._latest_frames, self._latest_meta, self._latest_preview_src
        image = _preview_image(self._primary_output, src) if include_image else None
        return image, frames, meta, age

    # -- heartbeat ----------------------------------------------------------------

    def take_stats(self) -> dict[str, float]:
        with self._stats_lock:
            stats = {
                "read": self._n_read, "sent": self._n_sent, "dropped": self._n_dropped,
                "read_ms": float(np.median(self._read_ms)) if self._read_ms else float("nan"),
                "send_ms": float(np.median(self._send_ms)) if self._send_ms else float("nan"),
            }
            self._n_read = self._n_sent = self._n_dropped = 0
            self._read_ms.clear()
            self._send_ms.clear()
        return stats

    # -- internals ------------------------------------------------------------

    def _meta(self, read_start: float, read_end: float, timestamp: float, source: str,
              encoding: dict[str, dict]) -> dict[str, Any]:
        return {
            "local_read_start": read_start,
            "local_read_end": read_end,
            "timestamp": timestamp,
            "timestamp_source": source,
            "seq": self._seq,
            "image_codec": self._codec,
            "encoding": encoding,
        }

    def _reader_loop(self) -> None:
        from xensesdk import Sensor  # type: ignore[import-untyped]

        try:
            # [(attr, raw key)] for the extras, in the order they're requested.
            extras = list(self._extra_output_types.items())
            keys = [self._primary_output] + [key for _, key in extras]
            request = [getattr(Sensor.OutputType, self._primary_attr)] + [
                getattr(Sensor.OutputType, attr) for attr, _ in extras
            ]
            # The SDK's TimeStamp rides in the same call for meta["timestamp"];
            # reuse it when ~extra_outputs already asks for it.
            ts_enum = getattr(Sensor.OutputType, "TimeStamp", None)
            if "timestamp" in keys:
                ts_index = keys.index("timestamp")
            elif ts_enum is not None:
                request.append(ts_enum)
                ts_index = len(request) - 1
            else:
                ts_index = None

            while not self._stop.is_set():
                if self._gate is not None:
                    self._gate.wait_if_paused(self.name, self._stop)
                    if self._stop.is_set():
                        break
                read_start = time.time()
                # ONE selectSensorInfo() call for the primary output, every extra and
                # the TimeStamp, so everything in one /raw message (and the preview)
                # is from the same sensor frame. Separate calls each take ~10 ms and
                # land on different frames. One requested output comes back as the
                # bare value, several as a list in request order.
                values = self.sensor.selectSensorInfo(*request)
                read_end = time.time()
                values = [values] if len(request) == 1 else list(values)
                if len(values) != len(request):
                    raise RuntimeError(
                        f"XENSE {self.name}: selectSensorInfo returned {len(values)} "
                        f"outputs for {len(request)} requested"
                    )
                native = dict(zip(keys, values[:len(keys)]))
                ts_value = values[ts_index] if ts_index is not None else None
                if ts_value is not None:
                    timestamp, ts_source = float(np.asarray(ts_value, dtype=np.float64).ravel()[0]), "xense"
                else:
                    timestamp, ts_source = read_end, "local_read_end"
                self._seq += 1
                with self._stats_lock:
                    self._n_read += 1
                    self._read_ms.append((read_end - read_start) * 1000.0)

                if self._publish_mode == "timer":
                    frames, encoding = _encode_bundle(native, self._codec, self._jpeg_quality)
                    meta = self._meta(read_start, read_end, timestamp, ts_source, encoding)
                    with self._lock:
                        self._latest_frames = frames
                        self._latest_meta = meta
                        self._latest_preview_src = native[self._primary_output]
                        self._latest_read_end = read_end
                elif self._limiter.allow(read_end):
                    self._publish(native, read_start, read_end, timestamp, ts_source)
                else:
                    with self._stats_lock:
                        self._n_dropped += 1
                self._first_frame.set()

                # Respect poll interval (0 = read as fast as possible).
                if self._poll_interval_s > 0:
                    self._stop.wait(self._poll_interval_s)
        except BaseException as exc:
            with self._lock:
                self._error = exc
            self._first_frame.set()

    def _publish(self, native: dict[str, Any], read_start: float, read_end: float,
                 timestamp: float, ts_source: str) -> None:
        """on_new_frame mode: encode, pickle and publish this frame from the reader thread."""
        t0 = time.time()
        frames, encoding = _encode_bundle(native, self._codec, self._jpeg_quality)
        msg = UInt8MultiArray()
        msg.data = _build_raw_payload(frames, self._meta(read_start, read_end, timestamp, ts_source, encoding))
        self._raw_pub.publish(msg)
        self._sent += 1
        if self._preview_pub is not None and (self._sent - 1) % self._preview_every == 0:
            preview = self._bridge.cv2_to_imgmsg(
                _preview_image(self._primary_output, native[self._primary_output]), encoding=self._preview_encoding
            )
            preview.header.stamp = rospy.Time.from_sec(read_end)
            preview.header.frame_id = f"tactile_{self.name}"
            self._preview_pub.publish(preview)
        with self._stats_lock:
            self._n_sent += 1
            self._send_ms.append((time.time() - t0) * 1000.0)


# ---------------------------------------------------------------------------
# Zeroing + runtime config
# ---------------------------------------------------------------------------


class RuntimeConfigManager:
    """Zeroes the sensors and publishes their runtime configs on the latched topic."""

    def __init__(self, readers: list[AsyncXenseReader], gate: PauseGate, settings: dict[str, Any],
                 mock: bool = False):
        self._readers = readers
        self._gate = gate
        self._settings = settings
        self._mock = mock
        self._sdk_version = "mock" if mock else _sdk_version()
        self._lock = threading.Lock()
        self._pub = rospy.Publisher(RUNTIME_CONFIG_TOPIC, UInt8MultiArray, queue_size=1, latch=True)

    def zero_and_publish(self, calibrate: bool) -> tuple[bool, str]:
        """Optionally zero every sensor, export every runtime config, publish them.

        Readers that are running are parked around it, so the SDK handles are never
        used from two threads at once. Returns (success, message).
        """
        with self._lock:
            calibrated: dict[str, bool] = {}
            configs: dict[str, bytes] = {}
            errors: dict[str, str] = {}
            running = [r.name for r in self._readers if r.running]
            late = self._gate.pause(running) if running else []
            if late:
                # A reader still inside selectSensorInfo() must not have its handle
                # used from this thread too: give up rather than race the SDK.
                self._gate.resume()
                message = f"readers {late} did not pause within 2 s; nothing was zeroed"
                rospy.logerr(f"xense_tactile_publisher: {message}")
                return False, message
            try:
                for reader in self._readers:
                    if self._mock:
                        calibrated[reader.name] = calibrate
                        continue
                    if calibrate:
                        try:
                            reader.sensor.calibrateSensor()
                            calibrated[reader.name] = True
                        except Exception as exc:
                            calibrated[reader.name] = False
                            errors[reader.name] = f"calibrateSensor: {exc!r}"
                    try:
                        configs[reader.name] = _export_runtime_config(reader.sensor)
                    except Exception as exc:
                        errors[reader.name] = errors.get(reader.name, "") + f" exportRuntimeConfig: {exc!r}"
            finally:
                self._gate.resume()
            now = time.time()
            blob = {
                "runtime_configs": configs,
                "serials": {r.name: r.serial for r in self._readers},
                "sdk_version": self._sdk_version,
                "calibrated": calibrated,
                "calibrated_at": now if calibrate else None,
                "exported_at": now,
                "settings": dict(self._settings),
                "mock": self._mock,
            }
            msg = UInt8MultiArray()
            msg.data = pickle.dumps(blob, protocol=pickle.HIGHEST_PROTOCOL)
            self._pub.publish(msg)
            ok = not errors and (not calibrate or all(calibrated.values()))
            what = "zeroed and exported" if calibrate else "exported"
            message = f"{what} {len(configs) if not self._mock else len(self._readers)} sensors"
            if errors:
                message += f"; errors: {errors}"
            (rospy.loginfo if ok else rospy.logerr)(f"xense_tactile_publisher: runtime config {message}")
            return ok, message

    def handle_calibrate(self, _req: Any) -> TriggerResponse:
        ok, message = self.zero_and_publish(calibrate=True)
        return TriggerResponse(success=ok, message=message)


# ---------------------------------------------------------------------------
# ROS node
# ---------------------------------------------------------------------------


def _mock_raw_frame(rng: np.random.Generator, t: float) -> dict[str, np.ndarray]:
    """Synthetic stand-ins for every output type except rectify (the mock loop
    supplies that from its own blob), for --mock testing. Colour images are BGR,
    like the SDK."""
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


def _run_mock(frame_rate: float, primary_output: str, extra_output_types: dict[str, str],
              preview_every_n_frames: int, image_codec: str, jpeg_quality: int, settings: dict[str, Any]) -> None:
    """Publish synthetic frames at frame_rate so the pipeline can be tested without hardware."""
    rospy.logwarn("XENSE tactile publisher running in MOCK mode — no real sensors.")
    bridge = CvBridge()
    preview_encoding = PRIMARY_OUTPUTS[primary_output][1]
    previews = {n: rospy.Publisher(f"/tactile/{n}/{primary_output}", Image, queue_size=10) for n in TACTILE_SENSORS}
    raws = {n: rospy.Publisher(f"/tactile/{n}/{RAW_TOPIC_SUFFIX}", UInt8MultiArray, queue_size=1) for n in TACTILE_SENSORS}

    class _MockReader:  # what RuntimeConfigManager needs from a reader
        running = False

        def __init__(self, name, serial):
            self.name, self.serial = name, serial

    gate = PauseGate()
    manager = RuntimeConfigManager([_MockReader(n, s) for n, s in TACTILE_SENSORS.items()], gate, settings, mock=True)
    manager.zero_and_publish(calibrate=bool(settings.get("calibrate_on_start")))
    rospy.Service(CALIBRATE_SERVICE, Trigger, manager.handle_calibrate)

    rate = rospy.Rate(frame_rate)
    rng = np.random.default_rng(seed=42)
    ys, xs = np.ogrid[:MOCK_H, :MOCK_W]
    selected_keys = set(extra_output_types.values())
    seq = 0
    while not rospy.is_shutdown():
        seq += 1
        for idx, name in enumerate(TACTILE_SENSORS):
            read_start = time.time()
            cy = MOCK_H * (0.35 + 0.1 * idx)
            cx = MOCK_W * (0.35 + 0.08 * idx)
            blob = np.exp(-((ys - cy) ** 2 + (xs - cx) ** 2) / (2 * 18**2))
            noisy = np.clip(blob + rng.normal(0, 0.02, (MOCK_H, MOCK_W)), 0, 1)
            # Grey, so BGR == RGB; stands in for the SDK's native BGR image.
            img = cv2.cvtColor(_to_uint8(noisy.astype(np.float32)), cv2.COLOR_GRAY2BGR)
            # rectify / difference: the grey blob image. depth: same blob as a
            # press MOCK_DEPTH_PEAK_MM deep.
            primary = img if primary_output != "depth" else (noisy * MOCK_DEPTH_PEAK_MM).astype(np.float32)
            mock_frame = _mock_raw_frame(rng, time.time())
            mock_frame["rectify"] = img
            native = {k: v for k, v in mock_frame.items() if k in selected_keys}
            native[primary_output] = primary
            read_end = time.time()
            frames, encoding = _encode_bundle(native, image_codec, jpeg_quality)
            meta = {"local_read_start": read_start, "local_read_end": read_end, "timestamp": read_end,
                    "timestamp_source": "mock", "seq": seq, "image_codec": image_codec, "encoding": encoding}
            msg = UInt8MultiArray()
            msg.data = _build_raw_payload(frames, meta)
            raws[name].publish(msg)
            if (seq - 1) % preview_every_n_frames == 0:
                preview = bridge.cv2_to_imgmsg(_preview_image(primary_output, primary), encoding=preview_encoding)
                preview.header.stamp = rospy.Time.from_sec(read_end)
                preview.header.frame_id = f"tactile_{name}"
                previews[name].publish(preview)
        rate.sleep()


def main() -> None:
    # NOT anonymous: this is a singleton hardware-driver node, and anonymous=True
    # would make rospy.get_name() (and therefore every "~param" lookup below)
    # resolve to a random suffixed name that the launch file's nested <param>
    # tags were never written to — silently discarding every launch-file override.
    rospy.init_node("xense_tactile_publisher")

    # Optional ROS params (can be set in launch file). -----------------------
    use_gpu = rospy.get_param("~use_gpu", True)
    # timer mode: publish rate; mock mode: synthetic frame rate.
    frame_rate = rospy.get_param("~frame_rate", 60)
    poll_interval = rospy.get_param("~async_poll_interval_s", 0.0)
    max_frame_age = rospy.get_param("~async_max_frame_age_s", 0.25)
    mock = rospy.get_param("~mock", False)
    # Heartbeat log (per sensor: SDK read time, new frames/s, sent/s) every this many seconds; 0 = off.
    stats_period = float(rospy.get_param("~stats_log_period_s", 60.0))

    publish_mode = str(rospy.get_param("~publish_mode", "on_new_frame")).strip().lower()
    image_codec = str(rospy.get_param("~image_codec", "raw")).strip().lower()
    primary_output = str(rospy.get_param("~primary_output", DEFAULT_PRIMARY_OUTPUT)).strip().lower()
    for name, value, valid in (("publish_mode", publish_mode, PUBLISH_MODES),
                               ("image_codec", image_codec, IMAGE_CODECS),
                               ("primary_output", primary_output, tuple(PRIMARY_OUTPUTS))):
        if value not in valid:
            rospy.logfatal(f"xense_tactile_publisher: unknown ~{name} {value!r} (valid: {', '.join(valid)})")
            return
    # on_new_frame mode: at most this many /raw messages per second per sensor
    # (0 = every frame the SDK delivers).
    max_rate_hz = float(rospy.get_param("~max_rate_hz", 60.0))
    jpeg_quality = int(min(100, max(1, int(rospy.get_param("~jpeg_quality", 90)))))
    # Zero every sensor before the first frame is published. Nothing may touch the gels.
    calibrate_on_start = bool(rospy.get_param("~calibrate_on_start", True))
    # The /<primary_output> preview is only for human monitoring, so it's fine to lag
    # — publish it once every N /raw messages.
    preview_every_n_frames = max(1, int(rospy.get_param("~preview_every_n_frames", 6)))
    # Comma-separated subset of RAW_OUTPUT_TYPES' keys to also read from the SDK
    # and bundle into /raw, e.g. "depth,force". Empty (default) = primary output
    # only — nothing else is requested from the SDK, not just "not published".
    extra_output_types = _parse_extra_outputs(
        str(rospy.get_param("~extra_outputs", "")), primary_output
    )
    if extra_output_types:
        rospy.loginfo(f"Extra /raw outputs enabled: {', '.join(sorted(extra_output_types.values()))}")
    else:
        rospy.loginfo(
            f"No extra /raw outputs enabled — publishing {primary_output} only (see ~extra_outputs)."
        )
    settings = {
        "primary_output": primary_output,
        "extra_outputs": sorted(extra_output_types.values()),
        "publish_mode": publish_mode,
        "max_rate_hz": max_rate_hz,
        "frame_rate": frame_rate,
        "image_codec": image_codec,
        "jpeg_quality": jpeg_quality,
        "calibrate_on_start": calibrate_on_start,
    }

    if mock:
        _run_mock(frame_rate, primary_output, extra_output_types, preview_every_n_frames,
                  image_codec, jpeg_quality, settings)
        return

    # Connect to real sensors. -----------------------------------------------
    from xensesdk import Sensor  # type: ignore[import-untyped]

    found = set(Sensor.scanSerialNumber())
    rospy.loginfo(f"Found XENSE serial numbers: {sorted(found)}")

    gate = PauseGate()
    readers: list[AsyncXenseReader] = []
    preview_pubs: dict[str, rospy.Publisher] = {}
    raw_pubs: dict[str, rospy.Publisher] = {}
    for name, serial in TACTILE_SENSORS.items():
        if serial not in found:
            rospy.logerr(f"XENSE sensor {name} (serial {serial}) NOT found!")
            continue

        preview_pubs[name] = rospy.Publisher(f"/tactile/{name}/{primary_output}", Image, queue_size=10)
        raw_pubs[name] = rospy.Publisher(f"/tactile/{name}/{RAW_TOPIC_SUFFIX}", UInt8MultiArray, queue_size=1)
        reader = AsyncXenseReader(
            name=name,
            serial=serial,
            use_gpu=use_gpu,
            poll_interval_s=poll_interval,
            max_frame_age_s=max_frame_age,
            extra_output_types=extra_output_types,
            primary_output=primary_output,
            publish_mode=publish_mode,
            max_rate_hz=max_rate_hz,
            image_codec=image_codec,
            jpeg_quality=jpeg_quality,
            preview_every_n_frames=preview_every_n_frames,
            raw_publisher=raw_pubs[name],
            preview_publisher=preview_pubs[name],
            gate=gate,
        )
        reader.connect()
        readers.append(reader)
        raw_contents = primary_output + (
            "+" + "+".join(sorted(extra_output_types.values())) if extra_output_types else ""
        )
        rospy.loginfo(
            f"Connected XENSE sensor: {name} ({serial}) → "
            f"/tactile/{name}/{primary_output} ({primary_output}), "
            f"/tactile/{name}/{RAW_TOPIC_SUFFIX} ({raw_contents})"
        )

    if not readers:
        rospy.logerr("No XENSE sensors connected! Check serial numbers.")
        return

    try:
        # Zero (no reader is running yet) and publish the runtime configs, then start.
        manager = RuntimeConfigManager(readers, gate, settings)
        if calibrate_on_start:
            rospy.loginfo("Zeroing XENSE sensors — nothing may touch the gels now.")
        manager.zero_and_publish(calibrate=calibrate_on_start)
        for reader in readers:
            reader.start()
        rospy.Service(CALIBRATE_SERVICE, Trigger, manager.handle_calibrate)

        if publish_mode == "timer":
            rospy.loginfo(
                f"Publishing {RAW_TOPIC_SUFFIX} at ~{frame_rate} Hz (timer mode), {primary_output} preview "
                f"every {preview_every_n_frames} frames (~{frame_rate / preview_every_n_frames:.1f} Hz), "
                f"image_codec={image_codec}, on {len(readers)} sensors."
            )
        else:
            cap = f"at most {max_rate_hz:g} Hz" if max_rate_hz > 0 else "no rate cap"
            rospy.loginfo(
                f"Publishing {RAW_TOPIC_SUFFIX} on every new frame ({cap}), {primary_output} preview every "
                f"{preview_every_n_frames} sent frames, image_codec={image_codec}, on {len(readers)} sensors."
            )
        _main_loop(readers, publish_mode, frame_rate, preview_every_n_frames, raw_pubs, preview_pubs, stats_period)
    finally:
        for reader in readers:
            reader.close()
        rospy.loginfo("XENSE tactile publisher shut down.")


def _main_loop(readers: list[AsyncXenseReader], publish_mode: str, frame_rate: float, preview_every_n_frames: int,
               raw_pubs: dict[str, rospy.Publisher], preview_pubs: dict[str, rospy.Publisher],
               stats_period: float) -> None:
    """timer mode: publish the newest frame of each sensor every 1/frame_rate s.
    on_new_frame mode: the readers publish; this only reports. Both log a heartbeat."""
    bridge = CvBridge()
    period = 1.0 / frame_rate if publish_mode == "timer" else 0.2
    rate = rospy.Rate(1.0 / period)
    frame_idx = 0
    last_stats = time.time()
    reported: set[str] = set()
    while not rospy.is_shutdown():
        if publish_mode == "timer":
            stamp = rospy.Time.now()
            publish_preview = frame_idx % preview_every_n_frames == 0
            for reader in readers:
                try:
                    img, frames, meta, _ = reader.snapshot(include_image=publish_preview)
                    if frames is None:
                        continue  # sensor not ready yet
                    if publish_preview and img is not None:
                        msg = bridge.cv2_to_imgmsg(img, encoding=_preview_encoding_of(img))
                        msg.header.stamp = stamp
                        msg.header.frame_id = f"tactile_{reader.name}"
                        preview_pubs[reader.name].publish(msg)
                    raw_msg = UInt8MultiArray()
                    raw_msg.data = _build_raw_payload(frames, meta)
                    raw_pubs[reader.name].publish(raw_msg)
                except Exception as exc:
                    rospy.logerr_throttle(5.0, f"Error reading {reader.name}: {exc}")
            frame_idx += 1
        else:
            for reader in readers:
                if reader.error is not None and reader.name not in reported:
                    reported.add(reader.name)
                    rospy.logerr(f"XENSE {reader.name} reader stopped: {reader.error!r}")

        now = time.time()
        if stats_period > 0 and now - last_stats >= stats_period:
            elapsed = now - last_stats
            last_stats = now
            parts = []
            for reader in readers:
                s = reader.take_stats()
                part = f"{reader.name} read {s['read_ms']:.1f} ms, new {s['read'] / elapsed:.1f}/s"
                if publish_mode != "timer":
                    part += f", sent {s['sent'] / elapsed:.1f}/s"
                    if s["dropped"]:
                        part += f" (capped {s['dropped'] / elapsed:.1f}/s)"
                parts.append(part)
            rospy.loginfo("xense_tactile_publisher: " + "; ".join(parts))
        rate.sleep()


def _preview_encoding_of(img: np.ndarray) -> str:
    """Image encoding for a preview array made by _preview_image."""
    return "32FC1" if img.dtype == np.float32 and img.ndim == 2 else "rgb8"


if __name__ == "__main__":
    main()
