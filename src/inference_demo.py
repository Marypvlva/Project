"""
Live on-the-fly inference for the temporal Transformer LIG regressor.

Supported live backends:
    --backend phantom   # real Vision Research Phantom camera via Windows SDK
    --backend opencv    # webcam / RTSP / local development fallback

Real installation:
    Phantom camera -> 1 Gb Ethernet -> Windows PC -> Phantom SDK -> this script

The model predicts FINAL sample resistance (kOhm/sq).

IMPORTANT TRAIN/INFERENCE CONTRACT
----------------------------------
The current train_transformer.py trains TemporalResistanceRegressor with:
    frames           [B, T, 3, H, W]
    laser_params     [B, 3] = z-scored [power_mW, speed_mm_s, distance_um]
    progress input   [B, 1] = z-scored scan_mm
    frame_mask       [B, T]
    frame_elapsed_s  [B, T] = z-scored physical elapsed seconds

Relative position 0..1 is used only to address the training grid and weight
the loss. The live program derives it from physical scan distance:

    scan_mm_now = elapsed_seconds_now * speed_mm_s
    relative_position_now = scan_mm_now / scan_length_mm

The Transformer itself receives z-scored scan_mm and frame_elapsed_s, exactly
as during training.

The Transformer is trained on a discrete relative-position grid
(position_step, usually 0.02 = 2%). The camera is read continuously, but the
model receives a fresh frame only when the laser reaches the next grid point
along the known scan length: 2%, 4%, 6%, ...

Camera capture runs in a separate thread and stores only the latest frame so
high-FPS acquisition does not create a backlog of stale frames.

DISPLAY CONTRACT
----------------
The video overlay intentionally contains ONLY the final resistance prediction.
All other diagnostics remain available in the terminal and optional CSV log.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import json
import math
import os
import platform
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


# -----------------------------------------------------------------------------
# Project imports
# -----------------------------------------------------------------------------

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from transformer_model import load_temporal_regressor
from video_processor import VideoProcessor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "temporal_regressor.pt"
DEFAULT_POSITION_STEP = 0.02
DEFAULT_CENSOR_THRESHOLD_KOHM = 86.0

# Vision Research Phantom SDK 13.4.787.0 constants from the provided headers.
PHCON_HEADER_VERSION = 787
PHANTOM_UC_VIEW = 1
PHANTOM_GCI_MAXIMGSIZE = 400
PHANTOM_MAX_STRING = 256


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------

def get_device(requested: str) -> torch.device:
    requested = requested.lower()

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested, but CUDA is unavailable.")
        return torch.device("cuda")

    if requested == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is None or not mps_backend.is_available():
            raise RuntimeError("MPS requested, but MPS is unavailable.")
        return torch.device("mps")

    if requested == "cpu":
        return torch.device("cpu")

    raise ValueError("--device must be auto, cuda, mps or cpu")


def parse_capture_source(source: str) -> int | str:
    value = source.strip()
    return int(value) if value.isdigit() else value


def _finite_positive(value: float) -> bool:
    return math.isfinite(value) and value > 0.0


# -----------------------------------------------------------------------------
# OpenCV latest-frame backend
# -----------------------------------------------------------------------------

class LatestFrameCapture:
    """Continuously read an OpenCV source and retain only the newest BGR frame."""

    def __init__(
        self,
        source: str,
        *,
        max_consecutive_failures: int = 200,
    ) -> None:
        if max_consecutive_failures <= 0:
            raise ValueError("max_consecutive_failures must be > 0")

        self.source_text = source
        self.source = parse_capture_source(source)
        self.max_consecutive_failures = int(max_consecutive_failures)

        self.capture = cv2.VideoCapture(self.source)
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open live source: {self.source_text}")

        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._first_frame_event = threading.Event()
        self._frame: np.ndarray | None = None
        self._frame_timestamp: float | None = None
        self._sequence = 0
        self._error: str | None = None

        self._thread = threading.Thread(
            target=self._reader_loop,
            name="latest-frame-capture",
            daemon=True,
        )

    @property
    def error(self) -> str | None:
        with self._lock:
            return self._error

    def start(self) -> "LatestFrameCapture":
        self._thread.start()
        return self

    def _reader_loop(self) -> None:
        failures = 0

        try:
            while not self._stop_event.is_set():
                ok, frame = self.capture.read()

                if not ok or frame is None:
                    failures += 1

                    if failures >= self.max_consecutive_failures:
                        with self._lock:
                            self._error = (
                                "Live source stopped producing frames after "
                                f"{failures} consecutive read failures."
                            )
                        self._stop_event.set()
                        break

                    time.sleep(0.01)
                    continue

                failures = 0
                timestamp = time.monotonic()

                with self._lock:
                    self._frame = frame
                    self._frame_timestamp = timestamp
                    self._sequence += 1

                self._first_frame_event.set()

        except Exception as exc:
            with self._lock:
                self._error = (
                    f"Capture thread error: {type(exc).__name__}: {exc}"
                )
            self._stop_event.set()

    def wait_for_first_frame(
        self,
        timeout_seconds: float,
    ) -> tuple[np.ndarray, float, int]:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")

        if not self._first_frame_event.wait(timeout_seconds):
            if self.error:
                raise RuntimeError(self.error)

            raise TimeoutError(
                f"No frame received from {self.source_text} "
                f"within {timeout_seconds:.1f}s"
            )

        latest = self.get_latest(copy=True)
        if latest is None:
            raise RuntimeError(
                "First-frame event set but no frame is available"
            )

        return latest

    def get_latest(
        self,
        *,
        copy: bool = True,
    ) -> tuple[np.ndarray, float, int] | None:
        with self._lock:
            if self._frame is None or self._frame_timestamp is None:
                return None

            frame = self._frame.copy() if copy else self._frame
            return frame, self._frame_timestamp, self._sequence

    def is_stopped(self) -> bool:
        return self._stop_event.is_set()

    def stop(self) -> None:
        self._stop_event.set()

        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

        self.capture.release()


# -----------------------------------------------------------------------------
# Phantom SDK backend (Windows, SDK 13.4.787.0)
# -----------------------------------------------------------------------------

class PhantomImageRange(ctypes.Structure):
    _fields_ = [
        ("First", ctypes.c_int32),
        ("Cnt", ctypes.c_uint32),
    ]


class PhantomImageHeader(ctypes.Structure):
    """Vision Research IH structure from phint.h."""

    _fields_ = [
        ("biSize", ctypes.c_uint32),
        ("biWidth", ctypes.c_int32),
        ("biHeight", ctypes.c_int32),
        ("biPlanes", ctypes.c_uint16),
        ("biBitCount", ctypes.c_uint16),
        ("biCompression", ctypes.c_uint32),
        ("biSizeImage", ctypes.c_uint32),
        ("biXPelsPerMeter", ctypes.c_int32),
        ("biYPelsPerMeter", ctypes.c_int32),
        ("biClrUsed", ctypes.c_uint32),
        ("biClrImportant", ctypes.c_uint32),
        ("BlackLevel", ctypes.c_int32),
        ("WhiteLevel", ctypes.c_int32),
    ]


class PhantomSDK:
    """
    Minimal read-only ctypes wrapper for the provided Phantom SDK.

    It registers the SDK client, discovers cameras, opens the live cine and
    reads live images. It does NOT modify exposure, recording, trigger,
    partitions or other acquisition settings.
    """

    def __init__(self, sdk_dir: Path) -> None:
        if platform.system() != "Windows":
            raise RuntimeError(
                "The provided Phantom SDK DLLs are Windows-only. "
                "Run --backend phantom on the Windows acquisition computer."
            )

        self.sdk_dir = Path(sdk_dir).expanduser().resolve()
        if not self.sdk_dir.is_dir():
            raise FileNotFoundError(
                f"Phantom SDK directory not found: {self.sdk_dir}"
            )

        phcon_path = self._find_dll("PhCon.dll")
        phfile_path = self._find_dll("PhFile.dll")

        self._dll_dir_handle = None
        add_dll_directory = getattr(os, "add_dll_directory", None)
        if add_dll_directory is not None:
            self._dll_dir_handle = add_dll_directory(str(self.sdk_dir))

        self.phcon = ctypes.CDLL(str(phcon_path))
        self.phfile = ctypes.CDLL(str(phfile_path))
        self._configure_signatures()
        self._registered = False

    def _find_dll(self, name: str) -> Path:
        direct = self.sdk_dir / name
        if direct.exists():
            return direct

        matches = [
            path
            for path in self.sdk_dir.iterdir()
            if path.is_file() and path.name.casefold() == name.casefold()
        ]

        if len(matches) == 1:
            return matches[0]

        raise FileNotFoundError(f"{name} not found in {self.sdk_dir}")

    def _configure_signatures(self) -> None:
        self.phcon.PhLVRegisterClientEx.argtypes = [
            ctypes.c_char_p,
            ctypes.c_void_p,
            ctypes.c_uint32,
        ]
        self.phcon.PhLVRegisterClientEx.restype = ctypes.c_int32

        self.phcon.PhLVUnregisterClient.argtypes = []
        self.phcon.PhLVUnregisterClient.restype = ctypes.c_int32

        self.phcon.PhConfigPoolUpdate.argtypes = [ctypes.c_uint32]
        self.phcon.PhConfigPoolUpdate.restype = ctypes.c_int32

        self.phcon.PhGetCameraCount.argtypes = [
            ctypes.POINTER(ctypes.c_uint32)
        ]
        self.phcon.PhGetCameraCount.restype = ctypes.c_int32

        self.phcon.PhGetCameraID.argtypes = [
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_char_p,
        ]
        self.phcon.PhGetCameraID.restype = ctypes.c_int32

        self.phcon.PhGetErrorMessage.argtypes = [
            ctypes.c_int32,
            ctypes.c_char_p,
        ]
        self.phcon.PhGetErrorMessage.restype = ctypes.c_int32

        self.phfile.PhGetCineLive.argtypes = [
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.phfile.PhGetCineLive.restype = ctypes.c_int32

        self.phfile.PhDestroyCine.argtypes = [ctypes.c_void_p]
        self.phfile.PhDestroyCine.restype = ctypes.c_int32

        self.phfile.PhSetUseCase.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int32,
        ]
        self.phfile.PhSetUseCase.restype = ctypes.c_int32

        self.phfile.PhGetCineInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_void_p,
        ]
        self.phfile.PhGetCineInfo.restype = ctypes.c_int32

        self.phfile.PhGetCineImage.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(PhantomImageRange),
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.c_uint32,
            ctypes.POINTER(PhantomImageHeader),
        ]
        self.phfile.PhGetCineImage.restype = ctypes.c_int32

    def error_text(self, hr: int) -> str:
        buffer = ctypes.create_string_buffer(PHANTOM_MAX_STRING)

        try:
            self.phcon.PhGetErrorMessage(ctypes.c_int32(hr), buffer)
            text = buffer.value.decode(
                "utf-8",
                errors="replace",
            ).strip()
        except Exception:
            text = ""

        return text or f"HRESULT={hr}"

    def check_hr(self, hr: int, operation: str) -> None:
        value = int(hr)

        if value < 0:
            raise RuntimeError(
                f"Phantom SDK {operation} failed: {self.error_text(value)} "
                f"(HRESULT={value})"
            )

    def register(self) -> None:
        if self._registered:
            return

        hr = self.phcon.PhLVRegisterClientEx(
            None,
            None,
            PHCON_HEADER_VERSION,
        )
        self.check_hr(hr, "PhLVRegisterClientEx")
        self._registered = True

        hr = self.phcon.PhConfigPoolUpdate(1500)
        self.check_hr(hr, "PhConfigPoolUpdate")

    def unregister(self) -> None:
        if not self._registered:
            return

        try:
            hr = self.phcon.PhLVUnregisterClient()
            self.check_hr(hr, "PhLVUnregisterClient")
        finally:
            self._registered = False

            if self._dll_dir_handle is not None:
                try:
                    self._dll_dir_handle.close()
                except Exception:
                    pass

                self._dll_dir_handle = None

    def camera_count(self) -> int:
        count = ctypes.c_uint32(0)

        hr = self.phcon.PhGetCameraCount(
            ctypes.byref(count)
        )
        self.check_hr(hr, "PhGetCameraCount")

        return int(count.value)

    def wait_for_cameras(
        self,
        timeout_seconds: float,
    ) -> int:
        deadline = time.monotonic() + timeout_seconds

        while True:
            count = self.camera_count()

            if count > 0:
                return count

            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "No Phantom camera discovered. Check Ethernet connection, "
                    "PCC/Phantom network configuration, Windows firewall and SDK."
                )

            time.sleep(0.25)

    def camera_identity(
        self,
        camera_index: int,
    ) -> tuple[int, str]:
        serial = ctypes.c_uint32(0)
        name = ctypes.create_string_buffer(PHANTOM_MAX_STRING)

        hr = self.phcon.PhGetCameraID(
            camera_index,
            ctypes.byref(serial),
            name,
        )
        self.check_hr(hr, "PhGetCameraID")

        return (
            int(serial.value),
            name.value.decode("utf-8", errors="replace").strip(),
        )

    def open_live_cine(
        self,
        camera_index: int,
    ) -> tuple[ctypes.c_void_p, int]:
        handle = ctypes.c_void_p()

        hr = self.phfile.PhGetCineLive(
            camera_index,
            ctypes.byref(handle),
        )
        self.check_hr(hr, "PhGetCineLive")

        if not handle.value:
            raise RuntimeError(
                "PhGetCineLive returned a null CINEHANDLE"
            )

        try:
            hr = self.phfile.PhSetUseCase(
                handle,
                PHANTOM_UC_VIEW,
            )
            self.check_hr(hr, "PhSetUseCase(UC_VIEW)")

            max_size = ctypes.c_uint32(0)

            hr = self.phfile.PhGetCineInfo(
                handle,
                PHANTOM_GCI_MAXIMGSIZE,
                ctypes.byref(max_size),
            )
            self.check_hr(
                hr,
                "PhGetCineInfo(GCI_MAXIMGSIZE)",
            )

            if max_size.value <= 0:
                raise RuntimeError(
                    "Phantom SDK returned GCI_MAXIMGSIZE <= 0"
                )

            return handle, int(max_size.value)

        except Exception:
            try:
                self.phfile.PhDestroyCine(handle)
            except Exception:
                pass
            raise

    def destroy_cine(
        self,
        handle: ctypes.c_void_p | None,
    ) -> None:
        if handle is None or not handle.value:
            return

        hr = self.phfile.PhDestroyCine(handle)
        self.check_hr(hr, "PhDestroyCine")


def _scale_u16_to_u8(
    image: np.ndarray,
    black_level: int,
    white_level: int,
) -> np.ndarray:
    values = image.astype(np.float32)
    low = float(black_level)
    high = float(white_level)

    if (
        not math.isfinite(low)
        or not math.isfinite(high)
        or high <= low
    ):
        low = float(values.min())
        high = float(values.max())

    if high <= low:
        return np.zeros(
            image.shape,
            dtype=np.uint8,
        )

    values = (
        (values - low)
        * (255.0 / (high - low))
    )

    return np.clip(
        values,
        0.0,
        255.0,
    ).astype(np.uint8)


def phantom_buffer_to_bgr(
    buffer: ctypes.Array,
    header: PhantomImageHeader,
    max_buffer_size: int,
    *,
    vertical_flip: bool,
) -> np.ndarray:
    """Convert a Phantom live image buffer to BGR uint8 [H,W,3]."""

    width = int(header.biWidth)
    height = abs(int(header.biHeight))
    bit_count = int(header.biBitCount)

    if width <= 0 or height <= 0:
        raise RuntimeError(
            f"Invalid Phantom image size: "
            f"{header.biWidth}x{header.biHeight}"
        )

    if bit_count not in (8, 16, 24, 48):
        raise RuntimeError(
            f"Unsupported Phantom live bit depth: {bit_count}; "
            "expected 8/16/24/48"
        )

    bytes_per_pixel = bit_count // 8
    row_bytes = width * bytes_per_pixel
    stride = row_bytes
    size_image = int(header.biSizeImage)

    if (
        size_image >= row_bytes * height
        and size_image % height == 0
    ):
        candidate = size_image // height
        if candidate >= row_bytes:
            stride = candidate

    required = stride * height

    if required > max_buffer_size:
        raise RuntimeError(
            f"Phantom image needs {required} bytes, "
            f"buffer has {max_buffer_size}"
        )

    raw = np.ctypeslib.as_array(
        buffer,
        shape=(max_buffer_size,),
    )

    rows = raw[:required].reshape(
        height,
        stride,
    )[:, :row_bytes]

    if bit_count == 8:
        gray = rows.reshape(
            height,
            width,
        ).copy()
        bgr = cv2.cvtColor(
            gray,
            cv2.COLOR_GRAY2BGR,
        )

    elif bit_count == 16:
        gray16 = (
            np.ascontiguousarray(rows)
            .view(np.uint16)
            .reshape(height, width)
        )

        gray8 = _scale_u16_to_u8(
            gray16,
            int(header.BlackLevel),
            int(header.WhiteLevel),
        )

        bgr = cv2.cvtColor(
            gray8,
            cv2.COLOR_GRAY2BGR,
        )

    elif bit_count == 24:
        bgr = rows.reshape(
            height,
            width,
            3,
        ).copy()

    else:
        bgr16 = (
            np.ascontiguousarray(rows)
            .view(np.uint16)
            .reshape(height, width, 3)
        )

        bgr = _scale_u16_to_u8(
            bgr16,
            int(header.BlackLevel),
            int(header.WhiteLevel),
        )

    if vertical_flip:
        bgr = np.ascontiguousarray(
            bgr[::-1]
        )

    if (
        bgr.dtype != np.uint8
        or bgr.ndim != 3
        or bgr.shape[2] != 3
    ):
        raise RuntimeError(
            f"Phantom conversion produced "
            f"shape={bgr.shape}, dtype={bgr.dtype}"
        )

    return bgr


class PhantomLatestFrameCapture:
    """Continuously read Phantom live cine and retain only the newest BGR frame."""

    def __init__(
        self,
        sdk_dir: Path,
        *,
        camera_index: int = 0,
        camera_timeout_seconds: float = 20.0,
        max_consecutive_failures: int = 50,
        vertical_flip: bool = False,
    ) -> None:
        if camera_index < 0:
            raise ValueError(
                "--camera-index must be >= 0"
            )

        if camera_timeout_seconds <= 0:
            raise ValueError(
                "--camera-timeout must be > 0"
            )

        if max_consecutive_failures <= 0:
            raise ValueError(
                "max_consecutive_failures must be > 0"
            )

        self.sdk = PhantomSDK(sdk_dir)
        self.camera_index = int(camera_index)
        self.camera_timeout_seconds = float(
            camera_timeout_seconds
        )
        self.max_consecutive_failures = int(
            max_consecutive_failures
        )
        self.vertical_flip = bool(vertical_flip)

        self.camera_count: int | None = None
        self.camera_serial: int | None = None
        self.camera_name: str | None = None

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._first_frame_event = threading.Event()
        self._frame: np.ndarray | None = None
        self._frame_timestamp: float | None = None
        self._sequence = 0
        self._error: str | None = None

        self._cine_handle: ctypes.c_void_p | None = None
        self._max_image_size = 0
        self._image_buffer = None

        self._thread = threading.Thread(
            target=self._reader_loop,
            name="phantom-latest-frame-capture",
            daemon=True,
        )

    @property
    def error(self) -> str | None:
        with self._lock:
            return self._error

    def start(
        self,
    ) -> "PhantomLatestFrameCapture":
        try:
            self.sdk.register()

            count = self.sdk.wait_for_cameras(
                self.camera_timeout_seconds
            )
            self.camera_count = count

            if self.camera_index >= count:
                raise ValueError(
                    f"--camera-index={self.camera_index}, "
                    f"but SDK found {count} camera(s)"
                )

            (
                self.camera_serial,
                self.camera_name,
            ) = self.sdk.camera_identity(
                self.camera_index
            )

            (
                self._cine_handle,
                self._max_image_size,
            ) = self.sdk.open_live_cine(
                self.camera_index
            )

            self._image_buffer = (
                ctypes.c_ubyte
                * self._max_image_size
            )()

            self._thread.start()
            return self

        except Exception:
            self._cleanup()
            raise

    def _reader_loop(self) -> None:
        failures = 0

        try:
            if (
                self._cine_handle is None
                or self._image_buffer is None
            ):
                raise RuntimeError(
                    "Phantom live cine is not initialized"
                )

            image_range = PhantomImageRange(
                First=0,
                Cnt=1,
            )

            while not self._stop_event.is_set():
                header = PhantomImageHeader()

                hr = self.sdk.phfile.PhGetCineImage(
                    self._cine_handle,
                    ctypes.byref(image_range),
                    self._image_buffer,
                    self._max_image_size,
                    ctypes.byref(header),
                )

                if int(hr) < 0:
                    failures += 1

                    if (
                        failures
                        >= self.max_consecutive_failures
                    ):
                        with self._lock:
                            self._error = (
                                f"PhGetCineImage failed {failures} times: "
                                f"{self.sdk.error_text(int(hr))} "
                                f"(HRESULT={int(hr)})"
                            )

                        self._stop_event.set()
                        break

                    time.sleep(0.005)
                    continue

                failures = 0

                frame = phantom_buffer_to_bgr(
                    self._image_buffer,
                    header,
                    self._max_image_size,
                    vertical_flip=self.vertical_flip,
                )

                timestamp = time.monotonic()

                with self._lock:
                    self._frame = frame
                    self._frame_timestamp = timestamp
                    self._sequence += 1

                self._first_frame_event.set()

        except Exception as exc:
            with self._lock:
                self._error = (
                    f"Phantom capture thread error: "
                    f"{type(exc).__name__}: {exc}"
                )

            self._stop_event.set()

    def wait_for_first_frame(
        self,
        timeout_seconds: float,
    ) -> tuple[np.ndarray, float, int]:
        if timeout_seconds <= 0:
            raise ValueError(
                "timeout_seconds must be > 0"
            )

        if not self._first_frame_event.wait(
            timeout_seconds
        ):
            if self.error:
                raise RuntimeError(self.error)

            raise TimeoutError(
                f"No live Phantom image received "
                f"within {timeout_seconds:.1f}s"
            )

        latest = self.get_latest(copy=True)

        if latest is None:
            raise RuntimeError(
                "First-frame event set but no Phantom frame is available"
            )

        return latest

    def get_latest(
        self,
        *,
        copy: bool = True,
    ) -> tuple[np.ndarray, float, int] | None:
        with self._lock:
            if (
                self._frame is None
                or self._frame_timestamp is None
            ):
                return None

            frame = (
                self._frame.copy()
                if copy
                else self._frame
            )

            return (
                frame,
                self._frame_timestamp,
                self._sequence,
            )

    def is_stopped(self) -> bool:
        return self._stop_event.is_set()

    def _cleanup(self) -> None:
        if self._cine_handle is not None:
            try:
                self.sdk.destroy_cine(
                    self._cine_handle
                )
            except Exception:
                pass

            self._cine_handle = None

        self._image_buffer = None

        try:
            self.sdk.unregister()
        except Exception:
            pass

    def stop(self) -> None:
        self._stop_event.set()

        if self._thread.is_alive():
            self._thread.join(
                timeout=3.0
            )

        self._cleanup()


# -----------------------------------------------------------------------------
# Capture source factory
# -----------------------------------------------------------------------------

def _resolve_phantom_sdk_dir(
    value: Path | None,
) -> Path:
    if value is not None:
        return Path(value)

    env = os.environ.get(
        "PHANTOM_SDK_DIR"
    )

    if env:
        return Path(env)

    raise ValueError(
        "--backend phantom requires --phantom-sdk-dir or PHANTOM_SDK_DIR. "
        "Use the Win64 folder that contains PhCon.dll and PhFile.dll."
    )


def create_frame_source(
    args: argparse.Namespace,
):
    if args.backend == "opencv":
        return LatestFrameCapture(
            args.source,
            max_consecutive_failures=(
                args.max_read_failures
            ),
        )

    if args.backend == "phantom":
        return PhantomLatestFrameCapture(
            _resolve_phantom_sdk_dir(
                args.phantom_sdk_dir
            ),
            camera_index=args.camera_index,
            camera_timeout_seconds=(
                args.camera_timeout
            ),
            max_consecutive_failures=(
                args.max_read_failures
            ),
            vertical_flip=args.phantom_vflip,
        )

    raise ValueError(
        f"Unsupported backend: {args.backend}"
    )


# -----------------------------------------------------------------------------
# Checkpoint + training normalization
# -----------------------------------------------------------------------------

def load_checkpoint_config(
    checkpoint_path: Path,
) -> dict[str, Any]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    if not checkpoint_path.is_file():
        raise ValueError(
            f"Checkpoint path is not a file: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    if not isinstance(checkpoint, dict):
        raise ValueError(
            "Temporal checkpoint must be a dict"
        )

    if "state_dict" not in checkpoint:
        raise ValueError(
            "Temporal checkpoint has no 'state_dict'"
        )

    config = checkpoint.get(
        "config",
        {},
    )

    if not isinstance(config, dict):
        raise ValueError(
            "checkpoint['config'] must be a dict"
        )

    return config


def _merge_norm_config(
    checkpoint_config: dict[str, Any],
    norm_json: Path | None,
) -> dict[str, Any]:
    """
    Use checkpoint values first. For missing values only, use norm JSON.
    """

    merged = dict(checkpoint_config)

    if norm_json is not None:
        if not norm_json.exists():
            raise FileNotFoundError(
                f"Normalization JSON not found: {norm_json}"
            )

        with norm_json.open(
            "r",
            encoding="utf-8",
        ) as file:
            json_config = json.load(file)

        if not isinstance(json_config, dict):
            raise ValueError(
                "Normalization JSON must contain an object/dict"
            )

        for key, value in json_config.items():
            if merged.get(key) is None:
                merged[key] = value

    return merged


def _required_scalar(
    config: dict[str, Any],
    name: str,
) -> float:
    value = config.get(name)

    if value is None:
        raise ValueError(
            f"Checkpoint is missing '{name}'. "
            "The current Transformer was trained with physical progress "
            "features, so live inference needs the same normalization. "
            "Use a checkpoint from the current train_transformer.py "
            "or pass --norm-json logs/transformer_norm.json."
        )

    value = float(value)

    if not math.isfinite(value):
        raise ValueError(
            f"{name} must be finite, got {value}"
        )

    return value


def load_training_normalization(
    checkpoint_config: dict[str, Any],
    norm_json: Path | None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    float,
    float,
    float,
    float,
]:
    config = _merge_norm_config(
        checkpoint_config,
        norm_json,
    )

    mean = config.get("laser_mean")
    std = config.get("laser_std")

    if mean is None or std is None:
        raise ValueError(
            "laser_mean/laser_std are missing. "
            "Use a checkpoint produced by the current train_transformer.py "
            "or pass --norm-json logs/transformer_norm.json"
        )

    mean_array = np.asarray(
        mean,
        dtype=np.float32,
    )
    std_array = np.asarray(
        std,
        dtype=np.float32,
    )

    if (
        mean_array.shape != (3,)
        or std_array.shape != (3,)
    ):
        raise ValueError(
            "laser_mean/std must both have shape (3,), "
            f"got {mean_array.shape} and {std_array.shape}"
        )

    if (
        not np.all(np.isfinite(mean_array))
        or not np.all(np.isfinite(std_array))
    ):
        raise ValueError(
            "laser_mean/std contain NaN or inf"
        )

    if np.any(std_array <= 0):
        raise ValueError(
            "All laser_std values must be > 0"
        )

    scan_mm_mean = _required_scalar(
        config,
        "scan_mm_mean",
    )
    scan_mm_std = _required_scalar(
        config,
        "scan_mm_std",
    )
    elapsed_s_mean = _required_scalar(
        config,
        "elapsed_s_mean",
    )
    elapsed_s_std = _required_scalar(
        config,
        "elapsed_s_std",
    )

    if scan_mm_std <= 0:
        raise ValueError(
            "scan_mm_std must be > 0"
        )

    if elapsed_s_std <= 0:
        raise ValueError(
            "elapsed_s_std must be > 0"
        )

    progress_feature = config.get(
        "progress_feature",
        "scan_mm",
    )

    if progress_feature != "scan_mm":
        raise ValueError(
            "This inference implementation expects "
            "progress_feature='scan_mm', "
            f"but checkpoint says {progress_feature!r}."
        )

    return (
        mean_array,
        std_array,
        scan_mm_mean,
        scan_mm_std,
        elapsed_s_mean,
        elapsed_s_std,
    )


def normalize_laser_params(
    power_mw: float,
    speed_mm_s: float,
    distance_um: float,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    raw = np.asarray(
        [
            power_mw,
            speed_mm_s,
            distance_um,
        ],
        dtype=np.float32,
    )

    if (
        raw.shape != (3,)
        or not np.all(np.isfinite(raw))
    ):
        raise ValueError(
            "power/speed/distance must be finite numbers"
        )

    normalized = (
        raw - mean
    ) / std

    if not np.all(
        np.isfinite(normalized)
    ):
        raise FloatingPointError(
            "Normalized laser parameters contain NaN/inf"
        )

    tensor = torch.from_numpy(
        normalized.astype(np.float32)
    ).unsqueeze(0)

    if tuple(tensor.shape) != (1, 3):
        raise RuntimeError(
            f"Normalized params have wrong shape: "
            f"{tuple(tensor.shape)}"
        )

    return tensor.to(device)


def physical_features_at_position(
    position: float,
    scan_length_mm: float,
    speed_mm_s: float,
    *,
    scan_mm_mean: float,
    scan_mm_std: float,
    elapsed_s_mean: float,
    elapsed_s_std: float,
) -> tuple[
    float,
    float,
    float,
    float,
]:
    """
    Reproduce training-time physical features from scan geometry.

    For a known total laser path and constant speed:
        raw_scan_mm   = position * scan_length_mm
        raw_elapsed_s = raw_scan_mm / speed_mm_s

    Both values are then z-scored using TRAIN-only statistics saved
    in the temporal checkpoint.
    """

    if (
        not math.isfinite(position)
        or not 0.0 <= position <= 1.0
    ):
        raise ValueError(
            f"position must be in [0,1], got {position}"
        )

    if (
        not math.isfinite(scan_length_mm)
        or scan_length_mm <= 0
    ):
        raise ValueError(
            f"scan_length_mm must be finite and > 0, got {scan_length_mm}"
        )

    if (
        not math.isfinite(speed_mm_s)
        or speed_mm_s <= 0
    ):
        raise ValueError(
            f"speed_mm_s must be finite and > 0, got {speed_mm_s}"
        )

    raw_scan_mm = (
        float(position)
        * float(scan_length_mm)
    )
    raw_elapsed_s = (
        raw_scan_mm
        / float(speed_mm_s)
    )

    elapsed_s_z = (
        raw_elapsed_s
        - elapsed_s_mean
    ) / elapsed_s_std

    scan_mm_z = (
        raw_scan_mm
        - scan_mm_mean
    ) / scan_mm_std

    values = [
        raw_elapsed_s,
        raw_scan_mm,
        elapsed_s_z,
        scan_mm_z,
    ]

    if not all(
        math.isfinite(value)
        for value in values
    ):
        raise FloatingPointError(
            "Physical progress features contain NaN/inf "
            "after normalization"
        )

    return (
        raw_elapsed_s,
        raw_scan_mm,
        elapsed_s_z,
        scan_mm_z,
    )


# -----------------------------------------------------------------------------
# Temporal input matching LIGTemporalDataset
# -----------------------------------------------------------------------------

def validate_processed_frame(
    frame: torch.Tensor,
    image_size: int,
) -> None:
    expected_shape = (
        3,
        image_size,
        image_size,
    )

    if not torch.is_tensor(frame):
        raise TypeError(
            f"VideoProcessor returned "
            f"{type(frame).__name__}, expected Tensor"
        )

    if tuple(frame.shape) != expected_shape:
        raise ValueError(
            f"Processed frame shape {tuple(frame.shape)} "
            f"does not match {expected_shape}"
        )

    if frame.dtype != torch.float32:
        raise TypeError(
            f"Processed frame dtype must be float32, "
            f"got {frame.dtype}"
        )

    if not torch.isfinite(frame).all():
        raise FloatingPointError(
            "Processed frame contains NaN/inf"
        )

    frame_min = float(
        frame.min().item()
    )
    frame_max = float(
        frame.max().item()
    )

    if (
        frame_min < -1e-6
        or frame_max > 1.0 + 1e-6
    ):
        raise ValueError(
            "Processed frame must be in [0,1], "
            f"got min={frame_min:.6f}, "
            f"max={frame_max:.6f}"
        )


def build_temporal_input(
    frame_buffer: deque[torch.Tensor],
    elapsed_z_buffer: deque[float],
    window_size: int,
    image_size: int,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Match LIGTemporalDataset:
      - real frames are right-aligned;
      - zeros are padded on the left;
      - frame_mask=False for padding;
      - frame_elapsed_s is also zero-padded on the left.
    """

    live_count = len(
        frame_buffer
    )

    if live_count <= 0:
        raise ValueError(
            "Cannot build temporal input without real frames"
        )

    if live_count != len(
        elapsed_z_buffer
    ):
        raise RuntimeError(
            "frame_buffer and elapsed_z_buffer lost alignment: "
            f"{live_count} != {len(elapsed_z_buffer)}"
        )

    if live_count > window_size:
        raise RuntimeError(
            "frame_buffer is longer than window_size"
        )

    live_tensor = torch.stack(
        list(frame_buffer),
        dim=0,
    )

    expected_live = (
        live_count,
        3,
        image_size,
        image_size,
    )

    if tuple(
        live_tensor.shape
    ) != expected_live:
        raise RuntimeError(
            f"live_tensor shape={tuple(live_tensor.shape)}, "
            f"expected={expected_live}"
        )

    live_elapsed = torch.tensor(
        list(elapsed_z_buffer),
        dtype=torch.float32,
    )

    if tuple(
        live_elapsed.shape
    ) != (live_count,):
        raise RuntimeError(
            f"live_elapsed shape={tuple(live_elapsed.shape)}, "
            f"expected={(live_count,)}"
        )

    pad_count = (
        window_size
        - live_count
    )

    if pad_count > 0:
        frame_padding = torch.zeros(
            (
                pad_count,
                3,
                image_size,
                image_size,
            ),
            dtype=live_tensor.dtype,
        )

        frames = torch.cat(
            [
                frame_padding,
                live_tensor,
            ],
            dim=0,
        )

        elapsed_padding = torch.zeros(
            pad_count,
            dtype=torch.float32,
        )

        frame_elapsed_s = torch.cat(
            [
                elapsed_padding,
                live_elapsed,
            ],
            dim=0,
        )
    else:
        frames = live_tensor
        frame_elapsed_s = live_elapsed

    frame_mask = torch.zeros(
        window_size,
        dtype=torch.bool,
    )
    frame_mask[pad_count:] = True

    frames = frames.unsqueeze(
        0
    ).to(device)

    frame_mask = frame_mask.unsqueeze(
        0
    ).to(device)

    frame_elapsed_s = frame_elapsed_s.unsqueeze(
        0
    ).to(device)

    expected_frames = (
        1,
        window_size,
        3,
        image_size,
        image_size,
    )

    if tuple(
        frames.shape
    ) != expected_frames:
        raise RuntimeError(
            f"frames shape={tuple(frames.shape)}, "
            f"expected={expected_frames}"
        )

    if tuple(
        frame_mask.shape
    ) != (1, window_size):
        raise RuntimeError(
            f"frame_mask shape={tuple(frame_mask.shape)}"
        )

    if tuple(
        frame_elapsed_s.shape
    ) != (1, window_size):
        raise RuntimeError(
            f"frame_elapsed_s shape="
            f"{tuple(frame_elapsed_s.shape)}"
        )

    return (
        frames,
        frame_mask,
        frame_elapsed_s,
    )


# -----------------------------------------------------------------------------
# Model inference
# -----------------------------------------------------------------------------

@torch.inference_mode()
def predict_final_resistance(
    model: torch.nn.Module,
    frames: torch.Tensor,
    normalized_params: torch.Tensor,
    normalized_scan_mm: float,
    frame_mask: torch.Tensor,
    frame_elapsed_s: torch.Tensor,
    device: torch.device,
) -> float:
    """
    Third model input is z-scored scan_mm, matching train_transformer.py.
    """

    if not math.isfinite(
        normalized_scan_mm
    ):
        raise ValueError(
            "normalized_scan_mm must be finite, "
            f"got {normalized_scan_mm}"
        )

    if tuple(
        normalized_params.shape
    ) != (1, 3):
        raise ValueError(
            "normalized_params must be [1,3], "
            f"got {tuple(normalized_params.shape)}"
        )

    progress = torch.tensor(
        [[normalized_scan_mm]],
        dtype=torch.float32,
        device=device,
    )

    prediction = model(
        frames,
        normalized_params,
        progress,
        frame_mask=frame_mask,
        frame_elapsed_s=frame_elapsed_s,
    )

    if prediction.numel() != 1:
        raise RuntimeError(
            "Temporal model must return one value, "
            f"got shape={tuple(prediction.shape)}"
        )

    value = float(
        prediction.detach().cpu().item()
    )

    if not math.isfinite(value):
        raise FloatingPointError(
            f"Model returned NaN/inf prediction: {value}"
        )

    return value


# -----------------------------------------------------------------------------
# UI + logging
# -----------------------------------------------------------------------------

def prediction_text(
    prediction: float | None,
    censor_threshold: float,
) -> str:
    if prediction is None:
        return "Final R: waiting for first ML sample"

    if prediction < 0:
        return (
            f"Final R: {prediction:.2f} kOhm/sq "
            "[WARN < 0]"
        )

    if prediction >= censor_threshold:
        return (
            f"Final R: >= "
            f"{censor_threshold:.1f} kOhm/sq"
        )

    return (
        f"Final R: {prediction:.2f} kOhm/sq"
    )


def draw_overlay(
    frame: np.ndarray,
    *,
    prediction: float | None,
    censor_threshold: float,
) -> np.ndarray:
    """
    Draw ONLY the final resistance prediction on top of the video frame.
    """

    output = frame.copy()
    text = prediction_text(
        prediction,
        censor_threshold,
    )

    x = 24
    y = 44

    cv2.putText(
        output,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (0, 0, 0),
        5,
        cv2.LINE_AA,
    )

    cv2.putText(
        output,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return output


def create_writer(
    output_path: Path,
    frame: np.ndarray,
    output_fps: float,
) -> cv2.VideoWriter:
    if not _finite_positive(
        output_fps
    ):
        raise ValueError(
            "--output-fps must be > 0"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    height, width = frame.shape[:2]

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),
        float(output_fps),
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Could not create output video: {output_path}"
        )

    return writer


def open_prediction_log(
    path: Path | None,
):
    if path is None:
        return None, None

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    file = path.open(
        "w",
        newline="",
        encoding="utf-8",
    )

    writer = csv.writer(
        file
    )

    writer.writerow(
        [
            "elapsed_seconds_now",
            "scan_mm_now",
            "relative_position_now",
            "model_position",
            "model_elapsed_s",
            "model_scan_mm",
            "model_scan_mm_z",
            "prediction_kOhm_sq",
            "window_count",
            "capture_sequence",
            "inference_ms",
        ]
    )

    file.flush()
    return file, writer


# -----------------------------------------------------------------------------
# Live loop
# -----------------------------------------------------------------------------

def run_live_inference(
    args: argparse.Namespace,
) -> None:
    checkpoint_path = Path(
        args.checkpoint
    )

    config = load_checkpoint_config(
        checkpoint_path
    )

    device = get_device(
        args.device
    )

    model = load_temporal_regressor(
        checkpoint_path,
        map_location="cpu",
    )
    model.to(device)
    model.eval()

    window_size = int(
        model.window_size
    )

    if window_size <= 0:
        raise ValueError(
            f"Invalid model window_size: {window_size}"
        )

    image_size = int(
        config.get(
            "img_size",
            128,
        )
    )

    if image_size <= 0:
        raise ValueError(
            f"Invalid checkpoint img_size: {image_size}"
        )

    checkpoint_position_step = config.get(
        "position_step"
    )

    if checkpoint_position_step is None:
        position_step = (
            float(args.position_step)
            if args.position_step is not None
            else DEFAULT_POSITION_STEP
        )

        print(
            "[WARN] checkpoint has no position_step; "
            f"using {position_step}."
        )

    else:
        position_step = float(
            checkpoint_position_step
        )

        if args.position_step is not None:
            cli_step = float(
                args.position_step
            )

            if not math.isclose(
                cli_step,
                position_step,
                rel_tol=0.0,
                abs_tol=1e-8,
            ):
                raise ValueError(
                    "--position-step does not match training checkpoint: "
                    f"CLI={cli_step}, checkpoint={position_step}"
                )

    if (
        not math.isfinite(position_step)
        or not 0.0 < position_step < 1.0
    ):
        raise ValueError(
            f"position_step must be in (0,1), "
            f"got {position_step}"
        )

    norm_json = (
        Path(args.norm_json)
        if args.norm_json is not None
        else None
    )

    (
        laser_mean,
        laser_std,
        scan_mm_mean,
        scan_mm_std,
        elapsed_s_mean,
        elapsed_s_std,
    ) = load_training_normalization(
        config,
        norm_json,
    )

    normalized_params = normalize_laser_params(
        power_mw=args.power_mw,
        speed_mm_s=args.speed_mm_s,
        distance_um=args.distance_um,
        mean=laser_mean,
        std=laser_std,
        device=device,
    )

    censor_threshold = float(
        config.get(
            "censor_threshold_kOhm",
            DEFAULT_CENSOR_THRESHOLD_KOHM,
        )
    )

    if (
        not math.isfinite(censor_threshold)
        or censor_threshold <= 0
    ):
        raise ValueError(
            f"Invalid censor threshold: {censor_threshold}"
        )

    expected_duration_seconds = (
        args.scan_length_mm
        / args.speed_mm_s
    )

    sampling_interval_seconds = (
        expected_duration_seconds
        * position_step
    )

    print("=" * 76)
    print("LIVE TEMPORAL INFERENCE")
    print("=" * 76)
    print(
        f"Backend:              {args.backend}"
    )

    if args.backend == "opencv":
        print(
            f"Source:               {args.source}"
        )
    else:
        print(
            f"Camera index:         {args.camera_index}"
        )
        print(
            "Phantom SDK dir:      "
            f"{_resolve_phantom_sdk_dir(args.phantom_sdk_dir)}"
        )

    print(
        f"Checkpoint:           {checkpoint_path}"
    )
    print(
        f"Device:               {device}"
    )
    print(
        f"Image size:           "
        f"{image_size}x{image_size}"
    )
    print(
        f"Window size:          {window_size}"
    )
    print(
        f"Position step:        {position_step}"
    )
    print(
        f"Scan length:          "
        f"{args.scan_length_mm:.6g} mm"
    )
    print(
        f"Expected scan time:   "
        f"{expected_duration_seconds:.3f} s "
        f"(scan_length / speed)"
    )
    print(
        f"ML sampling interval: "
        f"{sampling_interval_seconds:.4f} s"
    )
    print(
        "Raw params:           "
        f"[{args.power_mw}, "
        f"{args.speed_mm_s}, "
        f"{args.distance_um}]"
    )
    print(
        "Param order:           "
        "[power_mW, speed_mm_s, distance_um]"
    )
    print(
        "Normalized params:    "
        f"{normalized_params.detach().cpu().numpy().round(6).tolist()}"
    )
    print(
        "Progress input:       z-scored scan_mm "
        f"(mean={scan_mm_mean:.6g}, "
        f"std={scan_mm_std:.6g})"
    )
    print(
        "Frame time input:     z-scored elapsed_s "
        f"(mean={elapsed_s_mean:.6g}, "
        f"std={elapsed_s_std:.6g})"
    )
    print(
        f"Censor threshold:     "
        f"{censor_threshold:.3f} kOhm/sq"
    )
    print("=" * 76)

    if sampling_interval_seconds < 0.05:
        print(
            "[WARN] Neighboring ML positions are less than 50 ms apart. "
            "If model inference is slower than this, every grid update "
            "cannot be produced in real time."
        )

    processor = VideoProcessor(
        frame_size=(
            image_size,
            image_size,
        ),
        position_step=position_step,
    )

    frame_source = create_frame_source(
        args
    )

    output_writer: cv2.VideoWriter | None = None
    log_file = None
    log_writer = None
    gui_enabled = bool(
        args.display
    )

    try:
        frame_source.start()

        if isinstance(
            frame_source,
            PhantomLatestFrameCapture,
        ):
            print(
                "Phantom camera: "
                f"index={frame_source.camera_index}, "
                f"serial={frame_source.camera_serial}, "
                f"name={frame_source.camera_name!r}"
            )

        (
            first_frame,
            first_frame_timestamp,
            first_sequence,
        ) = frame_source.wait_for_first_frame(
            args.first_frame_timeout
        )

        print(
            f"First live frame: "
            f"shape={first_frame.shape}, "
            f"dtype={first_frame.dtype}, "
            f"seq={first_sequence}"
        )

        if (
            first_frame.ndim != 3
            or first_frame.shape[2] != 3
            or first_frame.dtype != np.uint8
        ):
            raise RuntimeError(
                "Capture backend must return BGR uint8 [H,W,3], got "
                f"shape={first_frame.shape}, dtype={first_frame.dtype}"
            )

        process_start_time = (
            first_frame_timestamp
            + args.start_delay_seconds
        )

        position_grid = np.arange(
            position_step,
            1.0,
            position_step,
            dtype=np.float32,
        )

        position_grid = position_grid[
            (position_grid > 0.0)
            & (position_grid < 1.0)
        ]

        if len(position_grid) == 0:
            raise RuntimeError(
                "position_grid is empty"
            )

        frame_buffer: deque[
            torch.Tensor
        ] = deque(
            maxlen=window_size
        )

        elapsed_z_buffer: deque[
            float
        ] = deque(
            maxlen=window_size
        )

        next_grid_index = 0
        last_sampled_sequence = -1
        last_prediction: float | None = None
        last_model_position: float | None = None
        last_inference_ms: float | None = None

        output_path = (
            Path(args.output)
            if args.output is not None
            else None
        )

        log_path = (
            Path(args.log_csv)
            if args.log_csv is not None
            else None
        )

        log_file, log_writer = (
            open_prediction_log(
                log_path
            )
        )

        preview_period = (
            1.0
            / args.preview_fps
        )

        next_preview_time = (
            time.monotonic()
        )

        while True:
            if frame_source.is_stopped():
                if frame_source.error:
                    raise RuntimeError(
                        frame_source.error
                    )

                raise RuntimeError(
                    "Live capture thread stopped unexpectedly"
                )

            latest = frame_source.get_latest(
                copy=True
            )

            if latest is None:
                time.sleep(0.001)
                continue

            (
                raw_frame,
                _capture_timestamp,
                capture_sequence,
            ) = latest

            now = time.monotonic()

            if now < process_start_time:
                elapsed_now = 0.0
                scan_mm_now = 0.0
                relative_position_now = 0.0

            else:
                elapsed_now = (
                    now
                    - process_start_time
                )

                scan_mm_now = (
                    elapsed_now
                    * args.speed_mm_s
                )

                relative_position_now = min(
                    max(
                        scan_mm_now
                        / args.scan_length_mm,
                        0.0,
                    ),
                    1.0,
                )

                if (
                    next_grid_index
                    < len(position_grid)
                ):
                    skipped_positions: list[
                        float
                    ] = []

                    while (
                        next_grid_index + 1
                        < len(position_grid)
                        and relative_position_now
                        >= float(
                            position_grid[
                                next_grid_index + 1
                            ]
                        )
                    ):
                        skipped_positions.append(
                            float(
                                position_grid[
                                    next_grid_index
                                ]
                            )
                        )
                        next_grid_index += 1

                    if skipped_positions:
                        print(
                            "[WARN] Inference fell behind; "
                            "skipped obsolete model positions: "
                            + ", ".join(
                                f"{p * 100:.1f}%"
                                for p
                                in skipped_positions
                            ),
                            flush=True,
                        )

                if (
                    next_grid_index
                    < len(position_grid)
                ):
                    target_position = float(
                        position_grid[
                            next_grid_index
                        ]
                    )

                    if (
                        relative_position_now >= target_position
                        and capture_sequence
                        != last_sampled_sequence
                    ):
                        position_lag = (
                            relative_position_now
                            - target_position
                        )

                        if (
                            position_lag
                            > position_step
                            * 1.25
                        ):
                            print(
                                "[WARN] ML sampling is behind current scan position: "
                                f"relative_position_now={relative_position_now:.4f}, "
                                f"model_position={target_position:.4f}",
                                flush=True,
                            )

                        processed_frame = (
                            processor.preprocess_frame(
                                raw_frame
                            )
                        )

                        validate_processed_frame(
                            processed_frame,
                            image_size,
                        )

                        (
                            model_elapsed_s,
                            model_scan_mm,
                            model_elapsed_s_z,
                            model_scan_mm_z,
                        ) = physical_features_at_position(
                            target_position,
                            args.scan_length_mm,
                            args.speed_mm_s,
                            scan_mm_mean=(
                                scan_mm_mean
                            ),
                            scan_mm_std=(
                                scan_mm_std
                            ),
                            elapsed_s_mean=(
                                elapsed_s_mean
                            ),
                            elapsed_s_std=(
                                elapsed_s_std
                            ),
                        )

                        frame_buffer.append(
                            processed_frame
                        )

                        elapsed_z_buffer.append(
                            model_elapsed_s_z
                        )

                        (
                            frames,
                            frame_mask,
                            frame_elapsed_s,
                        ) = build_temporal_input(
                            frame_buffer=(
                                frame_buffer
                            ),
                            elapsed_z_buffer=(
                                elapsed_z_buffer
                            ),
                            window_size=(
                                window_size
                            ),
                            image_size=(
                                image_size
                            ),
                            device=device,
                        )

                        inference_start = (
                            time.perf_counter()
                        )

                        prediction = (
                            predict_final_resistance(
                                model=model,
                                frames=frames,
                                normalized_params=(
                                    normalized_params
                                ),
                                normalized_scan_mm=(
                                    model_scan_mm_z
                                ),
                                frame_mask=(
                                    frame_mask
                                ),
                                frame_elapsed_s=(
                                    frame_elapsed_s
                                ),
                                device=device,
                            )
                        )

                        inference_ms = (
                            time.perf_counter()
                            - inference_start
                        ) * 1000.0

                        last_prediction = prediction
                        last_model_position = (
                            target_position
                        )
                        last_sampled_sequence = (
                            capture_sequence
                        )
                        last_inference_ms = (
                            inference_ms
                        )

                        print(
                            f"elapsed_now={elapsed_now:8.3f}s  "
                            f"scan_now={scan_mm_now:9.4f}mm  "
                            f"rel_pos={relative_position_now:6.3f}  "
                            "model_pos="
                            f"{target_position:6.3f}  "
                            f"scan_mm={model_scan_mm:9.4f}  "
                            f"scan_z={model_scan_mm_z:8.4f}  "
                            "window="
                            f"{len(frame_buffer):02d}/"
                            f"{window_size:02d}  "
                            f"pred={prediction:10.4f} "
                            "kOhm/sq  "
                            "inference="
                            f"{inference_ms:7.2f} ms",
                            flush=True,
                        )

                        if (
                            inference_ms
                            / 1000.0
                            > sampling_interval_seconds
                        ):
                            print(
                                "[WARN] One model inference is slower than the interval "
                                "between training positions. Faster compute or a longer "
                                "physical process is required for every grid update.",
                                flush=True,
                            )

                        if log_writer is not None:
                            log_writer.writerow(
                                [
                                    f"{elapsed_now:.6f}",
                                    f"{scan_mm_now:.6f}",
                                    f"{relative_position_now:.6f}",
                                    f"{target_position:.6f}",
                                    f"{model_elapsed_s:.6f}",
                                    f"{model_scan_mm:.6f}",
                                    f"{model_scan_mm_z:.8f}",
                                    f"{prediction:.8f}",
                                    len(frame_buffer),
                                    capture_sequence,
                                    f"{inference_ms:.6f}",
                                ]
                            )

                            log_file.flush()

                        next_grid_index += 1

            if now >= next_preview_time:
                display_frame = draw_overlay(
                    raw_frame,
                    prediction=(
                        last_prediction
                    ),
                    censor_threshold=(
                        censor_threshold
                    ),
                )

                if output_path is not None:
                    if output_writer is None:
                        output_writer = (
                            create_writer(
                                output_path,
                                display_frame,
                                args.output_fps,
                            )
                        )

                    output_writer.write(
                        display_frame
                    )

                if gui_enabled:
                    try:
                        cv2.imshow(
                            "LIG Live Final Resistance",
                            display_frame,
                        )

                        key = (
                            cv2.waitKey(1)
                            & 0xFF
                        )

                        if key in (
                            27,
                            ord("q"),
                        ):
                            print(
                                "Stopped by user"
                            )
                            break

                    except cv2.error as exc:
                        print(
                            "[WARN] cv2.imshow unavailable; "
                            f"GUI disabled. {exc}"
                        )
                        gui_enabled = False

                next_preview_time = (
                    now
                    + preview_period
                )

            if (
                now >= process_start_time
                and relative_position_now >= 1.0
                and not args.keep_running
            ):
                print(
                    "Process reached full progress"
                )
                break

            time.sleep(0.001)

    except KeyboardInterrupt:
        print(
            "\nCtrl+C: stopping live inference"
        )

    finally:
        try:
            frame_source.stop()
        finally:
            processor.close()

        if output_writer is not None:
            output_writer.release()

        if log_file is not None:
            log_file.close()

        if gui_enabled:
            cv2.destroyAllWindows()

    print("=" * 76)
    print("LIVE INFERENCE FINISHED")

    if (
        "last_prediction" not in locals()
        or last_prediction is None
    ):
        print(
            "No prediction was made before the first model position"
        )
    else:
        print(
            f"Last raw prediction: "
            f"{last_prediction:.4f} kOhm/sq"
        )

        if last_inference_ms is not None:
            print(
                f"Last inference time: "
                f"{last_inference_ms:.2f} ms"
            )

    if args.output is not None:
        print(
            f"Recorded preview: {args.output}"
        )

    if args.log_csv is not None:
        print(
            f"Prediction log: {args.log_csv}"
        )

    print("=" * 76)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "True live prediction of final LIG resistance with train/inference "
            "feature parity and latest-frame capture for high-FPS cameras."
        )
    )

    parser.add_argument(
        "--backend",
        choices=[
            "phantom",
            "opencv",
        ],
        default="phantom",
        help=(
            "phantom = real Vision Research SDK; "
            "opencv = webcam/RTSP development backend"
        ),
    )

    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help=(
            "OpenCV backend only: 0, /dev/video0 or rtsp://..."
        ),
    )

    parser.add_argument(
        "--phantom-sdk-dir",
        type=Path,
        default=None,
        help=(
            "Win64 SDK folder containing PhCon.dll and PhFile.dll; "
            "or set PHANTOM_SDK_DIR"
        ),
    )

    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help=(
            "Phantom SDK camera index; normally 0 "
            "when one camera is connected"
        ),
    )

    parser.add_argument(
        "--camera-timeout",
        type=float,
        default=20.0,
        help=(
            "Seconds to wait for Phantom camera discovery"
        ),
    )

    parser.add_argument(
        "--phantom-vflip",
        action="store_true",
        help=(
            "Flip Phantom SDK live image vertically "
            "only if preview is upside-down"
        ),
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=(
            "Path to checkpoints/temporal_regressor.pt"
        ),
    )

    parser.add_argument(
        "--norm-json",
        type=Path,
        default=None,
        help=(
            "Fallback normalization config: "
            "logs/transformer_norm.json"
        ),
    )

    parser.add_argument(
        "--scan-length-mm",
        type=float,
        required=True,
        help=(
            "Total physical laser scan length in millimetres. "
            "Live relative position is derived from "
            "elapsed_seconds * speed_mm_s / scan_length_mm."
        ),
    )

    parser.add_argument(
        "--start-delay-seconds",
        type=float,
        default=0.0,
        help=(
            "Temporary software synchronization: seconds after the first "
            "live frame before laser scan position is treated as 0 mm"
        ),
    )

    parser.add_argument(
        "--power-mw",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--speed-mm-s",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--distance-um",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=[
            "auto",
            "cuda",
            "mps",
            "cpu",
        ],
    )

    parser.add_argument(
        "--position-step",
        type=float,
        default=None,
        help=(
            "Fallback only for legacy checkpoints"
        ),
    )

    parser.add_argument(
        "--display",
        action="store_true",
        help=(
            "Show local cv2 preview window with "
            "ONLY Final R overlay"
        ),
    )

    parser.add_argument(
        "--preview-fps",
        type=float,
        default=30.0,
        help=(
            "UI preview rate, independent of camera acquisition FPS"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional MP4 recording of the preview "
            "with Final R overlay"
        ),
    )

    parser.add_argument(
        "--output-fps",
        type=float,
        default=30.0,
    )

    parser.add_argument(
        "--log-csv",
        type=Path,
        default=None,
        help=(
            "Optional CSV log of predictions and "
            "physical progress features"
        ),
    )

    parser.add_argument(
        "--first-frame-timeout",
        type=float,
        default=15.0,
    )

    parser.add_argument(
        "--max-read-failures",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--keep-running",
        action="store_true",
        help=(
            "Do not stop stream automatically after full process progress"
        ),
    )

    return parser


def validate_args(
    args: argparse.Namespace,
) -> None:
    for name, value in [
        (
            "--scan-length-mm",
            args.scan_length_mm,
        ),
        (
            "--preview-fps",
            args.preview_fps,
        ),
        (
            "--output-fps",
            args.output_fps,
        ),
        (
            "--first-frame-timeout",
            args.first_frame_timeout,
        ),
        (
            "--camera-timeout",
            args.camera_timeout,
        ),
    ]:
        if not _finite_positive(
            value
        ):
            raise SystemExit(
                f"{name} must be a finite number > 0"
            )

    if (
        not math.isfinite(
            args.start_delay_seconds
        )
        or args.start_delay_seconds < 0
    ):
        raise SystemExit(
            "--start-delay-seconds must be finite and >= 0"
        )

    for name, value in [
        (
            "--power-mw",
            args.power_mw,
        ),
        (
            "--speed-mm-s",
            args.speed_mm_s,
        ),
        (
            "--distance-um",
            args.distance_um,
        ),
    ]:
        if not math.isfinite(value):
            raise SystemExit(
                f"{name} must be finite"
            )

    if args.speed_mm_s <= 0:
        raise SystemExit(
            "--speed-mm-s must be > 0"
        )

    if args.camera_index < 0:
        raise SystemExit(
            "--camera-index must be >= 0"
        )

    if args.max_read_failures <= 0:
        raise SystemExit(
            "--max-read-failures must be > 0"
        )

    if args.position_step is not None:
        if (
            not math.isfinite(
                args.position_step
            )
            or not 0.0
            < args.position_step
            < 1.0
        ):
            raise SystemExit(
                "--position-step must be in (0,1)"
            )


def main() -> None:
    args = build_argparser().parse_args()
    validate_args(args)
    run_live_inference(args)


if __name__ == "__main__":
    main()
