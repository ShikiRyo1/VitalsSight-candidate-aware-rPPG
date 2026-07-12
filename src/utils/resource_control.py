from __future__ import annotations

from dataclasses import dataclass
import ctypes
import os
import sys
from time import sleep


@dataclass(frozen=True)
class ResourceProfile:
    name: str
    opencv_threads: int
    sample_every: int
    max_side: int
    max_rois: int | None
    sleep_between_samples_sec: float
    below_normal_priority: bool
    max_memory_load_pct: float
    memory_pause_sec: float


PROFILES: dict[str, ResourceProfile] = {
    "interactive": ResourceProfile(
        name="interactive",
        opencv_threads=1,
        sample_every=2,
        max_side=96,
        max_rois=6,
        sleep_between_samples_sec=0.5,
        below_normal_priority=True,
        max_memory_load_pct=70.0,
        memory_pause_sec=2.0,
    ),
    "balanced": ResourceProfile(
        name="balanced",
        opencv_threads=2,
        sample_every=1,
        max_side=120,
        max_rois=9,
        sleep_between_samples_sec=0.1,
        below_normal_priority=True,
        max_memory_load_pct=82.0,
        memory_pause_sec=1.0,
    ),
    "full": ResourceProfile(
        name="full",
        opencv_threads=0,
        sample_every=1,
        max_side=140,
        max_rois=None,
        sleep_between_samples_sec=0.0,
        below_normal_priority=False,
        max_memory_load_pct=95.0,
        memory_pause_sec=0.5,
    ),
}


def apply_thread_environment(profile_name: str) -> ResourceProfile:
    profile = PROFILES[profile_name]
    threads = str(max(1, profile.opencv_threads or 4))
    os.environ.setdefault("OMP_NUM_THREADS", threads)
    os.environ.setdefault("MKL_NUM_THREADS", threads)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", threads)
    os.environ.setdefault("NUMEXPR_NUM_THREADS", threads)
    return profile


def configure_runtime(profile: ResourceProfile) -> None:
    try:
        import cv2

        if profile.opencv_threads > 0:
            cv2.setNumThreads(profile.opencv_threads)
    except Exception:
        pass

    if profile.below_normal_priority and sys.platform.startswith("win"):
        _set_windows_below_normal_priority()


def _set_windows_below_normal_priority() -> None:
    below_normal_priority_class = 0x00004000
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    handle = kernel32.GetCurrentProcess()
    kernel32.SetPriorityClass(handle, below_normal_priority_class)


def nap(profile: ResourceProfile) -> None:
    wait_for_memory(profile)
    if profile.sleep_between_samples_sec > 0:
        sleep(profile.sleep_between_samples_sec)


def wait_for_memory(profile: ResourceProfile) -> None:
    while True:
        load = memory_load_percent()
        if load is None or load < profile.max_memory_load_pct:
            return
        sleep(profile.memory_pause_sec)


def memory_load_percent() -> float | None:
    if not sys.platform.startswith("win"):
        return None

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    if not kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return float(status.dwMemoryLoad)
