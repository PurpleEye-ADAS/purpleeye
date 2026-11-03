"""App-local PCIe control for the PurpleEye traffic FPGA variant."""

import ctypes
import mmap
import os
import re
import struct
import time
from pathlib import Path

OFFSET = 0x1C0
SEQUENCE_OFFSET = 0x1D0
MAGIC = 0x54524632
VIDEO_OFFSET = 0x200
VIDEO_MAGIC = 0x34444956
CAPTURE_OFFSET = 0x1E0
CAPTURE_MAGIC = 0x48444D49
NATIVE_OFFSET = 0x270
NATIVE_MAGIC = 0x4E415456
DIRECTIONS = ("LEFT", "STRAIGHT", "RIGHT")


def read_video_status(view):
    """Read display-channel counters and physical-source mapping from the firmware."""
    signature, geometry, valid_mask, format_word = struct.unpack_from("<4I", view, VIDEO_OFFSET)
    version = format_word & 255
    if (signature, geometry) != (VIDEO_MAGIC, (720 << 16) | 1280) or version not in (1, 2) or valid_mask & ~15:
        raise RuntimeError("VID4 four-input firmware is not responding; no video source was assumed valid")
    status = {"valid_mask": valid_mask,
            "full_hdmi_capture": bool(format_word & 2048),
            "full_camera_capture": bool(format_word & 8192),
            "source_channels": (0, 1, 2, 2) if version == 2 else (0, 1, 2, 3),
            "hdmi_flags": {"initialized": bool(format_word & 256), "chip_id_ok": bool(format_word & 512),
                           "i2c_error": bool(format_word & 1024)} if version == 2 else None,
            "good_frames": struct.unpack_from("<4I", view, VIDEO_OFFSET + 16),
            "pixels": struct.unpack_from("<4I", view, VIDEO_OFFSET + 32),
            "total_frames": struct.unpack_from("<4I", view, VIDEO_OFFSET + 48)}
    if status["full_camera_capture"]:
        magic, state, frames = struct.unpack_from("<3I", view, NATIVE_OFFSET)
        if magic != NATIVE_MAGIC or state & ~31 or (state & 3) > 2 or ((state >> 2) & 3) > 2:
            raise RuntimeError("Native video source register is invalid")
        status.update(native_source=(state >> 2) & 3, native_ready=bool(state & 16),
                      native_good_frames=frames)
    return status


class CPUMotionKey:
    """Read this board's existing C_KEY1 input without changing its configuration."""

    def __init__(self):
        if b"rockchip,rk3568" not in Path("/proc/device-tree/compatible").read_bytes():
            raise RuntimeError("C_KEY1 input requires the verified RK3568 board")
        fd = os.open("/dev/mem", os.O_RDONLY | os.O_SYNC)
        try:
            self.view = mmap.mmap(fd, 4096, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ, offset=0xFDD60000)
        finally:
            os.close(fd)
        if (struct.unpack_from("<I", self.view, 0x78)[0] != 0x01000C2B or
                struct.unpack_from("<I", self.view, 0x08)[0] & (1 << 8)):
            self.close()
            raise RuntimeError("C_KEY1 GPIO0_B0 is not the verified input; no configuration was changed")
        self.observed = self.stable = self.level()
        self.changed_at = time.monotonic()
        self.pressed_at = None
        print("C_KEY1 connected: short=move/stop, hold >=3s=auto", flush=True)

    def level(self):
        return (struct.unpack_from("<I", self.view, 0x70)[0] >> 8) & 1

    def poll(self, now=None):
        now = time.monotonic() if now is None else now
        level = self.level()
        if level != self.observed:
            self.observed, self.changed_at = level, now
        if level == self.stable or now - self.changed_at < 0.02:
            return None
        self.stable = level
        if level == 0:
            self.pressed_at = now
        elif self.pressed_at is not None:
            action = "AUTO" if now - self.pressed_at >= 3 else "TOGGLE"
            self.pressed_at = None
            return action
        return None

    def close(self):
        if self.view is not None:
            self.view.close()
            self.view = None


class FPGAInteraction:
    def __init__(self, device):
        if not re.fullmatch(r"[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]", device):
            raise ValueError("fpga-pci requires a PCI device address such as 0002:21:00.0")
        root = Path("/sys/bus/pci/devices") / device
        if ((root / "vendor").read_text().strip(), (root / "device").read_text().strip()) != ("0x0755", "0x0755"):
            raise ValueError("Selected PCI device is not the project's Pango endpoint")
        self.read_map = self.write_map = self.command_word = None
        self.sequence_word = None
        self.last_control = None
        try:
            with (root / "resource0").open("rb", buffering=0) as handle:
                self.read_map = mmap.mmap(handle.fileno(), 0x2000, access=mmap.ACCESS_READ)
            signature = struct.unpack_from("<I", self.read_map, OFFSET)[0]
            if signature != MAGIC:
                raise RuntimeError("Traffic FPGA signature missing. Load the matching bitstream; no registers were written.")
            fd = os.open(root / "resource1", os.O_RDWR | os.O_SYNC)
            try:
                self.write_map = mmap.mmap(fd, 0x1000, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE)
            finally:
                os.close(fd)
            self.command_word = ctypes.c_uint32.from_buffer(self.write_map, OFFSET)
            self.sequence_word = ctypes.c_uint32.from_buffer(self.write_map, SEQUENCE_OFFSET)
        except Exception:
            self.close()
            raise
        print(f"FPGA traffic interaction connected: {device}", flush=True)

    def poll(self):
        for _ in range(4):
            before = struct.unpack_from("<I", self.read_map, OFFSET + 8)[0]
            signature, control, sequence, command = struct.unpack_from("<4I", self.read_map, OFFSET)
            after = struct.unpack_from("<I", self.read_map, OFFSET + 8)[0]
            if before == sequence == after:
                break
        else:
            raise RuntimeError("FPGA controls changed repeatedly while reading; no decision was published")
        if signature != MAGIC or (control & 3) >= len(DIRECTIONS):
            raise RuntimeError("FPGA traffic status is invalid")
        self.source = (command >> 8) & 1
        self.sequence = sequence
        value = (control & 15, sequence)
        if value == self.last_control:
            return None
        self.last_control = value
        self.hardware_control = control & 15
        self.software_echo = value == getattr(self, 'software_control', None)
        if self.software_echo:
            return None
        return {"navigation": DIRECTIONS[control & 3],
                "manual_moving": bool(control & 4) if control & 8 else None}

    def publish(self, simulation, valid):
        if self.last_control is None:
            raise RuntimeError("Read the FPGA controls before publishing a decision")
        control = DIRECTIONS.index(simulation.nav)
        control |= int(simulation.manual_moving is True) << 2
        control |= int(simulation.manual_moving is not None) << 3
        override = control != self.hardware_control
        value = (0xA55A << 16) | control | int(simulation.moving) << 4 | int(simulation.alarm) << 5
        value |= int(valid) << 6 | int(override) << 7 | self.source << 8
        self.sequence_word.value = self.sequence
        # Complete the posted sequence write before sending the guarded command.
        struct.unpack_from("<I", self.read_map, OFFSET)
        self.command_word.value = value
        echo=struct.unpack_from("<I", self.read_map, OFFSET + 12)[0]
        self.software_control=((control,(self.sequence+1)&0xffffffff) if override and echo==value else None)
        return echo

    def close(self):
        if self.command_word is not None:
            self.command_word.value = 0xA55A0000 | (getattr(self, "source", 0) << 8)
            self.command_word = None
        self.sequence_word = None
        if self.write_map is not None:
            self.write_map.close()
            self.write_map = None
        if self.read_map is not None:
            self.read_map.close()
            self.read_map = None
