#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ "${1:-}" = "--help" ]; then
    printf '%s\n' 'Usage: sudo -E sh run_pcie.sh [--front-channel 0..3] [options]' \
        'Requires matching VID4 FPGA firmware and the already existing capture driver.' \
        'First select the front-channel signal group; G selects again; Q exits.'
    exit 0
fi
if [ ! -r "$ROOT/pcie_capture/libtraffic_capture.so" ]; then
    printf '%s\n' 'Build the app-local library first: sh pcie_capture/build_capture.sh' >&2
    exit 1
fi
if [ "$(id -u)" != 0 ]; then
    printf '%s\n' 'Run with sudo -E: FPGA controls and C_KEY1 require board device access.' >&2
    exit 1
fi
mkdir -p "$ROOT/tmp" "$ROOT/results"
exec 9>"$ROOT/tmp/traffic_capture.lock"
if ! flock -n 9; then
    printf '%s\n' 'Traffic capture is already running. Close its window with Q before starting again.' >&2
    exit 1
fi
# The existing vendor driver leaks its class on unload, so keep it loaded until reboot.
if [ ! -d /sys/module/pango_pci_driver ]; then
    DRIVER=${PANGO_PCIE_DRIVER:-}
    if [ -z "$DRIVER" ]; then
        printf "%s\n" "Set PANGO_PCIE_DRIVER to the matching kernel module." >&2
        exit 1
    fi
    if [ "$(uname -r)" != 6.1.99 ] || [ ! -r "$DRIVER" ] || \
       [ "$(modinfo -F vermagic "$DRIVER")" != '6.1.99 SMP mod_unload aarch64' ]; then
        printf '%s\n' 'The existing capture driver does not match this board kernel.' >&2
        exit 1
    fi
    insmod "$DRIVER"
fi
export PYTHONDONTWRITEBYTECODE=1
export PYTHONIOENCODING=utf-8
export TMPDIR="$ROOT/tmp"
DEVICE=$(python3 -B "$ROOT/check_fpga.py" --four-input --device-only)
HAS_ROI=0
MODE=traffic
PREVIOUS=
for option in "$@"; do
    if [ "$option" = "--signal-roi" ] || [ "$option" = "--pick-signal-group" ]; then HAS_ROI=1; fi
    if [ "$PREVIOUS" = "--assist-mode" ]; then MODE=$option; fi
    PREVIOUS=$option
done
if [ "$HAS_ROI" = 0 ] && [ "$MODE" = traffic ]; then set -- --pick-signal-group "$@"; fi
python3 -u -B "$ROOT/traffic_light_3way_v2_modified.py" \
    --pcie-capture --capture-library "$ROOT/pcie_capture/libtraffic_capture.so" \
    --weights "$ROOT/models/color.rknn" --imgsz 640 --direction-weights "$ROOT/direction.rknn" \
    --runtime "${RKNN_RUNTIME:?Set RKNN_RUNTIME to the installed librknnrt.so}" \
    --fpga-pci "$DEVICE" --front-channel 2 --show --no-record --confirm-frames 2 --tiles 1 --top-ratio 1 \
    --display-width 1024 --display-height 600 \
    --output "$ROOT/results/live_$(date +%Y%m%d_%H%M%S)_$$.mp4" "$@"
