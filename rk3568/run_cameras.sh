#!/bin/bash
# 姓名：丁黄炎
set -euo pipefail
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [[ ${1:-} == --help ]]; then
    printf '%s\n' 'Usage: bash run_cameras.sh [--reboot] [--qt-preview] [recognition options]' \
        'Run both cameras and the HDMI/mirrored views with the verified native preview.' \
        'Use --qt-preview to select the previous OpenCV display path.'
    exit 0
fi
REBOOT=()
if [[ ${1:-} == --reboot ]]; then REBOOT=(--reboot); shift; fi
PREVIEW=(--native-display)
if [[ ${1:-} == --qt-preview ]]; then PREVIEW=(); shift; fi
if [[ ! -r "$ROOT/native_camera/libcamera_threads.so" || "$ROOT/native_camera/camera_threads.cpp" -nt "$ROOT/native_camera/libcamera_threads.so" ]]; then
    sh "$ROOT/native_camera/build.sh"
fi
exec bash "$ROOT/run_performance.sh" "${REBOOT[@]}" --input-source CAM1 \
    --camera-direction-weights "$ROOT/camera_direction.rknn" "${PREVIEW[@]}" "$@"
