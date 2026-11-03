#!/bin/bash
set -euo pipefail
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REBOOT=()
if [[ ${1:-} == --reboot ]]; then REBOOT=(--reboot); shift; fi
if [[ ! -r "$ROOT/native_capture/libtraffic_capture.so" ]]; then
    sh "$ROOT/native_capture/build_capture.sh"
fi
exec bash "$ROOT/run_fusion.sh" "${REBOOT[@]}" --pipeline-capture --native-motion --realtime-threads \
    --capture-library "$ROOT/native_capture/libtraffic_capture.so" --cpu-threads 1 \
    --dma-chunk-bytes 3840 --dma-delay-loops 500 --auxiliary-fps 3.6 --signal-fps 2.7 "$@"
