#!/bin/bash
set -euo pipefail
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
LAUNCHER=run_without_reboot.sh
if [[ ${1:-} == --reboot ]]; then
    printf "%s\n" "Restart manually, then run again without --reboot." >&2
    exit 2
fi
exec bash "$ROOT/$LAUNCHER" --assist-mode fusion --lane-weights "$ROOT/lane.rknn" \
    --weights "$ROOT/models/color.rknn" --pedestrian-weights "$ROOT/models/person.rknn" \
    --conf-thres .65 --pedestrian-confidence .35 --display-width 640 --display-height 430 "$@"
