#!/bin/bash
set -euo pipefail
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [[ ${1:-} == --help ]]; then
    printf '%s\n' 'Usage: bash run_without_reboot.sh [run_pcie.sh options]' \
        'Open traffic recognition on the board display without rebooting.' \
        'Select the front signal group; Enter confirms, G selects again, Q exits.'
    exit 0
fi
if [[ $(id -u) != 0 ]]; then
    exec sudo -E /bin/bash "$ROOT/run_without_reboot.sh" "$@"
fi
export DISPLAY=${DISPLAY:-:0}
if [[ -z ${XAUTHORITY:-} && -n ${SUDO_USER:-} ]]; then
    desktop_home=$(getent passwd "$SUDO_USER" | cut -d: -f6)
    if [[ -n "$desktop_home" ]]; then export XAUTHORITY="$desktop_home/.Xauthority"; fi
fi
if [[ ${1:-} != --terminal ]]; then
    if ! xdpyinfo >/dev/null 2>&1; then
        printf '%s\n' 'The board desktop is not ready. Log into the board display and try again.' >&2
        exit 1
    fi
    exec xterm -geometry 78x26+660+20 -hold -T 'Traffic recognition - FPS terminal' \
        -e /bin/bash "$ROOT/run_without_reboot.sh" --terminal "$@"
fi
shift
mkdir -p "$ROOT/results"
LOG="$ROOT/results/start_$(date +%Y%m%d_%H%M%S)_$$.log"
printf 'Log: %s\n' "$LOG"
sh "$ROOT/run_pcie.sh" "$@" 2>&1 | python3 -u -c '
import logging, logging.handlers, re, sys
sink = logging.handlers.RotatingFileHandler(sys.argv[1], maxBytes=1048576, backupCount=2)
sink.setFormatter(logging.Formatter("%(message)s"))
logger = logging.getLogger("traffic-console")
logger.addHandler(sink)
logger.setLevel(logging.INFO)
for line in sys.stdin:
    logger.info(line.rstrip("\n"))
    if line.startswith("display="):
        rates = dict(re.findall(r"\b(display|signal|inference|lane|ped)=([\d.]+)", line))
        display_rate = rates.get("display", "0")
        model = rates.get("signal", rates.get("inference", "0"))
        detail = " | ".join(f"{label}={rates[key]}/s" for key, label in (("lane", "lane_model"), ("ped", "person_model")) if key in rates)
        sys.stdout.write(f"display={display_rate} FPS | signal_model={model}/s" + (" | " + detail if detail else "") + "\n")
    else:
        sys.stdout.write(line)
' "$LOG"
