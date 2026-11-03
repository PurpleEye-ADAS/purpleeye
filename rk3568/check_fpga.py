"""Read-only check of the app's FPGA endpoint and traffic protocol."""

import argparse
import json
import mmap
import struct
from pathlib import Path

from traffic_fpga import DIRECTIONS, MAGIC, OFFSET, read_video_status


def inspect_fpga(require_four=False):
    reports = []
    for root in sorted(Path('/sys/bus/pci/devices').glob('*')):
        if ((root / 'vendor').read_text().strip(), (root / 'device').read_text().strip()) != ('0x0755', '0x0755'):
            continue
        item = {'device': root.name, 'ready': False}
        try:
            with (root / 'resource0').open('rb', buffering=0) as handle:
                with mmap.mmap(handle.fileno(), 0x2000, access=mmap.ACCESS_READ) as view:
                    signature, control, sequence, echo = struct.unpack_from('<4I', view, OFFSET)
                    if require_four:
                        item['video'] = read_video_status(view)
            item.update(signature=f'0x{signature:08x}', sequence=sequence, command_echo=f'0x{echo:08x}')
            item['ready'] = signature == MAGIC and (control & 3) < len(DIRECTIONS)
            if item['ready']:
                item.update(navigation=DIRECTIONS[control & 3],
                            manual_moving=bool(control & 4) if control & 8 else None)
            else:
                item['reason'] = 'Matching TRF2 traffic bitstream is not responding. No registers were written.'
        except (OSError, RuntimeError) as error:
            item['reason'] = str(error)
        reports.append(item)
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device-only', action='store_true')
    parser.add_argument('--four-input', action='store_true', help='Require the VID4 independent input status interface')
    args = parser.parse_args()
    reports = inspect_fpga(args.four_input)
    ready = [item['device'] for item in reports if item['ready']]
    if len(ready) != 1:
        print(json.dumps({'expected_signature': f'0x{MAGIC:08x}', 'endpoints': reports}, ensure_ascii=False))
        return 1
    print(ready[0] if args.device_only else json.dumps(reports, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
