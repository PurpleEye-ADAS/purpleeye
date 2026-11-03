"""Prepare reviewed traffic-light crops without changing the detection dataset."""

import argparse
import csv
import hashlib
import json
import random
import shutil
import struct
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps, ImageEnhance, ImageFilter

CLASSES = ("circle", "left", "other", "right", "straight")


def save_prepared_image(image, destination, **options):
    """Reuse complete generated images and atomically replace interrupted writes."""
    ending = b"\x00\x00\x00\x00IEND\xaeB`\x82" if destination.suffix == ".png" else b"\xff\xd9"
    if destination.is_file() and destination.stat().st_size >= len(ending):
        with destination.open("rb") as stream:
            stream.seek(-len(ending), 2)
            if stream.read() == ending:
                return
    partial = destination.with_suffix(destination.suffix + ".part")
    image.save(partial, format="PNG" if destination.suffix == ".png" else "JPEG", **options)
    partial.replace(destination)


class PublisherZip:
    """Read selected publisher ZIP entries by HTTP range, including split ZIP64 archives."""

    def __init__(self, metadata, filenames, cache):
        self.files = [next(f for f in metadata["files"] if f["key"] == name) for name in filenames]
        cache.mkdir(parents=True, exist_ok=True)
        self.cache = cache
        index = cache / "zip_index.json"
        if index.exists():
            self.entries = json.loads(index.read_text(encoding="utf-8"))
            return
        volume = len(self.files) - 1
        tail = self.read_range(volume, max(0, self.files[volume]["size"] - 65536),
                               min(65536, self.files[volume]["size"]))
        end = tail.rfind(b"PK\x05\x06")
        if end < 0:
            raise ValueError("Publisher ZIP end record is missing")
        fields = struct.unpack_from("<4s4H2IH", tail, end)
        disk, count, size, offset = fields[2], fields[4], fields[5], fields[6]
        if count == 65535 or size == 0xFFFFFFFF or offset == 0xFFFFFFFF:
            locator = struct.unpack_from("<4sIQI", tail, end - 20)
            record = self.read_range(locator[1], locator[2], 56)
            large = struct.unpack("<4sQHHIIQQQQ", record)
            if large[0] != b"PK\x06\x06":
                raise ValueError("Invalid publisher ZIP64 record")
            disk, count, size, offset = large[5], large[7], large[8], large[9]
        central = self.read_range(disk, offset, size)
        entries, cursor = [], 0
        for _ in range(count):
            values = struct.unpack_from("<4s6H3I5H2I", central, cursor)
            if values[0] != b"PK\x01\x02":
                raise ValueError("Invalid publisher ZIP central directory")
            name_len, extra_len, comment_len = values[10:13]
            name = central[cursor + 46:cursor + 46 + name_len].decode("utf-8" if values[3] & 2048 else "cp437")
            extra = central[cursor + 46 + name_len:cursor + 46 + name_len + extra_len]
            uncompressed, compressed, local, disk = values[9], values[8], values[16], values[13]
            position = 0
            while position < len(extra):
                kind, length = struct.unpack_from("<HH", extra, position)
                payload = extra[position + 4:position + 4 + length]
                if kind == 1:
                    advance = 0
                    decoded = []
                    for value, sentinel, fmt in ((uncompressed, 0xFFFFFFFF, "<Q"),
                                                  (compressed, 0xFFFFFFFF, "<Q"),
                                                  (local, 0xFFFFFFFF, "<Q"), (disk, 65535, "<I")):
                        if value == sentinel:
                            value = struct.unpack_from(fmt, payload, advance)[0]
                            advance += struct.calcsize(fmt)
                        decoded.append(value)
                    uncompressed, compressed, local, disk = decoded
                position += 4 + length
            entries.append({"name": name, "disk": disk, "offset": local, "compressed": compressed,
                            "size": uncompressed, "crc": values[7], "method": values[4], "flags": values[3]})
            cursor += 46 + name_len + extra_len + comment_len
        self.entries = entries
        index.write_text(json.dumps(entries), encoding="utf-8")

    def read_range(self, disk, offset, size):
        import requests

        result = bytearray()
        while size:
            item = self.files[disk]
            amount = min(size, item["size"] - offset)
            for attempt in range(8):
                try:
                    with requests.get(item["links"]["self"],
                                      headers={"Range": f"bytes={offset}-{offset + amount - 1}"},
                                      stream=True, timeout=(30, 90)) as response:
                        if response.status_code == 429:
                            if attempt == 7:
                                response.raise_for_status()
                            time.sleep(max(float(response.headers.get("Retry-After", 1)), min(60, 2 ** (attempt + 1))))
                            continue
                        response.raise_for_status()
                        if response.status_code != 206:
                            raise RuntimeError("Publisher server did not honor the byte range")
                        expected = f"bytes {offset}-{offset + amount - 1}/{item['size']}"
                        if response.headers.get("Content-Range") != expected:
                            raise RuntimeError("Publisher returned an unexpected byte range")
                        content = response.content
                    if len(content) != amount:
                        raise RuntimeError("Incomplete publisher byte range")
                    result.extend(content)
                    break
                except requests.RequestException:
                    if attempt == 7:
                        raise
                    time.sleep(2)
            size -= amount
            disk, offset = disk + 1, 0
        return bytes(result)

    def read(self, entry):
        if entry["flags"] & 1:
            raise ValueError("Encrypted publisher entry is unsupported")
        available = sum(f["size"] for f in self.files[entry["disk"]:]) - entry["offset"]
        content = self.read_range(entry["disk"], entry["offset"],
                                  min(available, entry["compressed"] + 1024))
        fields = struct.unpack_from("<4s5H3I2H", content)
        if fields[0] != b"PK\x03\x04":
            raise ValueError(f"Invalid local ZIP header: {entry['name']}")
        start = 30 + fields[9] + fields[10]
        if start + entry["compressed"] > len(content):
            content = self.read_range(entry["disk"], entry["offset"], start + entry["compressed"])
        return self.decode(entry, content)

    @staticmethod
    def decode(entry, content):
        fields = struct.unpack_from("<4s5H3I2H", content)
        if fields[0] != b"PK\x03\x04" or entry["flags"] & 1:
            raise ValueError(f"Invalid/encrypted ZIP entry: {entry['name']}")
        start = 30 + fields[9] + fields[10]
        compressed = content[start:start + entry["compressed"]]
        if entry["method"] == 8:
            content = zlib.decompress(compressed, -15)
        elif entry["method"] == 0:
            content = compressed
        elif entry["method"] == 9:
            from inflate64 import Inflater

            content = Inflater().inflate(compressed)
        else:
            raise ValueError(f"Unsupported ZIP compression: {entry['method']}")
        if len(content) != entry["size"] or zlib.crc32(content) != entry["crc"]:
            raise ValueError(f"Publisher entry CRC mismatch: {entry['name']}")
        return content

    def read_many(self, entries):
        """Combine adjacent small annotation entries; stream image batches without retaining all images."""
        groups = []
        for entry in sorted(entries, key=lambda item: (item["disk"], item["offset"])):
            end = entry["offset"] + entry["compressed"] + 1024
            if (groups and groups[-1][0] == entry["disk"] and entry["offset"] <= groups[-1][2] + 4096
                    and end - groups[-1][1] <= 8 * 1024 * 1024):
                groups[-1][2] = max(groups[-1][2], end)
                groups[-1][3].append(entry)
            else:
                groups.append([entry["disk"], entry["offset"], end, [entry]])

        def fetch(group):
            disk, start, end, items = group
            available = sum(f["size"] for f in self.files[disk:]) - start
            content = self.read_range(disk, start, min(available, end - start))
            return [(entry, self.decode(entry, content[entry["offset"] - start:])) for entry in items]

        pending = iter(groups)
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {pool.submit(fetch, group) for group in [next(pending, None) for _ in range(3)] if group}
            while futures:
                future = next(as_completed(futures))
                futures.remove(future)
                for result in future.result():
                    yield result
                group = next(pending, None)
                if group:
                    futures.add(pool.submit(fetch, group))


def fetch_atlas(output, train_count=1800, test_count=240):
    """Select publisher-labeled scenes before model evaluation; keep complete sequence groups apart."""
    import re
    import requests
    import yaml

    response = requests.get("https://zenodo.org/api/records/14846709", timeout=60)
    response.raise_for_status()
    metadata = response.json()
    output.mkdir(parents=True, exist_ok=True)
    (output / "source_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    archive = PublisherZip(metadata, ["ATLAS.z01", "ATLAS.z02", "ATLAS.z03", "ATLAS.z04", "ATLAS.z05", "ATLAS.zip"], output)
    entries = {item["name"]: item for item in archive.entries}
    label_cache = output / "labels.json"
    if label_cache.exists():
        labels = json.loads(label_cache.read_text(encoding="utf-8"))
    else:
        labels = {}
        for item, content in archive.read_many([e for e in archive.entries if e["name"].endswith(".txt")]):
            labels[item["name"]] = content.decode("utf-8")
        label_cache.write_text(json.dumps(labels), encoding="utf-8")
    for filename in ("ATLAS/ATLAS_classes.yaml", "ATLAS/LICENSE", "ATLAS/README.md"):
        destination = output / Path(filename).name
        if not destination.exists():
            destination.write_bytes(archive.read(entries[filename]))
    classes = {int(k): v for k, v in yaml.load((output / "ATLAS_classes.yaml").read_text(),
                                             Loader=yaml.BaseLoader)["names"].items()}

    def group(name):
        timestamp = re.search(r"_(\d{10})", Path(name).stem)
        if not timestamp:
            # Relative-clock filenames lack recording identity; keep all of them together.
            return "atlas_relative_clock_all"
        return f"atlas_sequence_{int(timestamp[1]) // 600}"

    test_groups = {group(name) for name in labels if "/test/" in name}
    candidates = {"train": [], "val": [], "test": []}
    for name, text in labels.items():
        sequence = group(name)
        if "/test/" in name:
            split = "test"
        elif sequence in test_groups:
            continue
        else:
            split = "val" if int(hashlib.sha256(sequence.encode()).hexdigest()[:8], 16) % 7 == 0 else "train"
        ids = {int(line.split()[0]) for line in text.splitlines() if line.strip()}
        candidates[split].append({"label": name, "image": name.replace("/labels/", "/images/").replace(".txt", ".jpg"),
                                  "split": split, "group": sequence, "classes": sorted(ids)})
    rng = random.Random(20261003)
    selection = []
    for split, count in (("train", train_count), ("val", max(180, train_count // 5)), ("test", test_count)):
        pool = candidates[split]
        rng.shuffle(pool)
        chosen, chosen_names = [], set()
        for category in ("arrow_right_", "arrow_straight_", "arrow_left_", "circle_yellow", "off"):
            quota = max(1, count // 7)
            for row in pool:
                if len(chosen) >= count or quota == 0:
                    break
                if row["image"] not in chosen_names and any(str(classes[index]).startswith(category) for index in row["classes"]):
                    chosen.append(row)
                    chosen_names.add(row["image"])
                    quota -= 1
        for row in pool:
            if len(chosen) >= count:
                break
            if row["image"] not in chosen_names:
                chosen.append(row)
                chosen_names.add(row["image"])
        selection.extend(chosen)
    selection_path = output / "selection.json"
    if selection_path.exists():
        selection = json.loads(selection_path.read_text())
        if dict(Counter(r["split"] for r in selection)) != {
                "train": train_count, "val": max(180, train_count // 5), "test": test_count}:
            raise ValueError("Existing public selection has different sizes; choose a new output directory")
    else:
        selection_path.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    selected_groups = {split: {r["group"] for r in selection if r["split"] == split} for split in candidates}
    if any(selected_groups[a] & selected_groups[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("Publisher sequence leakage")
    print(f"Public scene selection: {dict(Counter(r['split'] for r in selection))}", flush=True)
    pending = []
    for row in selection:
        entry, destination = entries[row["image"]], output / row["image"]
        if not destination.is_file() or zlib.crc32(destination.read_bytes()) != entry["crc"]:
            pending.append(entry)
    completed = 0
    for entry, content in archive.read_many(pending):
        destination = output / entry["name"]
        if not destination.resolve().is_relative_to(output.resolve()):
            raise ValueError("Publisher archive path escapes output")
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(destination.suffix + ".part")
        partial.write_bytes(content)
        partial.replace(destination)
        completed += 1
        if completed % 50 == 0:
            print(f"Publisher images verified by CRC: {completed}/{len(pending)}", flush=True)
    print(f"Public data ready: {output}; full-archive MD5 is not claimed for ranged entries.", flush=True)
    (output / "acquisition_complete.json").write_text(json.dumps({"images": len(selection), "verification": "ZIP entry CRC"}),
                                                     encoding="utf-8")


def unpack_bosch(archive_path, output):
    """Decode the publisher's Deflate64 archive using a process-local optional decoder."""
    import zipfile

    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive, archive_path.open("rb") as stream:
        for info in archive.infolist():
            destination = output / info.filename
            if not destination.resolve().is_relative_to(output.resolve()):
                raise ValueError("Publisher archive path escapes output")
            if info.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            stream.seek(info.header_offset)
            header = stream.read(30)
            fields = struct.unpack("<4s5H3I2H", header)
            content = header + stream.read(fields[9] + fields[10] + info.compress_size)
            entry = {"name": info.filename, "compressed": info.compress_size, "size": info.file_size,
                     "flags": info.flag_bits, "crc": info.CRC, "method": info.compress_type}
            decoded = PublisherZip.decode(entry, content)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(decoded)
    print(f"Publisher entries verified by CRC: {output}", flush=True)


def fetch_bosch(output, count=450):
    import requests
    import yaml

    response = requests.get("https://zenodo.org/api/records/12706046", timeout=60)
    response.raise_for_status()
    metadata = response.json()
    output.mkdir(parents=True, exist_ok=True)
    (output / "source_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    archive = PublisherZip(metadata, ["dataset_train_rgb.zip"], output)
    entries = {entry["name"]: entry for entry in archive.entries}
    annotation_entry = next(entry for entry in archive.entries if entry["name"].endswith("train.yaml"))
    labels_path = output / "train.yaml"
    if not labels_path.exists():
        labels_path.write_bytes(archive.read(annotation_entry))
    rows = yaml.safe_load(labels_path.read_text())
    rng = random.Random(20261003)
    rng.shuffle(rows)
    selected = []
    paths = set()
    for kind in ("Yellow", "GreenRight", "RedRight", "GreenStraight", "RedStraight", "GreenLeft", "RedLeft"):
        for row in rows:
            if len(selected) >= count:
                break
            path = row["path"].removeprefix("./")
            if path not in paths and any(box["label"] == kind for box in row["boxes"]):
                sequence = Path(path).parent.name
                selected.append({"image": f"dataset_train_rgb/{path}", "boxes": row["boxes"], "group": f"bosch_{sequence}",
                                 "split": "val" if int(hashlib.sha256(sequence.encode()).hexdigest()[:8], 16) % 7 == 0 else "train"})
                paths.add(path)
                if sum(any(box["label"] == kind for box in item["boxes"]) for item in selected) >= count // 6:
                    break
    for row in rows:
        if len(selected) >= count:
            break
        path = row["path"].removeprefix("./")
        if path in paths:
            continue
        sequence = Path(path).parent.name
        selected.append({"image": f"dataset_train_rgb/{path}", "boxes": row["boxes"], "group": f"bosch_{sequence}",
                         "split": "val" if int(hashlib.sha256(sequence.encode()).hexdigest()[:8], 16) % 7 == 0 else "train"})
        paths.add(path)
    selection_path = output / "selection.json"
    if selection_path.exists() and json.loads(selection_path.read_text()) != selected:
        raise ValueError("Existing Bosch selection differs; choose another output")
    selection_path.write_text(json.dumps(selected, indent=2), encoding="utf-8")
    print(f"Bosch scene selection: {dict(Counter(r['split'] for r in selected))}", flush=True)
    pending = []
    for row in selected:
        entry, destination = entries[row["image"]], output / row["image"]
        if not destination.is_file() or zlib.crc32(destination.read_bytes()) != entry["crc"]:
            pending.append(entry)
    for index, (entry, content) in enumerate(archive.read_many(pending), 1):
        destination = output / entry["name"]
        if not destination.resolve().is_relative_to(output.resolve()):
            raise ValueError("Publisher path escapes output")
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(destination.suffix + ".part")
        partial.write_bytes(content)
        partial.replace(destination)
        if index % 50 == 0:
            print(f"Bosch publisher images verified by CRC: {index}/{len(pending)}", flush=True)
    (output / "acquisition_complete.json").write_text(json.dumps({"images": len(selected), "verification": "ZIP entry CRC"}),
                                                     encoding="utf-8")


def public_semantics(name):
    """Keep compound arrows unsupported and map red+yellow to the stop state."""
    normalized = name.lower().replace("_", "")
    color = "off" if normalized == "off" else ("red" if "red" in normalized else
            "yellow" if "yellow" in normalized else "green" if "green" in normalized else None)
    if color is None:
        raise ValueError(f"Unknown publisher class: {name}")
    if color == "off" or ("straight" in normalized and ("left" in normalized or "right" in normalized)):
        direction = "other"
    elif "left" in normalized:
        direction = "left"
    elif "right" in normalized:
        direction = "right"
    elif "straight" in normalized:
        direction = "straight"
    else:
        direction = "circle"
    return color, direction


def mine_negatives(root, public, output):
    """Propose old-model false boxes on training frames for explicit visual review."""
    import cv2
    import torch
    from types import SimpleNamespace
    from models.common import DetectMultiBackend
    from traffic_light_3way_v2_modified import box_iou, detect_roi, second_nms

    output.mkdir(parents=True, exist_ok=False)
    (output / "crops").mkdir()
    atlas = public / "atlas_sparse"
    selection = [r for r in json.loads((atlas / "selection.json").read_text())
                 if r["split"] == "train" and (atlas / r["image"]).is_file()]
    random.Random(20261003).shuffle(selection)
    labels = json.loads((atlas / "labels.json").read_text())
    model = DetectMultiBackend(str(root / "yolov5/runs/train/traffic_real_v42/weights/best.pt"),
                               device=torch.device("cuda:0"))
    settings = SimpleNamespace(imgsz=960, conf_thres=.25, iou_thres=.45)
    rows = []
    for row in selection[:180]:
        source = atlas / row["image"]
        frame = cv2.imread(str(source))
        height, width = frame.shape[:2]
        boxes = []
        for line in labels[row["label"]].splitlines():
            _, x, y, w, h = map(float, line.split())
            boxes.append([(x-w/2)*width, (y-h/2)*height, (x+w/2)*width, (y+h/2)*height])
        tile = int(width * .62)
        with torch.inference_mode():
            detections = detect_roi(model, frame[:int(height*.75), :tile], settings)
            detections += detect_roi(model, frame[:int(height*.75), width-tile:], settings, width-tile)
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        for detection in second_nms(detections):
            if (detection[2]-detection[0] < 12 or detection[3]-detection[1] < 16 or
                    max((box_iou(detection, box) for box in boxes), default=0) >= .1):
                continue
            sample_id = f"negative_{len(rows):04d}"
            crop = f"crops/{sample_id}.png"
            square_crop(image, detection[:4]).save(output / crop)
            rows.append({"id": sample_id, "source": str(source.resolve()), "source_split": "train",
                         "box": detection[:4], "color": model.names[detection[5]], "crop": crop,
                         "source_sha256": hashlib.sha256(image.tobytes()).hexdigest(), "group": row["group"]})
        if len(rows) >= 80:
            break
    (output / "candidates.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    review_sheets(rows, output)
    print(f"Background proposals requiring visual review: {len(rows)}", flush=True)


def prepare_generalization(root, public, output, available_public=False, resume_preparation=False):
    """Build new color/direction datasets from existing labels and publisher annotations."""
    import yaml
    import re
    from datetime import datetime, timezone

    atlas_root, bosch_root = public / "atlas_sparse", public / "bosch_sparse"
    atlas_selection = json.loads((atlas_root / "selection.json").read_text())
    atlas_labels = json.loads((atlas_root / "labels.json").read_text())
    atlas_names = {int(k): v for k, v in yaml.load((atlas_root / "ATLAS_classes.yaml").read_text(),
                                                Loader=yaml.BaseLoader)["names"].items()}
    bosch_selection = json.loads((bosch_root / "selection.json").read_text())
    if available_public:
        atlas_selection = [r for r in atlas_selection if (atlas_root / r["image"]).is_file()]
        bosch_selection = [r for r in bosch_selection if (bosch_root / r["image"]).is_file()]
    for path in [atlas_root / r["image"] for r in atlas_selection] + [bosch_root / r["image"] for r in bosch_selection]:
        if not path.is_file():
            raise FileNotFoundError(f"Public acquisition is incomplete: {path}")
    output.mkdir(parents=True, exist_ok=resume_preparation)
    review = output / "direction_review"
    (review / "crops").mkdir(parents=True, exist_ok=resume_preparation)
    rows, annotations, frame_manifest = [], [], []
    color_ids = {name: index for index, name in enumerate(("red", "yellow", "green", "off"))}
    detection = output / "detection"
    for split in ("train", "val", "test"):
        for kind in ("images", "labels"):
            (detection / kind / split).mkdir(parents=True, exist_ok=resume_preparation)

    local_rows = json.loads((root / "direction_artifacts/review_v2/candidates.json").read_text())
    with (root / "direction_artifacts/review_v2/annotations.csv").open(encoding="utf-8") as stream:
        local_labels = list(csv.DictReader(stream))
    local_by_id = {row["id"]: row for row in local_rows}
    for annotation in local_labels:
        if annotation["direction"] not in CLASSES:
            continue
        original = local_by_id[annotation["id"]]
        row = {**original, "id": f"local_{original['id']}", "crop": f"crops/local_{original['id']}.png"}
        shutil.copyfile(root / "direction_artifacts/review_v2" / original["crop"], review / row["crop"])
        rows.append(row)
        annotations.append({**annotation, "id": row["id"]})

    def add_frame(source, boxes, split, group, origin):
        with Image.open(source) as decoded:
            image = decoded.convert("RGB")
        width, height = image.size
        frame_id = f"{origin}_{len(frame_manifest):06d}"
        digest = hashlib.sha256(image.tobytes()).hexdigest()
        frame_manifest.append({"id": frame_id, "source": str(source.resolve()), "split": split,
                               "group": group, "origin": origin, "source_sha256": digest,
                               "width": width, "height": height, "boxes": boxes})
        if origin != "local":
            for index, box in enumerate(boxes):
                x1, y1, x2, y2 = box["box"]
                if x2 - x1 < 12 or y2 - y1 < 16:
                    continue
                sample_id = f"{frame_id}_{index:02d}"
                crop = f"crops/{sample_id}.png"
                save_prepared_image(square_crop(image, box["box"]), review / crop)
                rows.append({"id": sample_id, "source": str(source.resolve()), "source_split": split,
                             "box": box["box"], "color": box["color"], "crop": crop, "source_sha256": digest})
                annotations.append({"id": sample_id, "direction": box["direction"], "split": split,
                                    "group": group, "reviewer": f"publisher_{origin}"})
        if split == "test":
            # Test full frames remain untouched; tiled end-to-end evaluation uses this manifest.
            return
        top = min(height, max(int(height * .75), int(max((b["box"][3] for b in boxes), default=0)) + 4))
        tile_width = max(1, int(width * .62))
        for tile, left in enumerate((0, width - tile_width)):
            right = left + tile_width
            labels = []
            for box in boxes:
                x1, y1, x2, y2 = box["box"]
                a, b, c, d = max(left, x1), max(0, y1), min(right, x2), min(top, y2)
                if c - a < 2 or d - b < 2:
                    continue
                labels.append(f"{color_ids[box['color']]} {(a + c - 2 * left) / (2 * tile_width):.8f} "
                              f"{(b + d) / (2 * top):.8f} {(c - a) / tile_width:.8f} {(d - b) / top:.8f}")
            cropped = image.crop((left, 0, right, top))
            if max(cropped.size) > 960:
                cropped.thumbnail((960, 960), Image.Resampling.LANCZOS)
            stem = f"{frame_id}_tile{tile}"
            save_prepared_image(cropped, detection / "images" / split / f"{stem}.jpg", quality=95)
            (detection / "labels" / split / f"{stem}.txt").write_text("\n".join(labels), encoding="utf-8")
        if len(frame_manifest) % 200 == 0:
            print(f"Prepared labeled frames: {len(frame_manifest)}", flush=True)

    for split in ("train", "val"):
        for source in sorted((root / "images" / split).glob("*.jpg")):
            if source.stem.startswith("real3"):
                continue
            path = root / "labels" / split / source.with_suffix(".txt").name
            if not path.is_file():
                continue
            with Image.open(source) as image:
                width, height = image.size
            boxes = []
            for line in path.read_text().splitlines():
                cls, x, y, w, h = map(float, line.split())
                boxes.append({"color": tuple(color_ids)[int(cls)], "direction": None,
                              "box": [max(0, (x - w / 2) * width), max(0, (y - h / 2) * height),
                                      min(width, (x + w / 2) * width), min(height, (y + h / 2) * height)]})
            add_frame(source, boxes, split, f"legacy_{split}_{source.stem}", "local")
    for row in atlas_selection:
        source = atlas_root / row["image"]
        with Image.open(source) as image:
            width, height = image.size
        boxes = []
        for line in atlas_labels[row["label"]].splitlines():
            cls, x, y, w, h = map(float, line.split())
            color, direction = public_semantics(atlas_names[int(cls)])
            boxes.append({"color": color, "direction": direction,
                          "box": [max(0, (x - w / 2) * width), max(0, (y - h / 2) * height),
                                  min(width, (x + w / 2) * width), min(height, (y + h / 2) * height)]})
        timestamp = re.search(r"_(\d{10})", source.stem)
        day = datetime.fromtimestamp(int(timestamp[1]), timezone.utc).strftime("%Y-%m-%d") if timestamp else "relative_clock_all"
        split = "test" if row["split"] == "test" else (
            "val" if int(hashlib.sha256(day.encode()).hexdigest()[:8], 16) % 7 == 0 else "train")
        add_frame(source, boxes, split, f"atlas_day_{day}", "atlas")
    for row in bosch_selection:
        boxes = []
        for original in row["boxes"]:
            color, direction = public_semantics(original["label"])
            boxes.append({"color": color, "direction": direction,
                          "box": [original["x_min"], original["y_min"], original["x_max"], original["y_max"]]})
        add_frame(bosch_root / row["image"], boxes, row["split"], row["group"], "bosch")
    additional = public / "bosch_additional"
    additional_rows = yaml.safe_load((additional / "additional_train.yaml").read_text())
    for row in additional_rows:
        relative = row["path"].removeprefix("./")
        sequence = Path(relative).parent.name
        split = "val" if int(hashlib.sha256(sequence.encode()).hexdigest()[:8], 16) % 7 == 0 else "train"
        boxes = []
        for original in row["boxes"]:
            color, direction = public_semantics(original["label"])
            boxes.append({"color": color, "direction": direction,
                          "box": [original["x_min"], original["y_min"], original["x_max"], original["y_max"]]})
        add_frame(additional / relative, boxes, split, f"bosch_{sequence}", "bosch_additional")
    negative_review = public / "background_review"
    if (negative_review / "annotations.csv").is_file():
        negative_rows = {r["id"]: r for r in json.loads((negative_review / "candidates.json").read_text())}
        frames_by_source = {r["source"]: r for r in frame_manifest}
        for annotation in csv.DictReader((negative_review / "annotations.csv").open(encoding="utf-8")):
            if annotation["direction"] != "other" or not annotation["reviewer"]:
                continue
            row = negative_rows[annotation["id"]]
            frame = frames_by_source[row["source"]]
            if frame["split"] != "train":
                continue
            shutil.copyfile(negative_review / row["crop"], review / row["crop"])
            rows.append(row)
            annotations.append({"id": row["id"], "direction": "other", "split": "train",
                                "group": frame["group"], "reviewer": annotation["reviewer"]})
    (review / "candidates.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with (review / "annotations.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("id", "direction", "split", "group", "reviewer"))
        writer.writeheader()
        writer.writerows(annotations)
    (output / "frame_manifest.json").write_text(json.dumps(frame_manifest, indent=2), encoding="utf-8")
    (output / "sources.json").write_text(json.dumps({
        "atlas": "https://zenodo.org/records/14846709", "bosch": "https://zenodo.org/records/12706046",
        "public_split": "ATLAS whole UTC recording days; Bosch whole bag names, shared between primary/additional archives.",
        "legacy_split": "Retained original S2TLD splits; road independence is not established.",
        "license": "Non-commercial research only; retain publisher licenses and attribution.",
        "frames": {split: dict(Counter(r['origin'] for r in frame_manifest if r['split'] == split))
                   for split in ('train', 'val', 'test')}}, indent=2), encoding="utf-8")
    (detection / "data.yaml").write_text(yaml.safe_dump({"path": str(detection.resolve()), "train": "images/train",
                                                       "val": "images/val", "names": list(color_ids)}, sort_keys=False),
                                         encoding="utf-8")
    hyp = yaml.safe_load((Path(__file__).parent / "data/hyps/hyp.scratch-low.yaml").read_text(encoding="utf-8"))
    hyp.update(lr0=.0005, lrf=.1, hsv_h=0., hsv_s=.2, hsv_v=.25, degrees=3., scale=.3, mosaic=.2, mixup=0.)
    (detection / "hyp.yaml").write_text(yaml.safe_dump(hyp, sort_keys=False), encoding="utf-8")
    build(review, "dataset_gray", grayscale=True, mirror_directions=True, resume_preparation=resume_preparation)
    print(f"Prepared mixed training data: {output}", flush=True)


def download_public(output, datasets):
    """Download publisher archives with resumable transfers and publisher checksums."""
    import requests

    def checksum(path):
        digest = hashlib.md5()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    catalog = {"bosch-additional": ("12706046", "dataset_additional_rgb.zip"),
               "bosch-train": ("12706046", "dataset_train_rgb.zip"),
               "atlas-sample": ("14846709", "ATLAS_sample.zip")}
    output.mkdir(parents=True, exist_ok=True)
    for name in datasets:
        record, filename = catalog[name]
        response = requests.get(f"https://zenodo.org/api/records/{record}", timeout=60)
        response.raise_for_status()
        metadata = response.json()
        item = next(entry for entry in metadata["files"] if entry["key"] == filename)
        destination, partial = output / filename, output / f"{filename}.part"
        expected = item["checksum"].split(":", 1)[1]
        if not destination.exists():
            for attempt in range(3):
                try:
                    offset = partial.stat().st_size if partial.exists() else 0
                    headers = {"Range": f"bytes={offset}-"} if offset else {}
                    with requests.get(item["links"]["self"], headers=headers, stream=True,
                                      timeout=(30, 90)) as transfer:
                        transfer.raise_for_status()
                        if transfer.status_code != 206:
                            offset = 0
                        elif not transfer.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                            raise RuntimeError("Server returned an unexpected resume position")
                        last_report = time.monotonic()
                        with partial.open("ab" if offset else "wb") as stream:
                            for chunk in transfer.iter_content(1024 * 1024):
                                stream.write(chunk)
                                offset += len(chunk)
                                if time.monotonic() - last_report >= 10:
                                    print(f"{filename}: {offset / item['size']:.1%} "
                                          f"({offset / 1048576:.0f} MiB)", flush=True)
                                    last_report = time.monotonic()
                    if partial.stat().st_size != item["size"]:
                        raise RuntimeError(f"Incomplete download: {filename}")
                    break
                except (requests.RequestException, RuntimeError) as error:
                    if attempt == 2:
                        raise
                    print(f"Retrying {filename}: {error}", flush=True)
                    time.sleep(2)
            if checksum(partial) != expected:
                raise ValueError(f"Publisher checksum mismatch: {filename}")
            partial.rename(destination)
        else:
            if checksum(destination) != expected:
                raise ValueError(f"Existing archive checksum mismatch: {filename}")
        (output / f"{name}_source.json").write_text(json.dumps({
            "source": f"https://zenodo.org/records/{record}", "title": metadata["metadata"]["title"],
            "license": metadata["metadata"].get("license"), "file": item,
            "usage": "Local non-commercial research; retain publisher attribution and license."}, indent=2),
            encoding="utf-8")
        print(f"Verified publisher archive: {destination}", flush=True)


def square_crop(image, box, size=128):
    """Keep the whole lamp, including arrows at the top of a vertical housing."""
    return ImageOps.pad(image.crop(tuple(box)), (size, size), color=(114, 114, 114))


def review_sheets(rows, output):
    for start in range(0, len(rows), 80):
        sheet = Image.new("RGB", (1280, 1440), "#20232a")
        draw = ImageDraw.Draw(sheet)
        for index, row in enumerate(rows[start : start + 80]):
            x, y = (index % 8) * 160, (index // 8) * 144
            crop = Image.open(output / row["crop"])
            sheet.paste(crop, (x + 16, y + 15))
            draw.text((x + 3, y + 1), f'{row["id"]} {row["color"]}', fill="white")
        sheet.save(output / f"review_{start // 80 + 1:02d}.jpg", quality=95)


def prepare(root, output, count, seed):
    output.mkdir(parents=True, exist_ok=False)
    (output / "crops").mkdir()
    rng = random.Random(seed)
    sources = []
    for split, amount in (("train", count), ("val", count // 2)):
        paths = sorted((root / "images" / split).glob("*.jpg"))
        original = [p for p in paths if not p.stem.startswith("real")]
        rng.shuffle(original)
        sources.extend((split, p) for p in original[:amount])
        if split == "train":
            for prefix in ("real1_", "real2_"):
                video = [p for p in paths if p.stem.startswith(prefix)]
                sources.extend((split, p) for p in video[::max(1, len(video) // 16)])
    rows = []
    for split, path in sources:
        label = root / "labels" / split / f"{path.stem}.txt"
        if not label.exists():
            continue
        if path.stem.startswith(("real1_", "real2_")):
            import cv2
            video_id = path.stem.split("_")[1]
            frame_index = int(path.stem.split("_")[-1])
            source_path = root / "raw_video" / f"{video_id}.mp4"
            cap = cv2.VideoCapture(str(source_path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            cap.release()
            if not ok:
                raise RuntimeError(f"Cannot decode {source_path}, frame {frame_index}")
            source = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            with Image.open(path) as original:
                if original.size != source.size:
                    raise ValueError(f"Source/annotation dimensions differ: {path}")
        else:
            source_path, frame_index = path, None
            source = Image.open(path)
        with source:
            source = source.convert("RGB")
            width, height = source.size
            for line_index, line in enumerate(label.read_text().splitlines()):
                cls, x, y, w, h = map(float, line.split())
                if int(cls) == 3 or w * width < 12 or h * height < 16:
                    continue
                box = [max(0, int((x - w / 2) * width)), max(0, int((y - h / 2) * height)),
                       min(width, int((x + w / 2) * width)), min(height, int((y + h / 2) * height))]
                sample_id = f"{len(rows):04d}"
                crop = f"crops/{sample_id}.png"
                square_crop(source, box).save(output / crop)
                rows.append({"id": sample_id, "source": str(source_path.resolve()), "source_split": split,
                             "annotation_image": str(path.resolve()), "frame_index": frame_index,
                             "label_line": line_index, "box": box, "color": ("red", "yellow", "green")[int(cls)],
                             "crop": crop, "source_sha256": hashlib.sha256(source.tobytes()).hexdigest()})
    (output / "candidates.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with (output / "annotations.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("id", "direction", "split", "group", "reviewer"))
        writer.writeheader()
        writer.writerows({"id": r["id"], "direction": "", "split": "", "group": "", "reviewer": ""} for r in rows)
    review_sheets(rows, output)
    print(json.dumps({"candidates": len(rows), "output": str(output)}, indent=2))


def build(output, dataset_name="dataset", grayscale=False, mirror_directions=False, resume_preparation=False):
    rows = {r["id"]: r for r in json.loads((output / "candidates.json").read_text())}
    with (output / "annotations.csv").open(encoding="utf-8") as stream:
        annotations = list(csv.DictReader(stream))
    if Path(dataset_name).name != dataset_name or dataset_name in (".", ".."):
        raise ValueError("dataset-name must be one directory name")
    dataset = output / dataset_name
    if dataset.exists() and not resume_preparation:
        raise FileExistsError(f"Use a new dataset output; refusing to overwrite {dataset}")
    accepted = []
    groups = {}
    hashes = {}
    for annotation in annotations:
        direction, split, group = annotation["direction"], annotation["split"], annotation["group"]
        if not direction or direction == "skip":
            continue
        if direction not in CLASSES or split not in ("train", "val", "test") or not group or not annotation["reviewer"]:
            raise ValueError(f"Incomplete annotation: {annotation}")
        if group in groups and groups[group] != split:
            raise ValueError(f"Source group crosses splits: {group}")
        groups[group] = split
        row = {**rows[annotation["id"]], **annotation}
        digest = row["source_sha256"]
        if digest in hashes and hashes[digest] != split:
            raise ValueError("The same source image appears in different splits")
        hashes[digest] = split
        accepted.append(row)
    for split in sorted({row["split"] for row in accepted}):
        for direction in CLASSES:
            (dataset / split / direction).mkdir(parents=True, exist_ok=resume_preparation)
    for row in accepted:
        source = output / row["crop"]
        destination = dataset / row["split"] / row["direction"]
        crop = Image.open(source).convert("RGB")
        if grayscale:
            crop = crop.convert("L").convert("RGB")
        save_prepared_image(crop, destination / f'{row["id"]}.png')
        if row["split"] == "train":
            variants = [(crop, destination, row["id"])]
            if mirror_directions and row["direction"] in ("left", "right"):
                opposite = "right" if row["direction"] == "left" else "left"
                mirrored = ImageOps.mirror(crop)
                opposite_dir = dataset / "train" / opposite
                save_prepared_image(mirrored, opposite_dir / f'{row["id"]}_mirror.png')
                variants.append((mirrored, opposite_dir, f'{row["id"]}_mirror'))
            for variant, folder, stem in variants:
                for index, (brightness, angle, blur) in enumerate(((0.65, -3, 0), (1.3, 3, 0), (0.9, 0, 0.6), (1.1, -2, 0.3))):
                    augmented = ImageEnhance.Brightness(variant).enhance(brightness)
                    augmented = augmented.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(114, 114, 114))
                    if blur:
                        augmented = augmented.filter(ImageFilter.GaussianBlur(blur))
                    save_prepared_image(augmented, folder / f'{stem}_aug{index}.png')
    (output / "manifest.json").write_text(json.dumps(accepted, indent=2), encoding="utf-8")
    report = {split: dict(Counter(r["direction"] for r in accepted if r["split"] == split)) for split in ("train", "val", "test")}
    report["preprocessing"] = {"size": 128, "grayscale": grayscale, "mirror_directions": mirror_directions}
    (dataset / "manifest.json").write_text(json.dumps(accepted, indent=2), encoding="utf-8")
    (dataset / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "dataset_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def video_review(root, output, videos):
    """Propose boxes on held-out videos; direction labels still require visual review."""
    import cv2
    import numpy as np
    import torch
    from types import SimpleNamespace
    from models.common import DetectMultiBackend
    from traffic_light_3way_v2_modified import detect_roi, second_nms

    output.mkdir(parents=True, exist_ok=False)
    (output / "crops").mkdir()
    model = DetectMultiBackend(str(root / "yolov5/runs/train/traffic_real_v2/weights/best.pt"),
                               device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    args = SimpleNamespace(imgsz=960, conf_thres=0.15, iou_thres=0.45)
    rows = []
    with torch.inference_mode():
        for video in videos:
            path = root / "raw_video" / f"{video}.mp4"
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open {path}")
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for frame_index in np.linspace(0, total - 1, 20, dtype=int):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f"Cannot decode {path}:{frame_index}")
                h, w = frame.shape[:2]
                tile = int(w * 0.62)
                detections = detect_roi(model, frame[:int(h * 0.75), :tile], args)
                detections += detect_roi(model, frame[:int(h * 0.75), w - tile:], args, w - tile)
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                for detection in second_nms(detections):
                    x1, y1, x2, y2, confidence, cls = detection
                    if x2 - x1 < 12 or y2 - y1 < 16 or cls == 3:
                        continue
                    sample_id = f"{len(rows):04d}"
                    crop = f"crops/{sample_id}.png"
                    square_crop(image, (x1, y1, x2, y2)).save(output / crop)
                    rows.append({"id": sample_id, "source": str(path.resolve()), "source_split": "test",
                                 "frame_index": int(frame_index), "box": [x1, y1, x2, y2],
                                 "color": str(model.names[cls]), "crop": crop,
                                 "source_sha256": hashlib.sha256(image.tobytes()).hexdigest()})
            cap.release()
    (output / "candidates.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with (output / "annotations.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("id", "direction", "split", "group", "reviewer"))
        writer.writeheader()
        writer.writerows({"id": r["id"], "direction": "", "split": "test",
                          "group": f'video_{Path(r["source"]).stem}', "reviewer": ""} for r in rows)
    review_sheets(rows, output)
    print(json.dumps({"candidates": len(rows), "videos": videos}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=220)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--dataset-name", default="dataset")
    parser.add_argument("--grayscale", action="store_true")
    parser.add_argument("--mirror-directions", action="store_true")
    parser.add_argument("--video-review", nargs="+", help="held-out video stems, e.g. 4 5")
    parser.add_argument("--download-public", nargs="+",
                        choices=("bosch-additional", "bosch-train", "atlas-sample"))
    parser.add_argument("--fetch-atlas", type=int, help="publisher training image count; adds grouped validation and test")
    parser.add_argument("--fetch-bosch", type=int, help="publisher training images to fetch by range")
    parser.add_argument("--unpack-bosch", type=Path, help="publisher Deflate64 archive")
    parser.add_argument("--tool-deps", type=Path, help="optional isolated archive decoder directory")
    parser.add_argument("--prepare-generalization", type=Path, help="public data root for new mixed datasets")
    parser.add_argument("--available-public", action="store_true", help="pin a snapshot of already acquired public frames")
    parser.add_argument("--resume-preparation", action="store_true", help="resume interrupted generation of the same pinned dataset")
    parser.add_argument("--mine-negatives", type=Path, help="public data root to propose training-only background crops")
    args = parser.parse_args()
    if args.tool_deps:
        import sys

        sys.path.insert(0, str(args.tool_deps.resolve()))
    if args.mine_negatives:
        mine_negatives(args.root, args.mine_negatives, args.output)
    elif args.prepare_generalization:
        prepare_generalization(args.root, args.prepare_generalization, args.output, args.available_public, args.resume_preparation)
    elif args.unpack_bosch:
        unpack_bosch(args.unpack_bosch, args.output)
    elif args.fetch_bosch:
        fetch_bosch(args.output, args.fetch_bosch)
    elif args.fetch_atlas:
        fetch_atlas(args.output, args.fetch_atlas)
    elif args.download_public:
        download_public(args.output, args.download_public)
    elif args.video_review:
        video_review(args.root, args.output, args.video_review)
    elif args.build:
        build(args.output, args.dataset_name, args.grayscale, args.mirror_directions)
    else:
        prepare(args.root, args.output, args.count, args.seed)
