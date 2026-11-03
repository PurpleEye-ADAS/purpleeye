"""Color and arrow recognition, with explicit signal-group calibration."""

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from functools import wraps, lru_cache
from pathlib import Path
from threading import Condition, Thread
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image

from prepare_direction_data import CLASSES, square_crop
from traffic_rknn import source_scene_changed

# The board uses its existing RKNN runtime and does not require PyTorch.
torch = DetectMultiBackend = classify_transforms = letterbox = None
check_img_size = non_max_suppression = scale_boxes = select_device = None


def load_pc_runtime():
    global torch, DetectMultiBackend, classify_transforms, letterbox
    global check_img_size, non_max_suppression, scale_boxes, select_device
    import importlib

    for module_name, names in (
        ("torch", ("torch",)),
        ("models.common", ("DetectMultiBackend",)),
        ("utils.augmentations", ("classify_transforms", "letterbox")),
        ("utils.general", ("check_img_size", "non_max_suppression", "scale_boxes")),
        ("utils.torch_utils", ("select_device",)),
    ):
        module = importlib.import_module(module_name)
        for name in names:
            if globals()[name] is None:
                globals()[name] = module if name == "torch" else getattr(module, name)

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
DIRECTIONS = ("LEFT", "STRAIGHT", "RIGHT")
CONTROLS = ("LEFT", "STRAIGHT", "RIGHT", "STOP", "MOVE", "AUTO")
INPUT_CHANNELS = {"CAM1": 0, "CAM2": 1, "HDMI": 2}
CONTROL_PANEL_HEIGHT = 70
NATIVE_SIGNALS = None
KEY_ACTIONS = {ord("a"): "LEFT", ord("w"): "STRAIGHT", ord("d"): "RIGHT", ord("s"): "STOP",
               ord("f"): "MOVE", ord("r"): "AUTO", ord(" "): "TOGGLE"}


class VehicleSimulation:
    """Keep simulated vehicle motion separate from movement recorded in the input video."""

    def __init__(self, nav="STRAIGHT", force_moving=False):
        if nav not in DIRECTIONS:
            raise ValueError(f"Unsupported navigation direction: {nav}")
        self.nav = nav
        self.requested_nav = nav
        self.manual_moving = True if force_moving else None
        self.moving = False
        self.alarm = False
        self.decision = "WAIT_UNKNOWN"

    @property
    def mode(self):
        return "AUTO" if self.manual_moving is None else "MANUAL"

    @property
    def guidance_visible(self):
        return self.moving and self.decision == f"GO_{self.nav}"

    def command(self, action):
        if action in DIRECTIONS:
            self.nav = self.requested_nav = action
        elif action == "AUTO":
            self.manual_moving = None
        elif action == "MOVE":
            self.manual_moving = True
        elif action == "STOP":
            self.manual_moving = False
        elif action == "TOGGLE":
            self.manual_moving = not (self.manual_moving if self.manual_moving is not None else self.moving)
        else:
            raise ValueError(f"Unsupported control: {action}")

    def update(self, states, calibrated, action=None):
        if action == "TOGGLE":
            self.moving = ((calibrated and states.get(self.nav) == "green")
                           if self.manual_moving is None else self.manual_moving)
        if action is not None:
            self.command(action)
        state = states.get(self.nav, "UNKNOWN")
        self.moving = (calibrated and state == "green") if self.manual_moving is None else self.manual_moving
        self.decision = make_decision(self.nav, states, calibrated, self.moving)
        if calibrated and state == "green" and not self.moving:
            self.decision = "STOP_MANUAL"
        self.alarm = self.decision == "ALARM_RED"
        return self.decision

    def snapshot(self):
        leds = [int(self.moving and self.nav == direction) for direction in DIRECTIONS]
        return {"mode": self.mode, "navigation": self.nav, "moving": self.moving,
                "alarm": self.alarm, "guidance_visible": self.guidance_visible,
                "simulated_leds": leds + [int(self.alarm)]}


def control_rectangles(shape, actions=CONTROLS):
    height, width = shape[:2]
    return [(action, (int(i * width / len(actions)) + 3, height - 59,
                     int((i + 1) * width / len(actions)) - 3, height - 9))
            for i, action in enumerate(actions)]


def handle_mouse(event, x, y, flags, context):
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    simulation, source_shape, preview_shape = context[:3]
    requested_input = context[3] if len(context) == 4 else None
    actions = tuple(getattr(simulation, 'controls', CONTROLS))
    if requested_input is not None:
        actions += tuple(INPUT_CHANNELS)
    source_x = x * source_shape[1] / preview_shape[1]
    source_y = y * source_shape[0] / preview_shape[0]
    for action, (x1, y1, x2, y2) in control_rectangles(source_shape, actions):
        if x1 <= source_x < x2 and y1 <= source_y < y2:
            if action in INPUT_CHANNELS:
                requested_input[0] = INPUT_CHANNELS[action]
            else:
                simulation.command(action)
            return


class StableState:
    """Missing/conflicting observations invalidate old state; green needs confirmation."""

    def __init__(self, confirm_frames=3):
        if confirm_frames < 1:
            raise ValueError("confirm_frames must be positive")
        self.history = deque(maxlen=confirm_frames)
        self.stable = "UNKNOWN"

    def update(self, state):
        if state not in ("red", "yellow", "green"):
            self.history.clear()
            self.stable = "UNKNOWN"
            return self.stable
        self.history.append(state)
        if state in ("red", "yellow"):
            self.stable = state
        elif len(self.history) == self.history.maxlen and len(set(self.history)) == 1:
            self.stable = state
        elif state != self.stable:
            self.stable = "UNKNOWN"
        return self.stable


def direction_states(lights, circle_direction=None):
    scores = {direction: {} for direction in DIRECTIONS}
    for light in lights:
        if not light.get("in_signal_group", True):
            continue
        direction = light["direction"].upper()
        if direction == "CIRCLE":
            direction = circle_direction
        if direction in scores and light["color"] in ("red", "yellow", "green", "off"):
            color = light["color"]
            confidence = min(max(light.get("color_conf", 1.0), 0.0), 0.999)
            previous = scores[direction].get(color, 0.0)
            scores[direction][color] = 1 - (1 - previous) * (1 - confidence)
    states = {}
    for direction, values in scores.items():
        ordered = sorted(values, key=values.get, reverse=True)
        states[direction] = ordered[0] if ordered else None
        if len(ordered) > 1 and (ordered[0] == "green" or values[ordered[0]] - values[ordered[1]] < 0.15):
            states[direction] = None
    return states


def changed_camera_channels(previous, current, previous_mask, current_mask):
    return {channel for channel in (0,1) if previous[channel]!=current[channel] or
            (previous_mask^current_mask)&(1<<channel)}


def select_fpga_control(*simulations):
    """Summarize hardware alarms without rewriting any source's perception or decision."""
    return next((item for item in simulations if item.alarm and item.decision=='BRAKE_PEDESTRIAN'),
        next((item for item in simulations if item.alarm),
             next((item for item in simulations if item.guidance_visible),simulations[0])))


def make_decision(nav, stable_states, calibrated=False, moving=False):
    if nav not in DIRECTIONS:
        raise ValueError(f"Unsupported navigation direction: {nav}")
    if not calibrated:
        return "WAIT_CALIBRATE_ROI"
    state = stable_states.get(nav, "UNKNOWN")
    if state == "red":
        return "ALARM_RED" if moving else "STOP_RED"
    if state == "yellow":
        return "WAIT_YELLOW"
    return f"GO_{nav}" if state == "green" else "WAIT_UNKNOWN"


def box_iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def camera_housing_overlap(a,b):
    aw,bw=a[2]-a[0],b[2]-b[0]
    ah,bh=a[3]-a[1],b[3]-b[1]
    if min(aw,bw,ah,bh)<=0 or max(aw,bw)>2*min(aw,bw):return False
    if abs((a[0]+a[2]-b[0]-b[2])/2)>.2*max(aw,bw):return False
    intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    return intersection/min(aw*ah,bw*bh)>=.7


def second_nms(detections, threshold=0.45):
    keep = []
    for detection in sorted(detections, key=lambda value: value[4], reverse=True):
        if all(box_iou(detection, previous) <= threshold for previous in keep):
            keep.append(detection)
    return keep


def validate_roi(roi, bounds=None):
    if roi is not None and not (0 <= roi[0] < roi[2] <= 1 and 0 <= roi[1] < roi[3] <= 1):
        raise ValueError("signal-roi requires normalized, increasing x1 y1 x2 y2")
    if roi is not None and bounds is not None and not (
            bounds[0] <= roi[0] < roi[2] <= bounds[2] and bounds[1] <= roi[1] < roi[3] <= bounds[3]):
        raise ValueError("The signal group must stay inside the selected front channel")


def signal_context_bounds(shape, roi):
    """Keep housing context around the selected group, with even bounds for the frame cache."""
    height, width = shape[:2]
    return [max(0, int(roi[0] * width) - 64) // 2 * 2,
            max(0, int(roi[1] * height) - 64) // 2 * 2,
            min(width, int(np.ceil(roi[2] * width)) + 65) // 2 * 2,
            min(height, int(np.ceil(roi[3] * height)) + 65) // 2 * 2]


def preview_dimensions(shape, max_width=1024, max_height=600):
    height, width = shape[:2]
    gain = min(1.0, max_width / width, max_height / height)
    return max(1, round(width * gain)), max(1, round(height * gain))


def display_canvas(frame, lights, roi, max_width=1024, max_height=600, resizer=None):
    width, height = preview_dimensions(frame.shape, max_width, max_height)
    canvas_width, canvas_height = max(width, min(640, max_width)), max(height, min(360, max_height))
    dx, dy = (canvas_width - width) // 2, (canvas_height - height) // 2
    picture = (frame if (width, height) == (frame.shape[1], frame.shape[0])
               else (resizer(frame,(width,height)) if resizer is not None else cv2.resize(frame,(width,height))))
    if (width,height)==(canvas_width,canvas_height) and picture is not frame and picture.flags.owndata:
        canvas=picture
    else:
        canvas = cv2.copyMakeBorder(picture, dy, canvas_height-height-dy, dx, canvas_width-width-dx,
                                    cv2.BORDER_CONSTANT, value=(34, 31, 27))
    gx, gy = width / frame.shape[1], height / frame.shape[0]
    scaled_lights = [{**light, "box": [round(light["box"][0] * gx + dx), round(light["box"][1] * gy + dy),
                                      round(light["box"][2] * gx + dx), round(light["box"][3] * gy + dy)]}
                     for light in lights]
    scaled_roi = [(roi[0] * width + dx) / canvas_width, (roi[1] * height + dy) / canvas_height,
                  (roi[2] * width + dx) / canvas_width, (roi[3] * height + dy) / canvas_height] if roi else None
    return canvas, scaled_lights, scaled_roi


def playback_preview(cap, source, folder, fps, shape, max_width, max_height, lossless=False, bounds=None, stride=1):
    """Cache small playback frames or exact decoded frames for fast inference seeks."""
    digest = hashlib.sha256()
    with Path(source).open("rb") as handle:
        for data in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(data)
    size = (shape[1], shape[0]) if lossless else tuple(
        max(2, value // 2 * 2) for value in preview_dimensions(shape, max_width, max_height))
    if bounds:
        size = (bounds[2] - bounds[0], bounds[3] - bounds[1])
    folder.mkdir(parents=True, exist_ok=True)
    suffix = "avi" if lossless else "mp4"
    prefix = "inference_hfyu" if lossless else "preview"
    region = "_" + "x".join(map(str, bounds)) if bounds else ""
    step = f"_s{stride}" if lossless else ""
    path = folder / f"{prefix}_{digest.hexdigest()[:20]}_{size[0]}x{size[1]}{region}{step}.{suffix}"
    if not path.is_file():
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if lossless and (size[0] % 2 or size[1] % 2 or shutil.disk_usage(folder).free < 96 * 1024 ** 2):
            raise RuntimeError("Exact-frame cache requires even dimensions and enough space in the project tmp folder")
        temporary = path.with_name(path.stem + ".part." + suffix)
        writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*("HFYU" if lossless else "mp4v")), fps / stride, size)
        if not writer.isOpened():
            raise RuntimeError("Cannot prepare playback preview")
        frames, first = 0, None
        if not cap.set(cv2.CAP_PROP_POS_FRAMES, 0):
            writer.release()
            temporary.unlink(missing_ok=True)
            raise RuntimeError("Cannot restart source for frame-cache preparation")
        print("Preparing exact-frame inference cache." if lossless else
              "Preparing a small playback copy; original video/model input remain unchanged.", flush=True)
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frames % stride == 0:
                    if bounds:
                        frame = frame[bounds[1]:bounds[3], bounds[0]:bounds[2]]
                    if first is None and lossless:
                        first = frame.copy()
                    writer.write(frame if lossless else cv2.resize(frame, size))
                frames += 1
                if frames % 60 == 0:
                    print(f"Preparing preview {frames}/{total}", flush=True)
                    if lossless and (temporary.stat().st_size > 128 * 1024 ** 2
                                     or shutil.disk_usage(folder).free < 64 * 1024 ** 2):
                        raise RuntimeError("Frame cache reached its project storage budget")
        except Exception:
            writer.release()
            temporary.unlink(missing_ok=True)
            raise
        finally:
            writer.release()
        if frames == 0 or total > 0 and frames != total:
            temporary.unlink(missing_ok=True)
            raise RuntimeError("Preview decoding did not complete")
        if lossless:
            check = cv2.VideoCapture(str(temporary))
            try:
                ok, restored = check.read()
                unchanged = ok and np.array_equal(restored, first)
            finally:
                check.release()
            if not unchanged:
                temporary.unlink(missing_ok=True)
                raise RuntimeError("Lossless cache changed the decoded model pixels")
        temporary.replace(path)
    preview = cv2.VideoCapture(str(path))
    if not preview.isOpened():
        preview.release()
        raise RuntimeError("Cannot open playback preview")
    return preview


def select_signal_group(frame, max_width=1024, max_height=600, bounds=None):
    """Select the complete relevant signal group in a scaled video preview."""
    if bounds is not None:
        height, width = frame.shape[:2]
        frame = frame[round(bounds[1] * height):round(bounds[3] * height),
                      round(bounds[0] * width):round(bounds[2] * width)]
    height, width = frame.shape[:2]
    preview = cv2.resize(frame, preview_dimensions(frame.shape, max_width, max_height))
    title = "Select our signal group: drag rectangle, Enter confirm, C retry, Esc cancel"
    try:
        cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
        x, y, w, h = cv2.selectROI(title, preview, showCrosshair=True, fromCenter=False)
    finally:
        cv2.destroyWindow(title)
    if w == 0 or h == 0:
        return None
    roi = [x / preview.shape[1], y / preview.shape[0],
           (x + w) / preview.shape[1], (y + h) / preview.shape[0]]
    if bounds is not None:
        roi = [bounds[0] + roi[0] * (bounds[2] - bounds[0]),
               bounds[1] + roi[1] * (bounds[3] - bounds[1]),
               bounds[0] + roi[2] * (bounds[2] - bounds[0]),
               bounds[1] + roi[3] * (bounds[3] - bounds[1])]
    validate_roi(roi, bounds)
    return roi


def in_roi(box, shape, roi):
    if roi is None:
        return True
    height, width = shape[:2]
    x, y = (box[0] + box[2]) / (2 * width), (box[1] + box[3]) / (2 * height)
    return roi[0] <= x <= roi[2] and roi[1] <= y <= roi[3]


def filter_signal_group(detections, shape, roi):
    return [detection for detection in detections if in_roi(detection[:4], shape, roi)]


def assign_calibrated_direction(detection, frame_width, centers, max_distance=0.24):
    center = (detection[0] + detection[2]) / (2 * frame_width)
    index = min(range(3), key=lambda i: abs(center - centers[i]))
    return DIRECTIONS[index] if abs(center - centers[index]) <= max_distance else "UNKNOWN"


class DirectionHistory:
    """Confirm arrow identity on the same housing without retaining missing light observations."""

    def __init__(self, confirm_frames=3, max_misses=7):
        self.confirm_frames, self.max_misses = confirm_frames, max_misses
        self.frame = 0
        self.tracks = []
        self.previous_gray = self.previous_shape = self.previous_index = None

    def advance(self, shape, gray, frames=None, motions=None, roi=None):
        """Move the selected group and its housing identities through the same checked motion."""
        scale = [shape[1], shape[0], shape[1], shape[0]]
        region = [{"box": (np.asarray(roi) * scale).tolist()}] if roi is not None else []
        if self.previous_gray is not None and (self.tracks or region):
            sequence = ([item for item in frames if self.previous_index is not None and item[0] >= self.previous_index]
                        if frames else [(None, self.previous_gray), (None, gray)])
            if shape != self.previous_shape or not sequence or frames and sequence[0][0] != self.previous_index:
                self.tracks, region = [], []
            else:
                cache = {}
                for (source_index, source), (current_index, current) in zip(sequence, sequence[1:]):
                    matrix = motions.get((source_index, current_index)) if motions is not None else None
                    if matrix is None:
                        aligned = align_light_boxes(source, current, self.tracks or region, shape, cache)
                        matrix = cache.get("motion") if aligned else None
                    if matrix is None:
                        self.tracks, region = [], []
                        break
                    self.tracks = warp_light_boxes(self.tracks, matrix, shape)
                    region = warp_light_boxes(region, matrix, shape)
        self.previous_gray, self.previous_shape = gray, shape
        self.previous_index = frames[-1][0] if frames else None
        if roi is not None and not region:
            self.tracks = []
        return (np.asarray(region[0]["box"]) / scale).tolist() if region else None

    def update(self, detections, predictions, image=None, frames=None, motions=None, preserve_identity=False):
        self.frame += 1
        self.tracks = [track for track in self.tracks if self.frame - track["last_seen"] <= self.max_misses]
        if image is not None:
            shape = image.shape[:2]
            gray = frames[-1][1] if frames else cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), preview_dimensions(shape, 640, 360))
            self.advance(shape, gray, frames, motions)
        pairs = sorted(((max(box_iou(detection, track["box"]),
                             .7 if preserve_identity and camera_housing_overlap(detection,track['box']) else 0), i, j)
                        for i, detection in enumerate(detections) for j, track in enumerate(self.tracks)), reverse=True)
        matched, used = {}, set()
        for overlap, index, track_index in pairs:
            if overlap >= 0.3 and index not in matched and track_index not in used:
                matched[index] = self.tracks[track_index]
                used.add(track_index)
        results = []
        for index, (detection, (direction, confidence)) in enumerate(zip(detections, predictions)):
            track = matched.get(index)
            if track is None:
                track = {"box": detection[:4], "last_seen": self.frame, "stable": "UNKNOWN", "confidence": 0.0,
                         "history": deque(maxlen=self.confirm_frames)}
                self.tracks.append(track)
            track["box"], track["last_seen"] = detection[:4], self.frame
            if preserve_identity and direction in ('UNKNOWN','CIRCLE') and track['stable'] in DIRECTIONS:
                direction, confidence = track['stable'], track['confidence']
            track["history"].append((direction, confidence))
            if preserve_identity and direction in DIRECTIONS and confidence >= .9:
                track['stable'],track['confidence']=direction,confidence
            elif len(track["history"]) == self.confirm_frames and len({item[0] for item in track["history"]}) == 1:
                track["stable"] = direction
                track["confidence"] = min(item[1] for item in track["history"]) if direction != "UNKNOWN" else 0.0
            elif direction == track["stable"]:
                track["confidence"] = confidence
            results.append((track["stable"], track["confidence"]))
        return results


class DirectionClassifier:
    def __init__(self, weights, device, threshold=0.9, margin=0.25, size=128, runtime=None):
        if not Path(weights).is_file():
            raise FileNotFoundError(f"Direction weights not found: {weights}")
        self.config = json.loads(Path(weights).with_suffix(".json").read_text(encoding="utf-8"))
        if self.config["input_size"] != size:
            raise ValueError("Direction model and crop sizes disagree")
        self.rknn = Path(weights).suffix == ".rknn"
        if self.rknn:
            from traffic_rknn import RKNNModel

            self.model = RKNNModel(weights, runtime)
        else:
            load_pc_runtime()
            self.model = DetectMultiBackend(str(weights), device=device, fp16=False)
        self.names = self.model.names
        if isinstance(self.names, (tuple, list)):
            self.names = dict(enumerate(self.names))
        if set(self.names.values()) != set(CLASSES):
            raise ValueError(f"Unexpected direction classes: {self.names}")
        if [self.names[index] for index in range(len(CLASSES))] != self.config["classes"]:
            raise ValueError("Direction model and metadata class order disagree")
        self.device, self.threshold, self.margin, self.size = device, threshold, margin, size
        self.transform = None if self.rknn else classify_transforms(size)

    def predict(self, frame, detections):
        if not detections:
            return []
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        tensors, indices = [], []
        results = [("UNKNOWN", 0.0) for _ in detections]
        for index, detection in enumerate(detections):
            x1, y1, x2, y2 = detection[:4]
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            crop = square_crop(image, (x1, y1, x2, y2), self.size)
            if self.config["grayscale"]:
                crop = crop.convert("L").convert("RGB")
            crop = np.asarray(crop)[:, :, ::-1].copy()
            if self.rknn:
                from traffic_rknn import normalized_input

                tensors.append(normalized_input(crop[:, :, ::-1], self.config))
            else:
                tensors.append(self.transform(crop))
            indices.append(index)
        if tensors:
            if self.rknn:
                logits = np.concatenate([self.model.infer(tensor)[0] for tensor in tensors])
                probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
                probabilities /= probabilities.sum(axis=1, keepdims=True)
                classes = np.argsort(-probabilities, axis=1)[:, :2]
                scores = np.take_along_axis(probabilities, classes, axis=1)
                scores, classes = scores.tolist(), classes.tolist()
            else:
                batch = torch.stack(tensors).to(self.device)
                logits = self.model(batch) if self.model.pt else torch.cat([self.model(x[None]) for x in batch])
                scores, classes = logits.softmax(1).topk(2, dim=1)
                scores, classes = scores.cpu().tolist(), classes.cpu().tolist()
            for index, values, ids in zip(indices, scores, classes):
                name = self.names[ids[0]]
                accepted = values[0] >= self.threshold and values[0] - values[1] >= self.margin and name != "other"
                results[index] = ("OTHER" if name == "other" and values[0] >= .5 else
                                  name.upper() if accepted else "UNKNOWN", values[0])
        return results


def detect_roi(model, roi, args, offset_x=0, offset_y=0):
    if uses_board_backend(args):
        return model.detect(roi, args, offset_x, offset_y)
    load_pc_runtime()
    size = check_img_size(args.imgsz, s=model.stride)
    image = letterbox(roi, (size, size), stride=model.stride, auto=False)[0]
    tensor = torch.from_numpy(np.ascontiguousarray(image.transpose((2, 0, 1))[::-1])).to(model.device)
    tensor = tensor.half() if model.fp16 else tensor.float()
    prediction = non_max_suppression(model(tensor[None] / 255.0), args.conf_thres, args.iou_thres, max_det=100)
    results = []
    for boxes in prediction:
        if len(boxes):
            boxes[:, :4] = scale_boxes(tensor.shape[1:], boxes[:, :4], roi.shape).round()
            for x1, y1, x2, y2, confidence, cls in boxes.cpu().tolist():
                results.append([int(x1) + offset_x, int(y1) + offset_y,
                                int(x2) + offset_x, int(y2) + offset_y, confidence, int(cls)])
    return results


@lru_cache(maxsize=32)
def _driving_arrow_geometry(height, width, navigation, control_height):
    outline = np.array([[-.22, -1.35], [.22, -1.35], [.22, .35], [.7, .35],
                        [0, 1.4], [-.7, .35], [-.22, .35]], dtype=np.float64)
    if navigation != "STRAIGHT":
        outline = np.array([[-.22, -1.35], [.22, -1.35], [.22, .75], [-.65, .75],
                            [-.65, 1.08], [-1.5, .5], [-.65, -.08], [-.65, .25], [-.22, .25]])
        if navigation == "RIGHT":
            outline[:, 0] *= -1
    lower = np.column_stack((outline, np.zeros(len(outline))))
    upper = lower.copy()
    upper[:, 2] = .30
    vertices = np.vstack((lower, upper))
    # This display pose is illustrative; it is not a calibrated road coordinate system.
    rotation = np.array([2 * np.pi / 3, 0, 0], dtype=np.float64)
    translation = np.array([0, 0, 6], dtype=np.float64)
    projected = cv2.projectPoints(vertices, rotation, translation, np.eye(3), None)[0].reshape(-1, 2)
    extent = np.ptp(projected, axis=0)
    road_top = max(min(124,height//2), min(int(height * .60), height - control_height - 55))
    road_bottom = height - control_height - 13
    scale = min(width * .52 / extent[0], max(12, min(height * .22, road_bottom - road_top)) / extent[1])
    center = (projected.min(axis=0) + projected.max(axis=0)) / 2
    pixels = np.rint((projected - center) * scale + [width / 2, (road_top + road_bottom) / 2]).astype(np.int32)
    count = len(outline)
    camera_vertices = vertices @ cv2.Rodrigues(rotation)[0].T + translation
    faces = [[i, (i + 1) % count, (i + 1) % count + count, i + count] for i in range(count)]
    rendered_faces = []
    for face in sorted(faces, key=lambda indices: camera_vertices[indices, 2].mean(), reverse=True):
        edge = outline[face[1]] - outline[face[0]]
        brightness = .45 + .22 * abs(edge[0]) / np.linalg.norm(edge)
        color = tuple(int(value * brightness) for value in (70, 240, 130))
        rendered_faces.append((pixels[face], color))
    return pixels, count, rendered_faces


def draw_driving_arrow(frame, simulation, control_height=CONTROL_PANEL_HEIGHT):
    """Project an extruded arrow mesh when the selected signal permits motion."""
    if not simulation.guidance_visible:
        return
    height, width = frame.shape[:2]
    pixels, count, faces = _driving_arrow_geometry(height,width,simulation.nav,control_height)
    shadow_points=pixels[:count]+[5,8]
    x,y,w,h=cv2.boundingRect(shadow_points)
    x1,y1,x2,y2=max(0,x-2),max(0,y-2),min(width,x+w+2),min(height,y+h+2)
    if x2>x1 and y2>y1:
        target=frame[y1:y2,x1:x2]
        shadow=target.copy()
        cv2.fillPoly(shadow,[shadow_points-[x1,y1]],(12,20,15),cv2.LINE_AA)
        cv2.addWeighted(shadow,.40,target,.60,0,target)
    for points, color in faces:
        cv2.fillConvexPoly(frame, points, color, cv2.LINE_AA)
        cv2.polylines(frame, [points], True, (32, 100, 55), 1, cv2.LINE_AA)
    cv2.fillPoly(frame, [pixels[count:]], (70, 240, 130), cv2.LINE_AA)
    cv2.polylines(frame, [pixels[count:]], True, (160, 255, 195), max(1, width // 500), cv2.LINE_AA)


def draw_detections(frame, lights):
    colors = {"red": (20, 20, 255), "yellow": (0, 220, 255), "green": (0, 220, 0), "off": (180, 180, 180)}
    for index, light in enumerate(lights):
        if not light.get("in_signal_group", True):
            continue
        x1, y1, x2, y2 = map(round, light["box"])
        color = light.get('box_color',colors.get(light["color"], (255, 255, 255)))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        direction = "PENDING" if light["direction"] == "UNKNOWN" else light["direction"]
        label = (f'{light["color"]} / {direction}' if light.get("color_source") == "current_frame" else
                 f'{light["color"]} {light["color_conf"]:.2f} / {direction}')
        cv2.putText(frame, label, (max(0, min(x1, frame.shape[1] - 180)), max(14, y1 - 5 - 14 * (index % 3))),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)


def draw_model_observation(result, width, height, now, max_age):
    panel = np.full((height, width, 3), (34, 31, 27), dtype=np.uint8)
    cv2.putText(panel, "MODEL OBSERVATION", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, .48, (240, 240, 240), 1)
    if result is None:
        cv2.putText(panel, "Waiting for first inference...", (10, 57), cv2.FONT_HERSHEY_SIMPLEX, .4, (230, 230, 230), 1)
        return panel
    age = now - result["captured_at"]
    status = "STALE - NO PERMISSION" if age > max_age else "RECENT OBSERVATION"
    cv2.putText(panel, f"Frame {result['frame'] + 1} | age {age:.1f}s", (10, 48),
                cv2.FONT_HERSHEY_SIMPLEX, .4, (230, 230, 230), 1)
    cv2.putText(panel, status, (10, 71), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 190, 255), 1)
    frame = result["image"]
    roi = result["signal_roi"] or (0, 0, 1, 1)
    fh, fw = frame.shape[:2]
    x1, y1, x2, y2 = (int(roi[0] * fw), int(roi[1] * fh), int(np.ceil(roi[2] * fw)), int(np.ceil(roi[3] * fh)))
    crop = frame[y1:y2, x1:x2]
    cw, ch = preview_dimensions(crop.shape, width - 16, height - 122)
    gain_x, gain_y = cw / crop.shape[1], ch / crop.shape[0]
    picture = cv2.resize(crop, (cw, ch))
    lights = [{**light, "box": [(light["box"][0] - x1) * gain_x, (light["box"][1] - y1) * gain_y,
                               (light["box"][2] - x1) * gain_x, (light["box"][3] - y1) * gain_y]}
              for light in result["lights"]]
    draw_detections(picture, lights)
    panel[86:86 + ch, 8:8 + cw] = picture
    cv2.putText(panel, f"Detected: {len(lights)} | {result['processing_ms']:.0f} ms", (10, height - 15),
                cv2.FONT_HERSHEY_SIMPLEX, .4, (230, 230, 230), 1)
    return panel


@lru_cache(maxsize=64)
def _control_panel(width, navigation, mode, manual_moving, controls, input_channel=None):
    panel=np.full((CONTROL_PANEL_HEIGHT,width,3),(34,31,27),np.uint8)
    for action,(x1,y1,x2,y2) in control_rectangles(panel.shape,controls):
        selected=(action==navigation or action=='AUTO' and mode=='AUTO' or
                  action=='MOVE' and manual_moving is True or action=='STOP' and manual_moving is False or
                  action in INPUT_CHANNELS and INPUT_CHANNELS[action] == (2 if input_channel == 3 else input_channel))
        cv2.rectangle(panel,(x1,y1),(x2,y2),(110,78,34) if selected else (64,60,55),-1)
        cv2.putText(panel,action,(x1+8,y1+31),cv2.FONT_HERSHEY_SIMPLEX,
                    min(.53,width/1370),(250,250,250),1)
    return panel


def draw_overlay(frame, lights, states, simulation, roi, progress=None, guidance_bounds=None, video_shape=None,
                 mirror_bounds=(), input_channel=None):
    nav = simulation.nav
    draw_detections(frame, lights)
    height, width = frame.shape[:2]
    guidance_frame = frame
    if guidance_bounds is not None:
        content_width, content_height = preview_dimensions(video_shape or frame.shape, width, height)
        dx, dy = (width - content_width) // 2, (height - content_height) // 2
        x1, y1, x2, y2 = (round(guidance_bounds[0] * content_width + dx),
                          round(guidance_bounds[1] * content_height + dy),
                          round(guidance_bounds[2] * content_width + dx),
                          round(guidance_bounds[3] * content_height + dy))
        guidance_frame = frame[y1:y2, x1:x2]
    draw_driving_arrow(guidance_frame,simulation,0 if guidance_bounds is not None else CONTROL_PANEL_HEIGHT)
    for bounds in mirror_bounds:
        x1, y1, x2, y2 = (round(bounds[0] * content_width + dx), round(bounds[1] * content_height + dy),
                          round(bounds[2] * content_width + dx), round(bounds[3] * content_height + dy))
        frame[y1:y2, x1:x2] = (guidance_frame if guidance_frame.shape[:2]==(y2-y1,x2-x1) else
                               cv2.resize(guidance_frame,(x2-x1,y2-y1)))
    controls = tuple(getattr(simulation, 'controls', CONTROLS))
    if input_channel is not None:
        controls += tuple(INPUT_CHANNELS)
    panel=_control_panel(width,nav,simulation.mode,simulation.manual_moving,controls,input_channel)
    frame[-CONTROL_PANEL_HEIGHT:]=panel[-min(height,CONTROL_PANEL_HEIGHT):]
    if simulation.alarm:
        cv2.rectangle(guidance_frame, (3, 3), (guidance_frame.shape[1] - 4, guidance_frame.shape[0] - 4), (0, 0, 255), 6)


def uses_board_backend(args):
    color = Path(args.weights).suffix
    return color == ".rknn" or (color == ".onnx" and Path(args.direction_weights).suffix == ".rknn")


def inference_runtime(function):
    @wraps(function)
    def invoke(args):
        if uses_board_backend(args):
            return function(args)
        load_pc_runtime()
        with torch.inference_mode():
            return function(args)
    return invoke


class LatestFrameInference:
    """One pending frame replaces old work; inference never blocks the UI thread."""

    def __init__(self, analyze, max_fps=0):
        self.analyze = analyze
        self.condition = Condition()
        self.pending = self.result = self.error = None
        self.stopped = False
        self.processing = False
        self.interval = 1/max_fps if max_fps else 0
        self.next_start = 0
        self.thread = Thread(target=self._run, name="traffic-inference")
        self.thread.start()

    def submit(self, job):
        with self.condition:
            self.pending = job
            self.condition.notify()

    def ready(self):
        with self.condition:
            return not self.processing and self.pending is None and time.perf_counter() >= self.next_start

    def take(self):
        with self.condition:
            if self.error is not None:
                raise RuntimeError("Traffic inference failed") from self.error
            result, self.result = self.result, None
            return result

    def _run(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.stopped or self.pending is not None)
                if self.stopped:
                    return
                delay = self.next_start - time.perf_counter()
                if delay > 0:
                    self.condition.wait(delay)
                    continue
                job, self.pending = self.pending, None
                self.processing = True
                self.next_start = time.perf_counter() + self.interval
            try:
                with torch.inference_mode() if torch is not None else nullcontext():
                    result = self.analyze(*job)
            except Exception as error:
                with self.condition:
                    self.error = error
                return
            with self.condition:
                self.result = result
                self.processing = False

    def close(self):
        with self.condition:
            self.stopped = True
            self.pending = None
            self.condition.notify()
        self.thread.join()


def native_capture_available(status, channel):
    if status is None or not status['full_hdmi_capture']:
        return False
    if status.get('full_camera_capture'):
        return status['native_ready'] and status['native_source'] == min(channel, 2)
    return channel in (2, 3)


def update_native_preview(mosaic, picture, channel):
    frame = mosaic.copy()
    reduced = picture[::2, ::2]
    if channel in (0, 1):
        frame[:360, channel*640:(channel+1)*640] = reduced
    else:
        frame[360:, :640] = reduced
        frame[360:, 640:] = reduced
    return frame


class LiveFrameCapture:
    """Overlap PCIe reads with processing, keeping pixels and status in one owned packet."""

    def __init__(self, capture, signal_worker, front_channel, prepare=None, both_cameras=False, include_hdmi=False):
        self.capture = capture
        ok, first = capture.read()
        if not ok:
            raise RuntimeError('Cannot capture the initial live mosaic')
        self.mosaic = first.copy()
        self.signal_worker, self.front_channel = signal_worker, front_channel
        self.both_cameras, self.next_native_channel = both_cameras, front_channel
        self.native_channels=(0,1,2) if include_hdmi else (0,1)
        self.frames = self.captured_frames = 0
        self.condition = Condition()
        self.raw_packets = {'latest': None, 'native': None}
        self.ready_packets = {'latest': None, 'native': None} if prepare is not None else self.raw_packets
        self.prepare, self.error, self.native_inflight = prepare, None, False
        self.stopped = False
        self.thread = Thread(target=self._run, name='traffic-capture')
        self.prepare_thread = Thread(target=self._prepare, name='traffic-motion') if prepare is not None else None
        self.thread.start()
        if self.prepare_thread is not None:
            self.prepare_thread.start()

    def _capture(self):
        if self.both_cameras:
            channels=getattr(self,'native_channels',(0,1))
            mask=self.capture.video_status['valid_mask']&(7 if 2 in channels else 3)
            if mask and not mask&(1<<self.next_native_channel):
                self.next_native_channel=next(channel for channel in channels if mask&(1<<channel))
        mosaic_hdmi=self.both_cameras and self.next_native_channel==2
        if (self.both_cameras and not mosaic_hdmi and self.capture.capture_mode == 0
                and self.capture.source_channel != self.next_native_channel):
            self.capture.select_native_source(self.next_native_channel)
        native_channel = self.next_native_channel if self.both_cameras else self.front_channel
        full = (not mosaic_hdmi and native_capture_available(self.capture.video_status, native_channel)
                and self.capture.capture_mode == 0 and not self.native_inflight and self.signal_worker.ready())
        ok, picture = self.capture.read(full_hdmi=full)
        if not ok:
            raise RuntimeError('Live PCIe capture ended')
        captured_at = time.perf_counter()
        inference_picture=picture if full else None
        if full:
            frame = update_native_preview(self.mosaic, picture, native_channel)
            if self.both_cameras:
                self.next_native_channel=channels[(channels.index(native_channel)+1)%len(channels)]
        else:
            self.mosaic = picture.copy()
            frame = picture
            if mosaic_hdmi and not self.native_inflight and self.signal_worker.ready():
                inference_picture=picture[360:,:640].copy()
                self.next_native_channel=channels[(channels.index(native_channel)+1)%len(channels)]
        self.captured_frames += 1
        packet = {'frame': frame, 'native': inference_picture,
                  'native_channel': native_channel,
                  'number': self.captured_frames-1, 'captured_at': captured_at,
                  'status': self.capture.video_status, 'dma_ms': self.capture.dma_ms,
                  'convert_ms': self.capture.convert_ms, 'capture_mode': int(full)}
        return packet

    def _run(self):
        while True:
            with self.condition:
                if self.stopped:
                    return
            try:
                packet = self._capture()
            except Exception as error:
                with self.condition:
                    self.error = error
                    self.condition.notify_all()
                return
            with self.condition:
                if packet['native'] is not None:
                    self.native_inflight = True
            self._publish(self.raw_packets, packet)

    def _publish(self, buffer, packet):
        with self.condition:
            buffer['native' if packet['native'] is not None else 'latest'] = packet
            self.condition.notify_all()

    def _take(self, buffer):
        with self.condition:
            self.condition.wait_for(lambda: buffer['native'] is not None or buffer['latest'] is not None
                                    or self.error is not None or self.stopped)
            if self.error is not None:
                raise RuntimeError('Live PCIe capture failed') from self.error
            if self.stopped:
                return None
            packet = buffer['native'] or buffer['latest']
            buffer['native'] = None
            if buffer['latest'] is not None and buffer['latest']['number'] <= packet['number']:
                buffer['latest'] = None
            return packet

    def _prepare(self):
        while True:
            try:
                packet = self._take(self.raw_packets)
                if packet is None:
                    return
                prepared_at=time.perf_counter()
                packet = self.prepare(packet)
                if self.both_cameras:
                    packet['prepare_ms']=(time.perf_counter()-prepared_at)*1000
            except Exception as error:
                with self.condition:
                    self.error = error
                    self.condition.notify_all()
                return
            self._publish(self.ready_packets, packet)

    def read(self):
        packet = self._take(self.ready_packets)
        if packet is None:
            return False, None
        if packet['native'] is not None:
            with self.condition:
                self.native_inflight = False
        self.native_frame, self.frame_id = packet['native'], packet['number']
        self.native_frame_channel = packet['native_channel']
        self.person_observation = packet.get('person_observation')
        self.lane_observation = packet.get('lane_observation')
        self.front_gray = packet.get('front_gray')
        self.camera_lanes, self.camera_people = packet.get('camera_lanes'), packet.get('camera_people')
        self.camera_grays = packet.get('camera_grays')
        self.camera_scene_frames = packet.get('camera_scene_frames',{})
        self.camera_prepare_costs = packet.get('camera_prepare_costs')
        self.prepare_ms=packet.get('prepare_ms',0.)
        self.captured_at, self.video_status = packet['captured_at'], packet['status']
        self.dma_ms, self.convert_ms, self.capture_mode = packet['dma_ms'], packet['convert_ms'], packet['capture_mode']
        self.frames = self.frame_id+1
        return True, packet['frame']

    def get(self, property_id):
        return self.capture.get(property_id)

    def release(self):
        with self.condition:
            self.stopped = True
            self.condition.notify_all()
        self.thread.join()
        if self.prepare_thread is not None:
            self.prepare_thread.join()
        self.capture.release()


class FrameAnalyzer:
    def __init__(self, detector, classifier, device, source_cap=None, source_bounds=None,
                 source_shape=None, source_stride=1, original_cap=None):
        self.detector, self.classifier, self.device = detector, classifier, device
        self.source_cap = source_cap
        self.source_bounds, self.source_shape = source_bounds, source_shape
        self.source_stride, self.original_cap = source_stride, original_cap
        self.generation = None
        self.review_count = 0
        self.live_result = self.live_geometry = None

    def reuse_live_tracks(self, shape, current, native_bounds, display_shape, snapshot=None):
        geometry=(*shape,tuple(native_bounds) if native_bounds is not None else None,
                  tuple(display_shape) if display_shape is not None else None)
        if self.live_result is None or self.live_geometry!=geometry:
            return False
        if snapshot is not None and snapshot[0]!=self.live_result['frame']:
            return False
        observed=snapshot[1] if snapshot is not None else self.live_result.get('_live_alignment')
        if not observed or not observed[1] or not 0<=current-observed[0]<=3:
            return False
        x,y,right,bottom=native_bounds or (0.,0.,1.,1.)
        display_h,display_w=display_shape or shape
        gain=np.array([(right-x)*display_w/shape[1],(bottom-y)*display_h/shape[0]]*2)
        offset=np.array([x*display_w,y*display_h]*2)
        moved={tuple(light['native_box']):(np.asarray(light['box'])-offset)/gain
               for light in observed[1] if 'native_box' in light}
        tracks=[]
        for track in self.direction_history.tracks:
            box=moved.get(tuple(track['box']))
            if box is not None:
                tracks.append(dict(track,box=box.tolist()))
        if not tracks:
            return False
        # The UI already checked this motion on current pixels; do not replay its history.
        self.direction_history.tracks=tracks
        return True

    def detection_bounds(self, shape, roi):
        if getattr(self.detector, 'config', {}).get('detection_context') == 'full_frame':
            return (0, 0, shape[1], shape[0])
        return signal_context_bounds(shape, roi)

    def __call__(self, frame, args, current, loop_index, group_revision, captured_at, frames=None, motions=None,
                 native_bounds=None, display_shape=None):
        tick = time.perf_counter()
        camera_input = getattr(args, 'pcie_capture', False) and getattr(args, 'front_channel', 2) in (0, 1)
        display_roi = args.signal_roi
        if native_bounds is not None:
            args = SimpleNamespace(**vars(args))
            x, y, right, bottom = native_bounds
            if display_roi is not None:
                args.signal_roi = ((display_roi[0] - x) / (right - x), (display_roi[1] - y) / (bottom - y),
                                   (display_roi[2] - x) / (right - x), (display_roi[3] - y) / (bottom - y))
            if args.direction_centers:
                args.direction_centers = [(center - x) / (right - x) for center in args.direction_centers]
        generation = (loop_index, group_revision)
        model_generation=(*generation,getattr(args,'camera_generation',0)) if camera_input or hasattr(args,'camera_generation') else generation
        if model_generation != self.generation:
            self.filters = {direction: StableState(args.confirm_frames) for direction in DIRECTIONS}
            self.direction_history = DirectionHistory(args.confirm_frames)
            self.group_roi = args.signal_roi
            self.generation = model_generation
            self.live_result = self.live_geometry = None
        tracking_frames = ([(index, gray) for index, loop, revision, gray in frames
                            if (loop, revision) == generation] if frames else None)
        tracking_motions = ({(source, target): matrix for (source, target, loop, revision), matrix in motions.items()
                             if (loop, revision) == generation} if motions is not None else None)
        if native_bounds is not None and tracking_frames:
            x, y, right, bottom = native_bounds
            tracking_frames = [(index, gray[round(y * gray.shape[0]):round(bottom * gray.shape[0]),
                                             round(x * gray.shape[1]):round(right * gray.shape[1])])
                               for index, gray in tracking_frames]
            if camera_input and tracking_motions is not None:
                display_h,display_w=display_shape or frame.shape[:2]
                transform=np.array([[(right-x)*display_w/frame.shape[1],0,x*display_w],
                                    [0,(bottom-y)*display_h/frame.shape[0],y*display_h],[0,0,1]])
                inverse=np.linalg.inv(transform)
                tracking_motions={key:inverse@matrix@transform for key,matrix in tracking_motions.items()}
            else:
                tracking_motions = None
        shape = self.source_shape if self.source_cap is not None else frame.shape[:2]
        managed_geometry=(hasattr(args,'camera_generation') and self.group_roi is None and args.signal_roi is None)
        reused=(managed_geometry and self.reuse_live_tracks(frame.shape[:2],current,native_bounds,display_shape,
                                                           getattr(args,'live_signal_alignment',None)))
        if managed_geometry and not reused:
            self.direction_history.tracks=[]
        if tracking_frames:
            if not managed_geometry and self.direction_history.previous_gray is None:
                self.direction_history.advance(shape, tracking_frames[0][1], tracking_frames[:1], roi=self.group_roi)
            gray = tracking_frames[-1][1]
            if not managed_geometry:
                moved_roi = self.direction_history.advance(shape, gray, tracking_frames, tracking_motions, self.group_roi)
                self.group_roi = moved_roi if moved_roi is not None else args.signal_roi
        if self.source_cap is not None:
            cap, bounds, stride = self.source_cap, self.source_bounds, self.source_stride
            if bounds:
                h, w = self.source_shape
                context = (self.detection_bounds(self.source_shape, self.group_roi)
                           if self.group_roi is not None else (0, 0, w, h))
                if not (bounds[0] <= context[0] and bounds[1] <= context[1]
                        and context[2] <= bounds[2] and context[3] <= bounds[3]):
                    cap, bounds, stride = self.original_cap, None, 1
            if current % stride:
                raise RuntimeError("Inference frame does not match the exact-frame cache")
            target = current // stride
            position = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) if bounds else -1
            if bounds and 0 <= target - position < 16:
                for _ in range(target - position):
                    if not cap.grab():
                        raise RuntimeError("Cannot advance exact-frame cache")
            elif not cap.set(cv2.CAP_PROP_POS_FRAMES, target):
                raise RuntimeError("Cannot seek original inference video")
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Cannot decode original inference frame")
            if bounds:
                full = np.zeros((*self.source_shape, 3), dtype=np.uint8)
                full[bounds[1]:bounds[3], bounds[0]:bounds[2]] = frame
                frame = full
        height, width = frame.shape[:2]
        if not tracking_frames:
            gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), preview_dimensions(frame.shape, 640, 360))
            if not managed_geometry:
                moved_roi = self.direction_history.advance(frame.shape[:2], gray, roi=self.group_roi)
                self.group_roi = moved_roi if moved_roi is not None else args.signal_roi
        if managed_geometry:
            self.direction_history.previous_gray,self.direction_history.previous_shape=gray,shape
            self.direction_history.previous_index=tracking_frames[-1][0] if tracking_frames else None
        top_h = max(1, int(height * args.top_ratio))
        detection_args=args
        if camera_input:
            detection_args=SimpleNamespace(**vars(args))
            detection_args.conf_thres=min(args.conf_thres,.35)
        if self.group_roi:
            x1, y1, x2, y2 = self.detection_bounds(frame.shape, self.group_roi)
            detections = detect_roi(self.detector, frame[y1:y2, x1:x2], detection_args, x1, y1)
        elif args.tiles == 2:
            tile_w = int(width * 0.62)
            detections = detect_roi(self.detector, frame[:top_h, :tile_w], detection_args)
            detections += detect_roi(self.detector, frame[:top_h, width - tile_w:], detection_args, width - tile_w)
        else:
            detections = detect_roi(self.detector, frame[:top_h], detection_args)
        detections = filter_signal_group(second_nms(detections, args.iou_thres), frame.shape, self.group_roi)
        camera_recovered=set()
        if camera_input or hasattr(args,'camera_generation'):
            camera_recovered={tuple(d[:4]) for d in detections if any(
                track['stable'] in DIRECTIONS and
                (box_iou(d,track['box'])>=.3 or camera_housing_overlap(d,track['box']))
                for track in self.direction_history.tracks)}
            unique=[]
            for detection in sorted(detections,key=lambda box:(box[2]-box[0])*(box[3]-box[1]),reverse=True):
                if not any(camera_housing_overlap(detection,previous) for previous in unique):unique.append(detection)
            detections=unique
        retained_camera=set()
        camera_identity={}
        if args.direction_centers:
            predictions=[(assign_calibrated_direction(d,width,args.direction_centers),1.0) for d in detections]
        else:
            predictions=[None]*len(detections)
            if camera_input and reused:
                matches=sorted(((box_iou(detection,track['box']),index,track_index)
                    for index,detection in enumerate(detections)
                    for track_index,track in enumerate(self.direction_history.tracks)
                    if track['stable'] in DIRECTIONS and track['confidence']>=.9),reverse=True)
                used=set()
                for overlap,index,track_index in matches:
                    if overlap<.7 or predictions[index] is not None or track_index in used:
                        continue
                    color,_=observe_light_color(frame,detections[index][:4],camera=True)
                    track=self.direction_history.tracks[track_index]
                    camera_identity[index]=(track['stable'],track['confidence'],color)
                    used.add(track_index)
                    if color in ('red','yellow'):
                        # A verified housing keeps its identity through a round red/amber lamp.
                        predictions[index]=(track['stable'],track['confidence'])
                        retained_camera.add(tuple(detections[index][:4]))
            pending=[index for index,prediction in enumerate(predictions) if prediction is None]
            if pending:
                for index,prediction in zip(pending,self.classifier.predict(frame,[detections[i] for i in pending])):
                    predictions[index]=prediction
                    identity=camera_identity.get(index)
                    if (prediction[0]=='OTHER' and identity is not None and detections[index][4]>=args.conf_thres
                            and identity[2] in ('red','yellow','green')):
                        predictions[index]=identity[:2]
                        retained_camera.add(tuple(detections[index][:4]))
        if camera_input and getattr(args,'camera_review_dir',None) is not None and self.review_count<args.camera_review_frames:
            review=Path(args.camera_review_dir)
            review.mkdir(parents=True,exist_ok=True)
            stem=f'cam{args.front_channel+1}_{current:08d}'
            if not cv2.imwrite(str(review/(stem+'.jpg')),frame,[cv2.IMWRITE_JPEG_QUALITY,92]):
                raise RuntimeError('Cannot write the camera training review image')
            (review/(stem+'.json')).write_text(json.dumps({'channel':args.front_channel,'frame':current,
                'shape':[height,width],'captured_at':captured_at,'detections':detections,'predictions':predictions}))
            self.review_count+=1
        candidates=[(d,p) for d,p in zip(detections,predictions) if p[0]!='OTHER']
        if camera_input:
            # A weak new proposal needs a strong independent arrow classification and a full housing shape.
            candidates=[(d,p) for d,p in candidates if d[4]>=args.conf_thres or
                        (not managed_geometry and tuple(d[:4]) in camera_recovered) or
                        tuple(d[:4]) in retained_camera or
                        p[0] in DIRECTIONS and p[1]>=.9 and d[3]-d[1]>=1.2*(d[2]-d[0])]
        detections=[d for d,_ in candidates]
        predictions=[p for _,p in candidates]
        directions = self.direction_history.update(detections, predictions,
            preserve_identity=camera_input or hasattr(args,'camera_generation'))
        lights = []
        for detection, (direction, confidence) in zip(detections, directions):
            color = str(self.detector.names[detection[5]])
            color = "off" if color == "False" else color
            lights.append({"box": detection[:4], "color": color, "color_conf": detection[4],
                           "direction": direction,
                           "direction_conf": None if args.direction_centers else confidence,
                           "direction_source": ("calibrated" if args.direction_centers else
                                                "tracked_model" if tuple(detection[:4]) in retained_camera else "model"),
                           "in_signal_group": in_roi(detection[:4], frame.shape, self.group_roi)})
        for track in self.direction_history.tracks:
            if (track['last_seen']==self.direction_history.frame or
                    track['stable'] not in (*DIRECTIONS,'CIRCLE') or
                    not in_roi(track['box'],frame.shape,self.group_roi)):
                continue
            if (camera_input or hasattr(args,'camera_generation')) and any(camera_housing_overlap(track['box'],detection) or
                    track['stable']==direction[0] and (box_iou(track['box'],detection)>=.15 or
                        camera_input and managed_geometry and direction[1]>=.9)
                    for detection,direction in zip(detections,directions)):
                continue
            color,confidence=observe_light_color(frame,track['box'],camera=camera_input)
            # Keep housing geometry through a blink; display frames re-read the lamp color.
            lights.append({'box':track['box'],'color':color,'color_conf':confidence,
                           'direction':track['stable'],'direction_conf':track['confidence'],
                           'direction_source':'tracked_model','in_signal_group':True})
        raw = direction_states(lights, args.circle_direction)
        states = {key: self.filters[key].update(value) for key, value in raw.items()}
        if hasattr(args,'camera_generation'):
            lights=[dict(light,native_box=list(light['box'])) for light in lights]
        if native_bounds is not None:
            x, y, right, bottom = native_bounds
            display_height, display_width = display_shape or (height, width)
            gain = np.array([(right - x) * display_width / width, (bottom - y) * display_height / height] * 2)
            offset = np.array([x * display_width, y * display_height] * 2)
            lights = [{**light, "box": (np.asarray(light["box"]) * gain + offset).tolist()} for light in lights]
            if args.show:
                display = np.zeros((display_height, display_width, 3), dtype=frame.dtype)
                tile = display[round(y * display_height):round(bottom * display_height),
                               round(x * display_width):round(right * display_width)]
                tile[:] = (frame if tile.shape == frame.shape else frame[::2, ::2]
                           if tile.shape[:2] == (height // 2, width // 2) else
                           cv2.resize(frame, (tile.shape[1], tile.shape[0])))
                frame = display
        if self.device is not None and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        result = {"frame": current, "loop": loop_index, "group_revision": group_revision,
                "signal_roi": display_roi, "captured_at": captured_at,
                "image": frame if args.show else None,
                "lights": lights, "raw_states": raw, "states": states,
                "geometry_reused": bool(reused),
                "processing_ms": (time.perf_counter() - tick) * 1000}
        if hasattr(args,'camera_generation'):
            self.live_result=result
            self.live_geometry=(height,width,tuple(native_bounds) if native_bounds is not None else None,
                                tuple(display_shape) if display_shape is not None else None)
        return result


def warp_light_boxes(lights, matrix, shape, minimum=2):
    if not lights:
        return []
    boxes = np.asarray([light["box"] for light in lights], dtype=np.float32).reshape(-1, 2, 2)
    a, b = boxes[:, 0], boxes[:, 1]
    corners = np.stack((a, np.column_stack((b[:, 0], a[:, 1])), b,
                        np.column_stack((a[:, 0], b[:, 1]))), axis=1)
    projected = corners @ matrix[:2, :2].T + matrix[:2, 2]
    low, high = projected.min(axis=1), projected.max(axis=1)
    visible = (np.isfinite(projected).all(axis=(1, 2)) & (low >= 0).all(axis=1)
               & (high <= [shape[1], shape[0]]).all(axis=1) & (high - low >= minimum).all(axis=1))
    return [{**light, "box": box} for light, box, valid in
            zip(lights, np.concatenate((low, high), axis=1).tolist(), visible) if valid]


def align_light_boxes(source, current, lights, original_shape, cache=None):
    """Move sampled-frame boxes to the live frame using local, checked optical flow."""
    if not lights:
        return []
    if NATIVE_SIGNALS is not None and cache is not None:
        return NATIVE_SIGNALS.align(source,current,lights,original_shape,cache)
    if cache is not None:
        cache.pop("motion", None)
    height, width = current.shape
    key = (id(source), original_shape, current.shape, tuple(tuple(light["box"]) for light in lights))
    prepared = cache.get(key) if cache is not None else None
    next_source = cache.get("next_source") if cache is not None else None
    if prepared is None and next_source is not None and next_source[0] == key:
        prepared = next_source[1]
        cache.clear()
        cache[key] = prepared
    if prepared is None:
        gain = np.array([width / original_shape[1], height / original_shape[0]], dtype=np.float32)
        boxes = np.asarray([light["box"] for light in lights], dtype=np.float32).reshape(-1, 2, 2) * gain
        padding = max(24, int(np.max(boxes[:, 1] - boxes[:, 0]) * 2))
        start = np.maximum(0, np.floor(boxes[:, 0].min(axis=0)).astype(int) - padding)
        end = np.minimum([width, height], np.ceil(boxes[:, 1].max(axis=0)).astype(int) + padding)
        x1, y1 = start
        x2, y2 = end
        if x2 <= x1 or y2 <= y1:
            return []
        previous = np.ascontiguousarray(source[y1:y2, x1:x2])
        mask = np.zeros(previous.shape, dtype=np.uint8)
        for box in boxes:
            margin = max(5, int((box[1, 1] - box[0, 1]) / 2))
            a, b = np.floor(box[0] - start - margin).astype(int), np.ceil(box[1] - start + margin).astype(int)
            cv2.rectangle(mask, tuple(a), tuple(b), 255, -1)
        # Lit arrows change position between lamp cells; track their surroundings.
        for box in boxes:
            a, b = np.floor(box[0] - start).astype(int), np.ceil(box[1] - start).astype(int)
            cv2.rectangle(mask, tuple(a), tuple(b), 0, -1)
        points = cv2.goodFeaturesToTrack(previous, 30, .02, 4, mask=mask, blockSize=3)
        prepared = (gain, start, end, previous, points)
        if cache is not None:
            cache.clear()
            cache[key] = prepared
    gain, start, end, previous, points = prepared
    x1, y1 = start
    x2, y2 = end
    live = np.ascontiguousarray(current[y1:y2, x1:x2])
    if points is None or len(points) < 6:
        return []
    options = dict(winSize=(15, 15), maxLevel=3,
                   criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, .02))
    if NATIVE_SIGNALS is not None:
        moved,returned,forward,backward=NATIVE_SIGNALS.flow(previous,live,points)
    else:
        moved, forward, _ = cv2.calcOpticalFlowPyrLK(previous, live, points, None, **options)
    if moved is None:
        return []
    if NATIVE_SIGNALS is None:
        returned, backward, _ = cv2.calcOpticalFlowPyrLK(live, previous, moved, None, **options)
    if returned is None:
        return []
    valid = ((forward.ravel() == 1) & (backward.ravel() == 1)
             & np.isfinite(moved).all(axis=(1, 2)) & np.isfinite(returned).all(axis=(1, 2))
             & (np.linalg.norm(returned[:, 0] - points[:, 0], axis=1) < 1.0))
    if valid.sum() < 6:
        return []
    matrix, inliers = cv2.estimateAffinePartial2D(points[valid], moved[valid], method=cv2.RANSAC,
                                                ransacReprojThreshold=1.5, maxIters=100,
                                                confidence=.95, refineIters=5)
    if matrix is None or not np.isfinite(matrix).all() or inliers.sum() < max(6, valid.sum() * .6):
        return []
    motion = np.eye(3)
    motion[:2, :2] = matrix[:, :2]
    motion[:2, 2] = matrix[:, 2] + start - matrix[:, :2] @ start
    scale = np.diag([*gain, 1])
    motion = np.diag([*(1 / gain), 1]) @ motion @ scale
    if cache is not None:
        cache["motion"] = motion
    aligned = warp_light_boxes(lights, motion, original_shape, 2 / gain)
    if cache is not None and aligned:
        next_key = (id(current), original_shape, current.shape, tuple(tuple(light["box"]) for light in aligned))
        next_points = moved[valid][inliers.ravel().astype(bool)]
        cache["next_source"] = (next_key, (gain, start, end, live, next_points))
    return aligned


def observe_light_color(frame, box, camera=False):
    """Read a lit, saturated component inside a model-confirmed housing on this frame."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = map(round, box)
    inset = max(0, (x2 - x1) // 10)
    x1, x2 = max(0, min(width, x1 + inset)), max(0, min(width, x2 - inset))
    y1, y2 = max(0, min(height, y1)), max(0, min(height, y2))
    crop = frame[y1:y2, x1:x2]
    if crop.shape[0] < 3 or crop.shape[1] < 3:
        return "UNKNOWN", 0.0
    if NATIVE_SIGNALS is not None:
        return NATIVE_SIGNALS.observe(crop, camera=True) if camera else NATIVE_SIGNALS.observe(crop)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    saturation, value = hsv[:, :, 1], hsv[:, :, 2]
    peak = int(cv2.minMaxLoc(value, mask=cv2.inRange(saturation, 60, 255))[1])
    floor, green_peak = (100, 140) if camera else (140, 180)
    if camera and peak - float(np.median(value)) < 30:
        return "UNKNOWN", 0.0
    if peak < floor or peak < green_peak and peak - float(np.median(value)) < 55:
        return "UNKNOWN", 0.0
    minimum = max(floor, (peak * 4 + 4) // 5)
    # Grow compressed arrow pixels around a bright seed without accepting dim reflections alone.
    component_minimum = max(floor, round(peak * .6))
    masks = (cv2.bitwise_or(cv2.inRange(hsv, (0, 60, component_minimum), (12, 255, 255)),
                            cv2.inRange(hsv, (165, 60, component_minimum), (179, 255, 255))),
             cv2.inRange(hsv, (13, 60, component_minimum), (35, 255, 255)),
             cv2.inRange(hsv, (36, 60, component_minimum), (100, 255, 255)))
    scores, components = [], []
    for mask in masks:
        _, labels, statistics, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        seeds = np.bincount(labels[value >= minimum], minlength=len(statistics))
        component = None
        for label, candidate in enumerate(statistics[1:], 1):
            x, y, component_width, component_height, area = map(int, candidate)
            center = x + component_width / 2
            seed_count = int(seeds[label])
            # Rounded box edges have half a pixel of uncertainty, significant for 5-6 pixel housings.
            if (.2 * crop.shape[1] - .5 <= center <= .8 * crop.shape[1] + .5
                    and component_height <= 4 * component_width and seed_count
                    and (component is None or (seed_count, area) > (component[5], component[4]))):
                component = (x, y, component_width, component_height, area, seed_count)
        components.append(component)
        scores.append(component[5] if component is not None else 0)
    best = max(scores)
    confidence = best / max(1, sum(scores))
    index = scores.index(best)
    component = components[index]
    # Tiny red/amber arrows can occupy two pixels; keep the stronger green permission threshold.
    if (component is None or component[4] < max(3 if index == 2 else 2, crop.shape[0] * crop.shape[1] * .004)
            or index == 2 and peak < green_peak
            or component[4] == 2 and component[5] < 2 or confidence < .75):
        return "UNKNOWN", 0.0
    color = ("red", "yellow", "green")[index]
    x, y, component_width, component_height, _ = components[index][:5]
    if color == "red" and crop.shape[0] >= crop.shape[1] * 1.7 and .35 <= (y + component_height / 2) / crop.shape[0] <= .65:
        region = crop[max(0, y - 1):y + component_height + 1, max(0, x - 1):x + component_width + 1]
        lit = region[region.max(axis=2) >= max(floor, peak * .8)].mean(axis=0)
        # Compression can push amber toward orange; require its middle cell and green energy.
        if lit[1] >= lit[2] * .5:
            color = "yellow"
    return color, confidence


def current_observation(result, loop_index, group_revision, now, max_age, frames=None, original_shape=None,
                        circle_direction=None, frame=None, filters=None, motions=None, camera_color=False):
    valid = (result is not None and (result["loop"], result["group_revision"]) == (loop_index, group_revision)
             and now - result["captured_at"] <= max_age)
    if valid and frames is None and frame is None:
        return result["lights"], result["states"]
    lights = []
    if valid:
        if frames is None:
            lights = result["lights"]
        else:
            tracked = result.get("_live_alignment")
            source_index, source_lights = (tracked[0], tracked[1]) if tracked and tracked[1] else (result["frame"], result["lights"])
            source = next((gray for index, loop, revision, gray in frames
                           if (index, loop, revision) == (source_index, loop_index, group_revision)), None)
            if source is not None:
                sequence = ([(index, gray) for index, loop, revision, gray in frames
                             if (loop, revision) == (loop_index, group_revision) and index >= source_index]
                            if motions is not None else [(source_index, source), (frames[-1][0], frames[-1][3])])
                lights = source_lights
                for (previous_index, previous), (current_index, current) in zip(sequence, sequence[1:]):
                    if previous is current:
                        continue
                    key = (previous_index, current_index, loop_index, group_revision)
                    matrix = motions.get(key) if motions is not None else None
                    if matrix is not None:
                        lights = warp_light_boxes(lights, matrix, original_shape)
                    else:
                        cache = result.setdefault("_alignment_features", {})
                        lights = align_light_boxes(previous, current, lights, original_shape, cache)
                        if lights and motions is not None and "motion" in cache:
                            motions[key] = cache["motion"]
                    if not lights:
                        break
                result["_live_alignment"] = (frames[-1][0], lights)
    if frame is not None:
        gain = np.array([frame.shape[1] / original_shape[1], frame.shape[0] / original_shape[0]] * 2)
        lights = [{**light, "model_color": light["color"], "color_source": "current_frame",
                   "color": color, "color_conf": confidence}
                  for light in lights
                  for color, confidence in [observe_light_color(frame, np.asarray(light["box"]) * gain, camera=camera_color)]]
        observed = direction_states(lights, circle_direction)
        states = {direction: filters[direction].update(observed[direction]) if filters is not None else
                  observed[direction] or "UNKNOWN" for direction in DIRECTIONS}
        return lights, states
    observed = direction_states(lights, circle_direction)
    states = {direction: result["states"].get(direction, "UNKNOWN") if valid and
              observed.get(direction) in ("red", "yellow", "green") else "UNKNOWN" for direction in DIRECTIONS}
    return lights, states


@inference_runtime
def run(args):
    if args.assist_mode == "lane":
        from traffic_lane import run as run_lane
        return run_lane(args)
    fusion=args.assist_mode=='fusion'
    if args.realtime_threads and (not hasattr(os,'sched_getaffinity') or
                                  not {0,1,2,3}.issubset(os.sched_getaffinity(0))):
        raise RuntimeError('The measured realtime thread layout requires four available Linux CPU cores')
    front_bounds = None
    if args.pcie_capture:
        from traffic_rknn import PCIeCapture
        front_bounds = PCIeCapture.channel_bounds(args.front_channel)
    validate_roi(args.signal_roi, front_bounds)
    if args.pick_signal_group and (not args.show or not (args.pcie_capture or Path(str(args.source)).is_file())):
        raise ValueError("Interactive signal selection requires --show and a video file or PCIe capture")
    if args.loop and not Path(str(args.source)).is_file():
        raise ValueError("Loop playback requires a local video file")
    if not Path(args.weights).is_file():
        raise FileNotFoundError(f"Color weights not found: {args.weights}")
    output = Path(args.output)
    if output.suffix.lower() != ".mp4":
        raise ValueError("Output filename must end in .mp4")
    event_path, report_path = output.with_suffix(".jsonl"), output.with_suffix(".json")
    if any(path.exists() for path in (output, event_path, report_path)):
        raise FileExistsError(f"Choose a new output filename: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    print("Loading traffic models...", flush=True)
    board = uses_board_backend(args)
    backend = "PyTorch/ONNX"
    if board:
        from traffic_rknn import ONNXRuntimeCPUModel, OpenCVDNNModel, RKNNModel
        if Path(args.direction_weights).suffix != ".rknn":
            raise ValueError("The direction model must be RKNN for board inference")
        device = None
        cv2.setNumThreads(args.cpu_threads or 1)
        detector_type = (RKNNModel if Path(args.weights).suffix == ".rknn" else
                         ONNXRuntimeCPUModel if args.cpu_runtime == "onnxruntime" else OpenCVDNNModel)
        detector = (detector_type(args.weights, args.runtime, threads=args.cpu_threads or (1 if args.show else 4))
                    if detector_type is ONNXRuntimeCPUModel else detector_type(args.weights, args.runtime))
        backend = ("RKNN NPU" if detector_type is RKNNModel else
                   "ONNX Runtime FP32 CPU color + RKNN NPU direction" if detector_type is ONNXRuntimeCPUModel else
                   "OpenCV FP32 CPU color + RKNN NPU direction")
    else:
        load_pc_runtime()
        device = select_device(args.device)
        detector = DetectMultiBackend(str(args.weights), device=device, fp16=False)
        image_size = check_img_size(args.imgsz, s=detector.stride)
        detector.warmup(imgsz=(1, 3, image_size, image_size))
    direction_weights = (args.camera_direction_weights if args.pcie_capture and args.front_channel in (0, 1)
                         and args.camera_direction_weights is not None else args.direction_weights)
    classifier = None if args.direction_centers else DirectionClassifier(
        direction_weights, device, args.direction_conf, args.direction_margin, runtime=args.runtime)
    if classifier is not None and direction_weights == args.camera_direction_weights:
        classifier.threshold=classifier.config['confidence_threshold']
    cap = (PCIeCapture(args.capture_library, args.source, args.dma_delay_loops, args.fpga_pci,
                       dma_chunk_bytes=args.dma_chunk_bytes, source_channel=args.front_channel) if args.pcie_capture else
           cv2.VideoCapture(int(args.source) if str(args.source).isdigit() else str(args.source)))
    both_cameras = bool(args.pipeline_capture and args.pcie_capture and args.front_channel in (0, 1)
                        and cap.video_status is not None and cap.video_status.get('full_camera_capture'))
    preview_resize = cap.resize_preview if args.native_motion else None
    native_vision = cap.lib if args.native_motion else None
    global NATIVE_SIGNALS
    from traffic_rknn import NativeSignals
    NATIVE_SIGNALS=(NativeSignals(native_vision,camera=args.pcie_capture and args.front_channel in (0,1))
                    if native_vision is not None else None)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open source: {args.source}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if np.isfinite(fps) and fps > 0 else 30.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width < 360 or height < 360:
        cap.release()
        raise ValueError("Interactive display requires a video of at least 360x360 pixels")
    mosaic_frame = None
    if args.pick_signal_group:
        ok, mosaic_frame = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"No frames decoded from source: {args.source}")
        try:
            args.signal_roi = select_signal_group(mosaic_frame, args.display_width, args.display_height, front_bounds)
        except Exception:
            cap.release()
            raise
        if args.signal_roi is None:
            cap.release()
            print("Signal selection cancelled; no recording created.")
            return
        if not args.pcie_capture and not cap.set(cv2.CAP_PROP_POS_FRAMES, 0):
            cap.release()
            raise RuntimeError("Cannot restart video after signal selection")
    writer = None if args.no_record else cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), (fps if args.show else fps / args.vid_stride), (width, height))
    if writer is not None and not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot create output video: {output}")
    source_cap = original_inference_cap = None
    source_bounds, source_stride = None, 1
    if args.show and args.no_record and not args.pcie_capture and Path(str(args.source)).is_file():
        source_cap = cap
        try:
            if board and Path(args.weights).suffix == ".rknn" and args.signal_roi:
                source_bounds = signal_context_bounds((height, width), args.signal_roi)
                source_stride = 3
                cached = playback_preview(source_cap, args.source, output.parent.parent / "tmp", fps,
                                          (height, width), width, height, lossless=True,
                                          bounds=source_bounds, stride=source_stride)
                original_inference_cap = source_cap
                source_cap = cached
            cap = playback_preview(original_inference_cap or source_cap, args.source, output.parent.parent / "tmp", fps,
                                   (height, width), args.display_width, args.display_height)
        except Exception:
            source_cap.release()
            if original_inference_cap is not None:
                original_inference_cap.release()
            if board:
                detector.close()
                if classifier is not None:
                    classifier.model.close()
            raise
    analyzer = FrameAnalyzer(detector, classifier, device, source_cap, source_bounds,
                             (height, width), source_stride, original_inference_cap)
    lane_model=lane_worker=lane_tracker=None
    lane_latest=None
    lane_count=0
    lane_timings=[]
    lane_states={}
    person_model=person_worker=person_guard=person_tracker=None
    person_enabled=fusion and bool(args.pedestrian_weights)
    person_max_age=1.5 if both_cameras else 1.
    person_count=0
    person_timings=[]
    navigation_reader=None
    camera_navigation = args.pcie_capture and args.front_channel in (0, 1) and args.navigation_roi is None
    navigation_transitions=deque(maxlen=128)
    person_state={'status':'DISABLED','people':[],'hazards':[],'shape':None}
    if fusion:
        from traffic_lane import LaneModel,LaneEvidenceTracker,FusionSimulation,draw_lanes
        lane_model=LaneModel(args.lane_weights,args.runtime,args.lane_confidence)
        lane_tracker=LaneEvidenceTracker(args.capture_library if args.native_motion else None)
        infer_lane = (lambda image,number,captured,channel:
                      dict(lane_model.infer(image,number,captured), camera_channel=channel)) if both_cameras else lane_model.infer
        lane_worker=LatestFrameInference(infer_lane,args.auxiliary_fps) if args.show and lane_model is not None else None
        simulation=FusionSimulation(args.nav,args.moving)
        if args.navigation_roi:
            from traffic_navigation import NavigationReader
            navigation_reader=NavigationReader(args.navigation_roi)
            simulation.navigation_roi=tuple(args.navigation_roi)
            simulation.navigation_enabled=True
        elif camera_navigation:
            from traffic_navigation import NavigationReader
            navigation_reader=NavigationReader(None,executor=cap.screen_executor)
        if args.pedestrian_weights:
            from traffic_pedestrian import PersonModel,PersonTracker,PedestrianGuard,draw_people
            person_model=PersonModel(args.pedestrian_weights,args.runtime,args.pedestrian_confidence)
            if both_cameras:
                def infer_person(image,number,captured,channel):
                    result=dict(person_model.infer(image,number,captured),camera_channel=channel)
                    return pair_person_trackers[channel].align_detection(result) if args.native_motion else result
            else:
                infer_person=person_model.infer
            person_worker=LatestFrameInference(infer_person,args.auxiliary_fps) if args.show else None
            person_guard=PedestrianGuard(max_age=person_max_age)
            person_tracker=PersonTracker(args.capture_library if args.native_motion else None)
    else:
        simulation = VehicleSimulation(args.nav, args.moving)
    other_channel = 1-args.front_channel if both_cameras else None
    other_analyzer = FrameAnalyzer(detector,classifier,device) if both_cameras else None
    other_latest = None
    other_filters = {direction:StableState(args.confirm_frames) for direction in DIRECTIONS}
    other_simulation = (FusionSimulation(args.nav,args.moving) if fusion else VehicleSimulation(args.nav,args.moving)) if both_cameras else None
    other_navigation = NavigationReader(None,executor=cap.screen_executor) if both_cameras and fusion else None
    other_guard = PedestrianGuard(max_age=person_max_age) if both_cameras and person_enabled else None
    hdmi_classifier=DirectionClassifier(args.direction_weights,device,args.direction_conf,args.direction_margin,
        runtime=args.runtime) if both_cameras else None
    hdmi_analyzer=FrameAnalyzer(detector,hdmi_classifier,device) if both_cameras else None
    hdmi_latest=None
    hdmi_filters={direction:StableState(args.confirm_frames) for direction in DIRECTIONS}
    hdmi_motion_history={}
    hdmi_simulation=(FusionSimulation(args.nav,args.moving) if fusion else VehicleSimulation(args.nav,args.moving)) if both_cameras else None
    if both_cameras and fusion:
        simulation.navigation_peers=(other_simulation,hdmi_simulation)
    hdmi_navigation=NavigationReader((.02,.02,.20,.27)) if both_cameras and fusion else None
    hdmi_guard=PedestrianGuard(max_age=person_max_age) if both_cameras and person_enabled else None
    hdmi_lights=[]
    camera_inferred = {0:0,1:0,2:0}
    camera_geometry_reused = {0:0,1:0,2:0}
    camera_generations = {0:0,1:0,2:0}
    def analyze_camera(*job):
        channel=job[1].front_channel
        selected=hdmi_analyzer if channel==2 else analyzer if channel==args.front_channel else other_analyzer
        result=selected(*job)
        result['camera_channel']=channel
        result['camera_generation']=job[1].camera_generation
        return result
    worker = LatestFrameInference(analyze_camera if both_cameras else analyzer,args.signal_fps) if args.show else None
    fpga = None
    cpu_key = None
    frame_index, decoded, displayed, processed, timings, direction_counts = 0, 0, 0, 0, [], {}
    loop_index, group_revision = 0, 0
    latest_result = None
    screen_revision = prepared_revision = (0,0) if both_cameras else 0
    last_camera_mask = prepared_camera_mask = cap.video_status['valid_mask']&3 if both_cameras else 0
    scene_start_frame = {0:0,1:0,2:0} if both_cameras else 0
    observed_scene_frames={0:0,1:0,2:0}
    pair_scene_grays={0:None,1:None,2:None}
    pair_lane_latest = {0:None,1:None,2:None}
    pair_lane_trackers = ({args.front_channel:lane_tracker,
                          other_channel:LaneEvidenceTracker(args.capture_library if args.native_motion else None)}
                         if both_cameras and fusion else {})
    camera_person_tracker=PersonTracker if person_enabled else None
    hdmi_person_tracker=PersonTracker if person_enabled else None
    if both_cameras and person_enabled and args.native_motion:
        from traffic_camera import CameraPersonTracker
        camera_person_tracker=CameraPersonTracker
        hdmi_person_tracker=lambda library:CameraPersonTracker(library,camera_refinement=False)
        person_tracker=CameraPersonTracker(args.capture_library)
    pair_person_trackers = ({args.front_channel:person_tracker,
                            other_channel:camera_person_tracker(args.capture_library if args.native_motion else None)}
                           if both_cameras and person_enabled else {})
    if both_cameras and fusion:
        pair_lane_trackers[2]=LaneEvidenceTracker(args.capture_library if args.native_motion else None)
        if person_enabled:pair_person_trackers[2]=hdmi_person_tracker(args.capture_library if args.native_motion else None)
    next_lane_channel = next_person_channel = args.front_channel
    def camera_prepare_affinity():
        if args.realtime_threads:os.sched_setaffinity(0,{0,1,3})
    camera_prepare_pool=(ThreadPoolExecutor(max_workers=3,initializer=camera_prepare_affinity)
                         if both_cameras else None)
    def prepare_camera_pair(packet):
        nonlocal person_count,lane_count,prepared_revision,prepared_camera_mask,scene_start_frame,next_lane_channel,next_person_channel
        prepare_started=time.perf_counter()
        revision=packet['status'].get('camera_screen_revisions',(0,0))
        mask=packet['status']['valid_mask']&7
        if revision!=prepared_revision or mask!=prepared_camera_mask:
            changed=changed_camera_channels(prepared_revision,revision,prepared_camera_mask,mask)
            if (prepared_camera_mask^mask)&4:changed.add(2)
            for channel in changed:
                scene_start_frame[channel]=packet['number']
                pair_lane_latest[channel]=None
                pair_lane_trackers[channel]=LaneEvidenceTracker(args.capture_library if args.native_motion else None)
                if person_enabled:
                    factory=hdmi_person_tracker if channel==2 else camera_person_tracker
                    pair_person_trackers[channel]=factory(args.capture_library if args.native_motion else None)
                    pair_person_trackers[channel].cut_frame=scene_start_frame[channel]
            prepared_revision,prepared_camera_mask=revision,mask
        fronts={channel:packet['frame'][(channel//2)*360:(channel//2+1)*360,
            (channel%2)*640:(channel%2+1)*640] for channel in (0,1,2)}
        grays={channel:cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) for channel,image in fronts.items()}
        for channel,gray in grays.items():
            if source_scene_changed(pair_scene_grays[channel],gray):
                scene_start_frame[channel]=packet['number']
                pair_lane_latest[channel]=None
                pair_lane_trackers[channel]=LaneEvidenceTracker(args.capture_library if args.native_motion else None)
                if person_enabled:
                    factory=hdmi_person_tracker if channel==2 else camera_person_tracker
                    pair_person_trackers[channel]=factory(args.capture_library if args.native_motion else None)
                    pair_person_trackers[channel].cut_frame=scene_start_frame[channel]
            pair_scene_grays[channel]=gray
        lane_result=lane_worker.take()
        if lane_result is not None and lane_result['frame']>=scene_start_frame[lane_result['camera_channel']]:
            pair_lane_latest[lane_result['camera_channel']]=lane_result
            lane_count+=1;lane_timings.append(lane_result['processing_ms'])
        person_result=person_worker.take() if person_worker is not None else None
        if person_result is not None:
            person_count+=1;person_timings.append(person_result['processing_ms'])
        if lane_worker.ready():
            channel=next_lane_channel;next_lane_channel=(channel+1)%3
            lane_worker.submit((fronts[channel].copy(),packet['number'],packet['captured_at'],channel))
        if person_worker is not None and person_worker.ready():
            channel=next_person_channel;next_person_channel=(channel+1)%3
            person_worker.submit((fronts[channel].copy(),packet['number'],packet['captured_at'],channel))
        channel_costs={}
        def prepare_channel(channel):
            channel_started=time.perf_counter();cpu_started=time.thread_time()
            valid=bool(packet['status']['valid_mask']&(1<<channel))
            latest=pair_lane_latest[channel]
            evidence=(latest['boundaries'] if latest is not None and valid and
                      time.perf_counter()-latest['captured_at']<=min(args.max_result_age,.7) else {})
            if not valid:
                pair_lane_latest[channel]=None
                return channel,{},None
            boundaries=pair_lane_trackers[channel].update(fronts[channel],evidence,latest['frame'] if latest else None,
                                    packet['number'],time.perf_counter(),grayscale=grays[channel])
            lane_finished=time.perf_counter()
            result=(person_result if person_result is not None and person_result.get('camera_channel')==channel
                    and person_result.get('source_frame',person_result['frame'])>=scene_start_frame[channel] else None)
            observation=(pair_person_trackers[channel].update(fronts[channel],result,packet['number'])
                         if valid and person_enabled else None)
            channel_costs[channel]=[round(1000*(lane_finished-channel_started),1),
                round(1000*(time.perf_counter()-lane_finished),1),round(1000*(time.thread_time()-cpu_started),1)]
            return channel,boundaries,dict(observation) if observation is not None else None
        setup_ms=1000*(time.perf_counter()-prepare_started)
        prepared=list(camera_prepare_pool.map(prepare_channel,(0,1,2)))
        lanes={channel:boundaries for channel,boundaries,_ in prepared}
        people={channel:observation for channel,_,observation in prepared}
        packet.update(camera_lanes=lanes,camera_people=people,camera_grays=grays,
                      camera_prepare_costs={'setup':round(setup_ms,1),'channels':channel_costs},
                      camera_scene_frames=dict(scene_start_frame),
                      lane_observation=lanes[args.front_channel],person_observation=people[args.front_channel],
                      front_gray=grays[args.front_channel])
        return packet
    def prepare_people(packet):
        nonlocal person_count,lane_count,lane_latest,lane_tracker,person_tracker,prepared_revision,scene_start_frame
        if both_cameras:
            return prepare_camera_pair(packet)
        revision = packet['status'].get('camera_screen_revision', 0)
        if revision != prepared_revision:
            prepared_revision, scene_start_frame = revision, packet['number']
            lane_latest = None
            lane_tracker = LaneEvidenceTracker(args.capture_library if args.native_motion else None)
            person_tracker = PersonTracker()
        observation = None
        x1,y1,x2,y2=front_bounds
        image=packet['frame']
        front=image[round(y1*height):round(y2*height),round(x1*width):round(x2*width)]
        gray=cv2.cvtColor(front,cv2.COLOR_BGR2GRAY)
        model_frame=front.copy()
        lane_result=lane_worker.take()
        if lane_result is not None and lane_result['frame'] < scene_start_frame:
            lane_result = None
        lane_worker.submit((model_frame,packet['number'],packet['captured_at']))
        if lane_result is not None:
            lane_latest=lane_result;lane_count+=1;lane_timings.append(lane_result['processing_ms'])
        boundaries=(lane_latest['boundaries'] if lane_latest is not None and
                    time.perf_counter()-lane_latest['captured_at']<=min(args.max_result_age,.7) else {})
        if not packet['status']['valid_mask'] & (1 << args.front_channel):
            boundaries={};lane_latest=None
            lane_tracker=LaneEvidenceTracker(args.capture_library if args.native_motion else None)
        packet['lane_observation']=lane_tracker.update(front,boundaries,lane_latest['frame'] if lane_latest else None,
                                                       packet['number'],time.perf_counter(),grayscale=gray)
        packet['front_gray']=gray
        if packet['status']['valid_mask'] & (1 << args.front_channel):
            result=person_worker.take()
            if result is not None and result['frame'] < scene_start_frame:
                result = None
            person_worker.submit((model_frame,packet['number'],packet['captured_at']))
            if result is not None:
                person_count+=1
                person_timings.append(result['processing_ms'])
            observation=person_tracker.update(front,result,packet['number'])
        packet['person_observation']=dict(observation) if observation is not None else None
        return packet
    live_frames = deque(maxlen=max(2, int(np.ceil(fps * args.max_result_age)) + 2))
    if args.pick_signal_group:
        selected = cv2.cvtColor(mosaic_frame, cv2.COLOR_BGR2GRAY)
        selected = cv2.resize(selected, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        live_frames.append((-1, loop_index, group_revision, selected))
    live_filters = {direction: StableState(args.confirm_frames) for direction in DIRECTIONS}
    motion_history = {}
    other_motion_history = {}
    camera_profile=deque(maxlen=120)
    camera_output_profile=deque(maxlen=120)
    camera_last_log_ms=0.
    last_log = 0.0
    last_decision = None
    next_mode = None
    requested_input = [args.front_channel] if args.pcie_capture and not both_cameras else None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video {width}x{height}, {'display limit' if args.pcie_capture else 'source'}={fps:.2f} FPS, "
          f"total={total_frames}; {backend}", flush=True)
    if args.pcie_capture:
        print(f"Live PCIe capture; front channel={args.front_channel}; hardware inputs require separate validation", flush=True)
    started = time.perf_counter()
    camera_display_rates=deque([(started,displayed)])
    analysis_width = (min(384, args.display_width - 640)
                      if args.show and args.observation_panel and args.display_width >= 880 else 0)
    live_width = args.display_width - analysis_width
    previous_switch_interval=sys.getswitchinterval()
    previous_affinity = os.sched_getaffinity(0) if args.realtime_threads else None
    tracking_pool=None
    display_backend=cv2
    try:
        if getattr(args,'native_display',False):
            if not (both_cameras and args.native_motion and args.show):
                raise ValueError('native-display requires the joint native preview')
            from traffic_camera import NativeDisplay
            display_backend=NativeDisplay()
        if args.fpga_pci:
            if not args.show:
                raise ValueError("FPGA interaction requires --show so keys and heartbeat remain responsive")
            from traffic_fpga import CPUMotionKey, FPGAInteraction
            fpga = FPGAInteraction(args.fpga_pci)
            cpu_key = CPUMotionKey()
        if args.show:
            preview_limit=args.display_height-(70 if fusion else 0)
            preview_width, preview_height = preview_dimensions((height, width), live_width, preview_limit)
            preview_shape = (max(preview_height, min(360, preview_limit))+(70 if fusion else 0),
                             max(preview_width, min(640, live_width)))
            display_backend.namedWindow("Traffic light direction", cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
            display_backend.setMouseCallback("Traffic light direction", handle_mouse,
                                 (simulation, preview_shape, preview_shape, requested_input))
        if args.pipeline_capture:
            cap = LiveFrameCapture(cap, worker, args.front_channel,
                                   prepare_people if person_enabled or both_cameras and fusion else None,
                                   both_cameras=both_cameras,include_hdmi=both_cameras)
            if args.realtime_threads:
                sys.setswitchinterval(.001)
                for inference_worker in (worker,lane_worker,person_worker):
                    if inference_worker is not None:
                        os.sched_setaffinity(inference_worker.thread.native_id,{0,3} if inference_worker is worker else {0})
                os.sched_setaffinity(cap.thread.native_id,{2})
                if cap.prepare_thread is not None:
                    os.sched_setaffinity(cap.prepare_thread.native_id,{3})
                os.sched_setaffinity(0,{1})
            if both_cameras:
                def tracking_affinity():
                    if args.realtime_threads:os.sched_setaffinity(0,{0,1,3})
                tracking_pool=ThreadPoolExecutor(max_workers=3,initializer=tracking_affinity)
        with (nullcontext(None) if args.no_record else event_path.open("w", encoding="utf-8")) as events:
            while True:
                if requested_input is not None and requested_input[0] != args.front_channel:
                    next_mode = requested_input[0]
                    break
                tick = time.perf_counter()
                analysis_args = SimpleNamespace(**vars(args))
                if args.pcie_capture and args.front_channel in (2, 3) and analysis_args.signal_roi is None:
                    analysis_args.signal_roi = front_bounds
                native_job = None
                full_hdmi = (not args.pipeline_capture and args.pcie_capture and cap.video_status is not None
                             and native_capture_available(cap.video_status, args.front_channel))
                if args.pipeline_capture:
                    ok, frame = cap.read()
                    full_hdmi = both_cameras or native_capture_available(cap.video_status, args.front_channel)
                    if ok and cap.native_frame is not None:
                        native_channel=cap.native_frame_channel if both_cameras else args.front_channel
                        analysis_args.front_channel=native_channel
                        if native_channel!=args.front_channel:
                            analysis_args.signal_roi=analysis_args.direction_centers=analysis_args.circle_direction=None
                        if native_channel==2 and not both_cameras:
                            analysis_args.signal_roi=PCIeCapture.channel_bounds(2)
                        native_job = (cap.native_frame, analysis_args, cap.frame_id, loop_index,
                                      group_revision, cap.captured_at)
                elif full_hdmi and cap.capture_mode == 0 and (worker is None or worker.ready()):
                    native_ok, native_frame = cap.read(full_hdmi=True)
                    if not native_ok:
                        raise RuntimeError("Cannot capture the full HDMI inference frame")
                    native_at = time.perf_counter()
                    native_job = (native_frame, analysis_args, frame_index, loop_index,
                                  group_revision, native_at)
                    frame = update_native_preview(mosaic_frame, native_frame, args.front_channel)
                    ok = native_ok
                else:
                    ok, frame = cap.read()
                    mosaic_frame = frame
                if not ok:
                    if args.loop:
                        if frame_index == 0 or not cap.set(cv2.CAP_PROP_POS_FRAMES, 0):
                            raise RuntimeError("Cannot restart video playback")
                        frame_index = 0
                        loop_index += 1
                        if fusion:
                            lane_latest=None
                            lane_tracker=LaneEvidenceTracker()
                            if person_guard is not None:
                                person_tracker=PersonTracker()
                                person_guard=PedestrianGuard()
                        continue
                    break
                current = cap.frame_id if args.pipeline_capture else frame_index
                camera_capture_tick=time.perf_counter()
                frame_index = current+1
                decoded += 1
                person_frame = current if args.pipeline_capture else decoded-1
                if not args.show and current % args.vid_stride:
                    continue
                if fpga is not None:
                    previous_control=fpga.last_control
                    control = fpga.poll()
                    if control is not None:
                        # Software direction echoes must not become manual steering commands.
                        if not (fusion and simulation.has_navigation_context()) or (
                                previous_control is not None and not fpga.software_echo):
                            simulation.command(control["navigation"])
                        simulation.manual_moving = control["manual_moving"]
                captured_at = cap.captured_at if args.pipeline_capture else time.perf_counter()
                revision = (cap.video_status.get('camera_screen_revisions', (0,0)) if both_cameras else
                            cap.video_status.get('camera_screen_revision', 0)
                            if args.pcie_capture and cap.video_status is not None else 0)
                camera_mask=cap.video_status['valid_mask']&7 if both_cameras else 0
                if both_cameras:
                    changed=changed_camera_channels(screen_revision,revision,last_camera_mask,camera_mask)
                    scene_frames=cap.camera_scene_frames
                    changed.update(channel for channel,number in scene_frames.items()
                                   if number>observed_scene_frames[channel])
                    observed_scene_frames.update(scene_frames)
                    if (last_camera_mask^camera_mask)&4:
                        changed.add(2)
                    if changed:
                        print(f'Camera reset: channels={sorted(changed)} screens={screen_revision}->{revision} inputs={last_camera_mask}->{camera_mask}',flush=True)
                    for channel in changed:camera_generations[channel]+=1
                    if args.front_channel in changed:
                        latest_result=None;motion_history.clear()
                        live_filters={direction:StableState(args.confirm_frames) for direction in DIRECTIONS}
                        if person_enabled:person_guard=PedestrianGuard(max_age=person_max_age)
                    if other_channel in changed:
                        other_latest=None;other_motion_history.clear()
                        other_filters={direction:StableState(args.confirm_frames) for direction in DIRECTIONS}
                        if person_enabled:other_guard=PedestrianGuard(max_age=person_max_age)
                    if 2 in changed:
                        hdmi_latest=None;hdmi_motion_history.clear()
                        hdmi_filters={direction:StableState(args.confirm_frames) for direction in DIRECTIONS}
                        if person_enabled:hdmi_guard=PedestrianGuard(max_age=person_max_age)
                    screen_revision,last_camera_mask=revision,camera_mask
                    analysis_args.camera_generation=camera_generations[analysis_args.front_channel]
                elif args.pcie_capture and cap.video_status is not None and (
                        revision != screen_revision or (camera_mask!=last_camera_mask if both_cameras else
                        not cap.video_status["valid_mask"] & (1 << args.front_channel))):
                    screen_revision = revision
                    last_camera_mask=camera_mask
                    latest_result = None
                    group_revision += 1
                    live_frames.clear()
                    motion_history.clear()
                    other_motion_history.clear()
                    live_filters = {direction: StableState(args.confirm_frames) for direction in DIRECTIONS}
                    other_latest=None
                    other_filters={direction:StableState(args.confirm_frames) for direction in DIRECTIONS}
                if both_cameras:
                    if not camera_mask&(1<<args.front_channel):latest_result=None
                    if not camera_mask&(1<<other_channel):other_latest=None
                front_gray = cap.front_gray if args.pipeline_capture and (person_enabled or both_cameras and fusion) else None
                if args.show:
                    if front_bounds is not None:
                        x1, y1, x2, y2 = (round(front_bounds[i] * (width if i % 2 == 0 else height)) for i in range(4))
                        gray = np.zeros(frame.shape[:2], dtype=np.uint8)
                        if front_gray is None:
                            front_gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                        gray[y1:y2, x1:x2] = front_gray
                        if both_cameras:
                            other_gray=(cap.camera_grays[other_channel] if cap.camera_grays is not None else
                                        cv2.cvtColor(frame[:360,other_channel*640:(other_channel+1)*640],cv2.COLOR_BGR2GRAY))
                            gray[:360,other_channel*640:(other_channel+1)*640]=other_gray
                            gray[360:,:640]=(cap.camera_grays[2] if cap.camera_grays is not None else
                                cv2.cvtColor(frame[360:,:640],cv2.COLOR_BGR2GRAY))
                    else:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    live_frames.append((current, loop_index, group_revision, gray))
                    motion_history = {key: matrix for key, matrix in motion_history.items()
                                      if key[2:] == (loop_index, group_revision) and key[1] >= live_frames[0][0]}
                    if both_cameras:
                        other_motion_history={key:matrix for key,matrix in other_motion_history.items()
                            if key[2:]==(loop_index,group_revision) and key[1]>=live_frames[0][0]}
                        hdmi_motion_history={key:matrix for key,matrix in hdmi_motion_history.items()
                            if key[2:]==(loop_index,group_revision) and key[1]>=live_frames[0][0]}
                job = (None if source_cap is not None or full_hdmi else frame.copy() if args.show else frame,
                       analysis_args,
                       current, loop_index, group_revision, captured_at, list(live_frames) if args.show else None,
                       dict(motion_history))
                if args.pcie_capture and args.front_channel in (0, 1) and not full_hdmi:
                    # Detect on the selected camera, so other tiles cannot shrink or supply its targets.
                    x1, y1, x2, y2 = front_bounds
                    camera = frame[round(y1 * height):round(y2 * height), round(x1 * width):round(x2 * width)]
                    job = (camera.copy(), *job[1:], front_bounds, (height, width))
                if native_job is not None:
                    if both_cameras:
                        previous=(hdmi_latest if native_channel==2 else latest_result
                                  if native_channel==args.front_channel else other_latest)
                        aligned=previous.get('_live_alignment') if previous is not None else None
                        if aligned and aligned[1]:
                            analysis_args.live_signal_alignment=(previous['frame'],
                                (aligned[0],[dict(light) for light in aligned[1]]))
                    native_job += (list(live_frames) if args.show else None,
                                   dict(hdmi_motion_history if native_channel==2 else
                                        motion_history if native_channel==args.front_channel else other_motion_history)
                                   if both_cameras else None,
                                   PCIeCapture.channel_bounds(native_channel) if both_cameras else front_bounds)
                    if both_cameras and native_channel==2:
                        native_job+=((height,width),)
                if worker is not None:
                    if native_job is not None:
                        worker.submit(native_job)
                    elif not full_hdmi and current % args.vid_stride == 0 and current % source_stride == 0:
                        worker.submit(job)
                    result = worker.take()
                else:
                    result = analyzer(*(native_job if native_job is not None else job))
                if (result is not None and both_cameras and
                        result['camera_generation']!=camera_generations[result['camera_channel']]):
                    result=None
                if result is not None:
                    if both_cameras:
                        camera_inferred[result['camera_channel']]+=1
                        if result.get('geometry_reused'):
                            camera_geometry_reused[result['camera_channel']]+=1
                    if both_cameras and result['camera_channel']==2:
                        hdmi_latest=result
                    elif both_cameras and result['camera_channel']!=args.front_channel:
                        other_latest=result
                    else:
                        latest_result = result
                    processed += 1
                    timings.append(result["processing_ms"])
                    for light in result["lights"]:
                        direction = light["direction"]
                        direction_counts[direction] = direction_counts.get(direction, 0) + 1
                if both_cameras:
                    tracking_frames=list(live_frames)
                    now=time.perf_counter()
                    tracking_jobs=((latest_result,args.circle_direction,live_filters,motion_history,True),
                                   (other_latest,None,other_filters,other_motion_history,True),
                                   (hdmi_latest,args.circle_direction,hdmi_filters,hdmi_motion_history,False))
                    def observe(job):
                        selected,circle,filters,motions,camera=job
                        return current_observation(selected,loop_index,group_revision,now,args.max_result_age,
                            tracking_frames,(height,width),circle,frame,filters,motions,camera_color=camera)
                    parallel=any(job[0] is not None and (job[0]['lights'] or
                        job[0].get('_live_alignment') and job[0]['_live_alignment'][1]) for job in tracking_jobs)
                    observations=tracking_pool.map(observe,tracking_jobs) if parallel else map(observe,tracking_jobs)
                    (lights,states),other_observation,(hdmi_lights,hdmi_states)=observations
                    view_states=states
                else:
                    lights, states = (current_observation(latest_result, loop_index, group_revision,
                                                     time.perf_counter(), args.max_result_age, live_frames,
                                                     (height, width), args.circle_direction, frame, live_filters, motion_history,
                                                     camera_color=args.pcie_capture and args.front_channel in (0, 1))
                                      if args.show else (result["lights"], result["states"]) if result is not None else ([],{}))
                raw = result["raw_states"] if result is not None else {key: None for key in DIRECTIONS}
                camera_signal_tick=time.perf_counter()
                signal_active = (args.signal_roi is not None or
                                 args.pcie_capture and args.front_channel in (0, 1) and
                                 bool(lights))
                cpu_action = cpu_key.poll() if cpu_key is not None else None
                if fusion:
                    x1,y1,x2,y2=front_bounds or (0,0,1,1)
                    front=frame[round(y1*height):round(y2*height),round(x1*width):round(x2*width)]
                    if not args.pipeline_capture:front=cv2.resize(front,(640,360))
                    front_valid=not args.pcie_capture or bool(cap.video_status['valid_mask']&(1<<args.front_channel))
                    if navigation_reader is not None and (simulation.navigation_enabled or camera_navigation):
                        navigation=(navigation_reader.update(front,captured_at) if front_valid else
                                    {'direction':'UNKNOWN','score':0.,'captured_at':captured_at})
                        if navigation['direction']!=simulation.navigation_direction:
                            navigation_transitions.append({'frame':current,**navigation})
                        simulation.navigation_direction=navigation['direction']
                        if camera_navigation:
                            simulation.navigation_enabled=navigation['direction'] in DIRECTIONS
                    if args.pipeline_capture and (person_enabled or both_cameras):
                        boundaries=cap.lane_observation
                    else:
                        lane_result=lane_worker.take() if lane_worker is not None else lane_model.infer(front,current,captured_at)
                        if lane_worker is not None:
                            lane_worker.submit((front.copy(),current,captured_at))
                        if lane_result is not None:
                            lane_latest=lane_result
                            lane_count+=1
                            lane_timings.append(lane_result['processing_ms'])
                        boundaries=(lane_latest['boundaries'] if lane_latest is not None and
                                    time.perf_counter()-lane_latest['captured_at']<=min(args.max_result_age,.7) else {})
                        if not front_valid:
                            boundaries={}
                            lane_latest=None
                            lane_tracker=LaneEvidenceTracker(args.capture_library if args.native_motion else None)
                        boundaries=lane_tracker.update(front,boundaries,lane_latest['frame'] if lane_latest else None,
                                                        current,time.perf_counter(),grayscale=front_gray)
                    target=frame[round(y1*height):round(y2*height),round(x1*width):round(x2*width)]
                    if native_vision is not None:draw_lanes(target,boundaries,native=native_vision)
                    else:draw_lanes(target,boundaries)
                    lane_states={side:item['kind'] for side,item in boundaries.items()}
                    decision=simulation.update_fused(boundaries,states,signal_active,cpu_action)
                    if person_enabled:
                        person_observation=None
                        if args.pipeline_capture:
                            person_observation=cap.person_observation
                        elif front_valid:
                            person_result=(person_worker.take() if person_worker is not None else
                                           person_model.infer(front,person_frame,captured_at))
                            if person_worker is not None:
                                person_worker.submit((front.copy(),person_frame,captured_at))
                            if person_result is not None:
                                person_count+=1
                                person_timings.append(person_result['processing_ms'])
                            person_observation=person_tracker.update(front,person_result,person_frame)
                        person_state=person_guard.update(person_observation,time.perf_counter())
                        person_guard.apply(simulation,person_state)
                        decision=simulation.decision
                        draw_people(frame[round(y1*height):round(y2*height),round(x1*width):round(x2*width)],person_state)
                else:
                    decision = simulation.update(states, signal_active, cpu_action)
                other_lights=[]
                if both_cameras:
                    other_lights,other_states=other_observation
                    other_simulation.nav=other_simulation.requested_nav=simulation.requested_nav
                    other_simulation.manual_moving=simulation.manual_moving
                    other_front=frame[:360,other_channel*640:(other_channel+1)*640]
                    if fusion:
                        other_simulation.lane_change=simulation.lane_change
                        other_simulation.navigation_manual=simulation.navigation_manual
                        navigation=(other_navigation.update(other_front,captured_at)
                            if cap.video_status['valid_mask']&(1<<other_channel) else
                            {'direction':'UNKNOWN','score':0.,'captured_at':captured_at})
                        other_simulation.navigation_direction=navigation['direction']
                        other_simulation.navigation_enabled=navigation['direction'] in DIRECTIONS
                        other_boundaries=cap.camera_lanes[other_channel] if cap.camera_lanes is not None else {}
                        other_simulation.update_fused(other_boundaries,other_states,bool(other_lights))
                        draw_lanes(other_front,other_boundaries,native=native_vision)
                        if other_guard is not None:
                            observation=cap.camera_people[other_channel] if cap.camera_people is not None else None
                            other_people=other_guard.update(observation,time.perf_counter())
                            other_guard.apply(other_simulation,other_people)
                            draw_people(other_front,other_people)
                    else:
                        other_simulation.update(other_states,bool(other_lights))
                    hdmi_front=frame[360:,:640]
                    hdmi_simulation.nav=hdmi_simulation.requested_nav=simulation.requested_nav
                    hdmi_simulation.manual_moving=simulation.manual_moving
                    hdmi_valid=bool(cap.video_status['valid_mask']&4)
                    if fusion:
                        hdmi_simulation.lane_change=simulation.lane_change
                        hdmi_simulation.navigation_manual=simulation.navigation_manual
                        route=(hdmi_navigation.update(hdmi_front,captured_at) if hdmi_valid else
                            {'direction':'UNKNOWN','score':0.,'captured_at':captured_at})
                        hdmi_simulation.navigation_direction=route['direction']
                        hdmi_simulation.navigation_enabled=route['direction'] in DIRECTIONS
                        hdmi_boundaries=cap.camera_lanes[2] if cap.camera_lanes is not None and hdmi_valid else {}
                        hdmi_simulation.update_fused(hdmi_boundaries,hdmi_states,bool(hdmi_lights))
                        draw_lanes(hdmi_front,hdmi_boundaries,native=native_vision)
                        if hdmi_guard is not None:
                            hdmi_people=hdmi_guard.update(cap.camera_people[2] if cap.camera_people is not None and hdmi_valid else None,
                                                         time.perf_counter())
                            hdmi_guard.apply(hdmi_simulation,hdmi_people)
                            draw_people(hdmi_front,hdmi_people)
                    else:
                        hdmi_simulation.update(hdmi_states,bool(hdmi_lights))
                    view_decisions=(simulation.decision,other_simulation.decision,hdmi_simulation.decision)
                    view_pedestrians=((person_state['status'],other_people['status'],hdmi_people['status'])
                                      if person_enabled else ())
                    view_person_meta=tuple((len(item['people']),bool(item.get('tracking_lost',False)),
                        round(time.perf_counter()-item['captured_at'],3),item.get('source_frame',item['frame']))
                        if item is not None else None for item in
                        ((cap.camera_people or {}).get(channel) for channel in (0,1,2))) if person_enabled else ()
                    decision=simulation.decision
                    draw_driving_arrow(other_front,other_simulation,0)
                    draw_driving_arrow(hdmi_front,hdmi_simulation,0)
                    frame[360:,640:]=hdmi_front
                    for channel,view in enumerate((simulation,other_simulation,hdmi_simulation,hdmi_simulation)):
                        if view.alarm:
                            target=frame[(channel//2)*360:(channel//2+1)*360,
                                         (channel%2)*640:(channel%2+1)*640]
                            cv2.rectangle(target,(3,3),(target.shape[1]-4,target.shape[0]-4),(0,0,255),4)
                render_lights=lights+other_lights
                if args.pcie_capture and args.front_channel in (0,1):
                    render_lights=[{**light,'box_color':(180,180,180)} if light['color'] not in ('red','yellow','green') else light
                                   for light in render_lights if light['color'] in ('red','yellow','green') or light['direction'] in DIRECTIONS]
                if both_cameras:
                    render_lights+=hdmi_lights+[{**light,'box':[light['box'][0]+640,light['box'][1],
                        light['box'][2]+640,light['box'][3]]} for light in hdmi_lights]
                camera_assist_tick=time.perf_counter()
                duration = max(time.perf_counter() - started, .001)
                age = time.perf_counter() - latest_result["captured_at"] if latest_result else None
                mirror_bounds = ([PCIeCapture.channel_bounds(channel)
                                  for channel, source in enumerate(cap.video_status["source_channels"])
                                  if channel != args.front_channel and
                                  source == cap.video_status["source_channels"][args.front_channel] and
                                  cap.video_status["valid_mask"] & (1 << channel)]
                                 if args.pcie_capture and cap.video_status is not None else [])
                if args.show:
                    display_lights = ([{**light, "box": [light["box"][0] * frame.shape[1] / width,
                                                         light["box"][1] * frame.shape[0] / height,
                                                         light["box"][2] * frame.shape[1] / width,
                                                         light["box"][3] * frame.shape[0] / height]}
                                       for light in render_lights] if source_cap is not None else render_lights)
                    preview, preview_lights, preview_roi = display_canvas(
                        frame, display_lights, args.signal_roi, live_width,args.display_height-(70 if fusion else 0),
                        resizer=preview_resize)
                    display_bounds,display_mirrors,video_shape=front_bounds,mirror_bounds,frame.shape
                    if fusion:
                        content_height=preview.shape[0]
                        preview=cv2.copyMakeBorder(preview,0,70,0,0,cv2.BORDER_CONSTANT)
                        ratio=content_height/preview.shape[0]
                        display_bounds=([front_bounds[0],front_bounds[1]*ratio,front_bounds[2],front_bounds[3]*ratio]
                                        if front_bounds else (0,0,1,ratio))
                        display_mirrors=[(x1,y1*ratio,x2,y2*ratio) for x1,y1,x2,y2 in mirror_bounds]
                        video_shape=preview.shape
                    draw_overlay(preview, preview_lights, states, simulation, preview_roi,
                                 guidance_bounds=display_bounds, video_shape=video_shape,mirror_bounds=display_mirrors,
                                 input_channel=args.front_channel if args.pcie_capture and not both_cameras else None)
                    if analysis_width:
                        observation = draw_model_observation(latest_result, analysis_width, preview.shape[0],
                                                             time.perf_counter(), args.max_result_age)
                        preview = np.concatenate((preview, observation), axis=1)
                if not args.show or writer is not None:
                    draw_overlay(frame, render_lights, states, simulation, args.signal_roi, guidance_bounds=front_bounds,
                                 mirror_bounds=mirror_bounds, input_channel=args.front_channel if args.pcie_capture and not both_cameras else None)
                command_echo = None
                publish_tick=time.perf_counter()
                if both_cameras:
                    camera_profile.append((1000*(camera_capture_tick-tick),
                        1000*(camera_signal_tick-camera_capture_tick),1000*(camera_assist_tick-camera_signal_tick),
                        1000*(time.perf_counter()-camera_assist_tick),cap.prepare_ms))
                if fpga is not None:
                    valid=(states.get(simulation.nav) in ('red','yellow','green') if signal_active else
                           fusion and any(kind in ('SOLID','DASHED') for kind in lane_states.values()))
                    valid=valid or person_state['status']=='BRAKE'
                    valid=valid or simulation.guidance_visible
                    valid=valid or both_cameras and simulation.alarm
                    control_simulation=(select_fpga_control(simulation,other_simulation,hdmi_simulation)
                                        if both_cameras else simulation)
                    valid=valid or control_simulation.alarm
                    command_echo = fpga.publish(control_simulation, valid or control_simulation.guidance_visible)
                    if cpu_action is not None:
                        print(f"C_KEY1: {simulation.mode}, manual_moving={simulation.manual_moving} "
                              f"signal={states.get(simulation.nav)} decision={decision} "
                              f"alarm={int(simulation.alarm)} fpga_echo={command_echo:#010x}", flush=True)
                published_tick=time.perf_counter()
                if writer is not None:
                    writer.write(frame)
                elapsed_ms = (time.perf_counter() - tick) * 1000
                if events is not None:
                    events.write(json.dumps({"frame": current, "loop": loop_index, "source_time_s": current / fps,
                                         "signal_roi": args.signal_roi, "group_revision": group_revision,
                                         "lights": lights, "raw_states": raw, "states": states,
                                         "pedestrian":person_state, **simulation.snapshot(),
                                             "decision": decision, "processing_ms": elapsed_ms,
                                             "inference_frame": latest_result["frame"] if latest_result else None,
                                             "result_age_ms": age * 1000 if age is not None else None}) + "\n")
                if (not args.show and result is not None and processed % args.log_every == 0
                        or args.show and (time.perf_counter() - last_log >= 1 or decision != last_decision and not both_cameras)):
                    log_tick=time.perf_counter()
                    last_ms = latest_result["processing_ms"] if latest_result else 0
                    detected = len(latest_result["lights"]) if latest_result else 0
                    video_status = (f" input={next(name for name, channel in INPUT_CHANNELS.items() if channel == min(args.front_channel, 2))} "
                                    f"inputs={cap.video_status['valid_mask']:04b} "
                                    f"sources={cap.video_status['source_channels']} "
                                    f"source_frames={cap.video_status['good_frames']} dma={cap.dma_ms:.1f}ms"
                                    if args.pcie_capture and cap.video_status is not None else "")
                    if args.pcie_capture and args.front_channel in (0, 1) and cap.video_status is not None:
                        video_status += (f" screen={int(cap.video_status.get('camera_screen_active', False))}"
                                         f" revision={cap.video_status.get('camera_screen_revision', 0)}")
                    if both_cameras:
                        video_status+=(f" camera_inferred={camera_inferred} camera_reused={camera_geometry_reused} primary_states={view_states} other_states={other_states}"
                                       f" other_decision={other_simulation.decision} other_boxes={len(other_lights)}"
                                       f" hdmi_decision={hdmi_simulation.decision} hdmi_boxes={len(hdmi_lights)}"
                                       f" navs={(simulation.nav,other_simulation.nav,hdmi_simulation.nav)}"
                                       f" guidance={(simulation.guidance_visible,other_simulation.guidance_visible,hdmi_simulation.guidance_visible)}"
                                       f" hdmi_route={hdmi_simulation.navigation_direction} nav_manual={int(simulation.navigation_manual)}"
                                       f" view_routes={(simulation.navigation_direction,other_simulation.navigation_direction,hdmi_simulation.navigation_direction) if fusion else ()}"
                                       f" view_alarms={(simulation.alarm,other_simulation.alarm,hdmi_simulation.alarm)}"
                                       f" requested_nav={simulation.requested_nav}"
                                       f" other_lanes={tuple(other_boundaries.get(side,{}).get('kind','UNKNOWN') for side in ('LEFT','RIGHT')) if fusion else ()}"
                                       f" view_decisions={view_decisions} view_pedestrians={view_pedestrians}"
                                       f" view_person_meta={view_person_meta}"
                                       f" hdmi_lanes={tuple(hdmi_boundaries.get(side,{}).get('kind','UNKNOWN') for side in ('LEFT','RIGHT')) if fusion else ()}"
                                       f" prepare_detail={getattr(cap,'camera_prepare_costs',None)}"
                                       f" costs_ms={np.round(np.mean(camera_profile,axis=0),1).tolist()}"
                                       f" output_ms={np.round(np.mean(camera_output_profile,axis=0),1).tolist() if camera_output_profile else []} log_ms={camera_last_log_ms:.1f}"
                                       f" nav_ms={[round(reader.total_ms/max(reader.count,1),1) for reader in (navigation_reader,other_navigation) if reader is not None]}")
                    lane_status=(f" signal={processed/duration:.2f} FPS lanes={lane_states} "
                                 f"intent={'CHANGE' if simulation.lane_change else 'TURN'}" if fusion else '')
                    if person_enabled:
                        lane_status+=f" ped={person_count/duration:.2f} FPS pedestrian={person_state['status']} hazards={len(person_state['hazards'])}"
                    inference_rate=(lane_count if fusion else processed)/duration
                    inference_label='lane' if fusion else 'inference'
                    display_rate=displayed/duration
                    if both_cameras:
                        sample=time.perf_counter();camera_display_rates.append((sample,displayed))
                        while len(camera_display_rates)>1 and camera_display_rates[1][0]<=sample-3:
                            camera_display_rates.popleft()
                        elapsed=sample-camera_display_rates[0][0]
                        display_rate=(displayed-camera_display_rates[0][1])/elapsed if elapsed>0 else 0.
                        video_status+=f" display_avg={displayed/duration:.1f} elapsed_s={duration:.3f}"
                    print(f"display={display_rate:.1f} FPS {inference_label}={inference_rate:.2f} FPS{lane_status} "
                          f"| Frame {current + 1}/{total_frames or 'live'} | decoded={decoded} displayed={displayed} inferred={processed} "
                          f"detected={detected} "
                          f"| last={last_ms:.0f} ms age={age if age is not None else -1:.2f}s | {states} {decision} "
                          f"aligned={len(lights)} source_frame={latest_result['frame'] + 1 if latest_result else 0} "
                          f"colors={[light['color'] for light in lights]} color_frame={current + 1} "
                          f"directions={[light['direction'] for light in lights]} "
                          f"moving={int(simulation.moving)} alarm={int(simulation.alarm)} "
                          f"route={simulation.navigation_direction if fusion and simulation.navigation_enabled else 'OFF'} "
                          f"fpga_echo={hex(command_echo) if command_echo is not None else 'none'}{video_status}", flush=True)
                    last_log = time.perf_counter()
                    if both_cameras:camera_last_log_ms=1000*(last_log-log_tick)
                last_decision = decision
                if args.show:
                    show_tick=time.perf_counter()
                    display_backend.imshow("Traffic light direction", preview)
                    shown_tick=time.perf_counter()
                    if displayed == 0:
                        display_backend.moveWindow("Traffic light direction", 0, 0)
                    displayed += 1
                    deadline = started + displayed / fps
                    key = display_backend.waitKey(max(1, round((deadline - time.perf_counter()) * 1000))) & 0xFF
                    if both_cameras:
                        camera_output_profile.append((1000*(published_tick-publish_tick),
                            1000*(shown_tick-show_tick),1000*(time.perf_counter()-shown_tick)))
                    if key in (ord("q"), 27) or display_backend.getWindowProperty("Traffic light direction", cv2.WND_PROP_VISIBLE) < 1:
                        break
                    if requested_input is not None and key in (ord('1'), ord('2'), ord('3')):
                        requested_input[0] = key - ord('1')
                        continue
                    if key == ord("l"):
                        next_mode = "lane"
                        break
                    if key == ord('b') and not fusion:
                        next_mode='fusion'
                        break
                    if key == ord('t') and fusion:
                        next_mode='traffic'
                        break
                    if 65 <= key <= 90:
                        key += 32
                    if fusion and key==ord('v'):
                        simulation.command('TURN' if simulation.lane_change else 'CHANGE')
                    elif fusion and key==ord('h'):
                        roi=select_signal_group(frame,args.display_width,args.display_height,front_bounds)
                        if roi is not None:
                            from traffic_navigation import NavigationReader
                            bx,by,br,bb=front_bounds or (0,0,1,1)
                            args.navigation_roi=((roi[0]-bx)/(br-bx),(roi[1]-by)/(bb-by),
                                                 (roi[2]-bx)/(br-bx),(roi[3]-by)/(bb-by))
                            navigation_reader=NavigationReader(args.navigation_roi)
                            simulation.navigation_roi=args.navigation_roi
                            simulation.command('NAV ON')
                            simulation.navigation_direction='UNKNOWN'
                    elif fusion and key==ord('n'):
                        args.signal_roi=None
                        latest_result=None
                        group_revision+=1
                        live_frames.clear()
                        motion_history.clear()
                        live_filters={direction:StableState(args.confirm_frames) for direction in DIRECTIONS}
                    elif key == ord("g"):
                        roi = select_signal_group(frame, args.display_width, args.display_height, front_bounds)
                        if roi is not None:
                            if classifier is None:
                                classifier = DirectionClassifier(
                                    args.direction_weights, device, args.direction_conf, args.direction_margin, runtime=args.runtime)
                                analyzer.classifier = classifier
                            args.signal_roi = roi
                            args.direction_centers = args.circle_direction = None
                            latest_result = None
                            simulation.update({}, True)
                            group_revision += 1
                            live_frames.clear()
                            live_frames.append((current, loop_index, group_revision, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
                    elif key in KEY_ACTIONS:
                        simulation.command(KEY_ACTIONS[key])
                if args.max_display_frames and displayed >= args.max_display_frames:
                    break
                if args.max_frames and processed >= args.max_frames:
                    break
    finally:
        playback_seconds = time.perf_counter() - started
        if tracking_pool is not None:
            tracking_pool.shutdown(wait=True)
        sys.setswitchinterval(previous_switch_interval)
        if previous_affinity is not None:
            os.sched_setaffinity(0, previous_affinity)
        if fpga is not None:
            fpga.close()
        if cpu_key is not None:
            cpu_key.close()
        if args.show:
            display_backend.destroyAllWindows()
        cap.release()
        if camera_prepare_pool is not None:camera_prepare_pool.shutdown(wait=True)
        if worker is not None:
            worker.close()
        if lane_worker is not None:
            lane_worker.close()
        if person_worker is not None:
            person_worker.close()
        if person_model is not None:
            person_model.close()
        if lane_model is not None:
            lane_model.close()
        if source_cap is not None:
            source_cap.release()
        if original_inference_cap is not None:
            original_inference_cap.release()
        if writer is not None:
            writer.release()
        if board:
            detector.close()
            if classifier is not None:
                classifier.model.close()
            if hdmi_classifier is not None:hdmi_classifier.model.close()
    duration = time.perf_counter() - started
    if not decoded:
        raise RuntimeError(f"No frames decoded from source: {args.source}")
    report = {"frames": processed, "decoded_frames": decoded, "displayed_frames": displayed,
              "display_fps": displayed / playback_seconds, "playback_seconds": playback_seconds,
              "inferred_frames": processed,
              "lane_inferred_frames":lane_count,"lane_inference_fps":lane_count/playback_seconds,
              "signal_inference_fps":processed/playback_seconds,
              "mean_lane_ms":float(np.mean(lane_timings)) if lane_timings else None,
              "pedestrian_inferred_frames":person_count,"pedestrian_inference_fps":person_count/playback_seconds,
              "mean_pedestrian_ms":float(np.mean(person_timings)) if person_timings else None,
              "last_pedestrian_state":person_state['status'],
              "last_lane_states":lane_states,
              "navigation_reads":navigation_reader.count if navigation_reader else 0,
              "navigation_mean_ms":navigation_reader.total_ms/max(1,navigation_reader.count) if navigation_reader else None,
              "navigation_transitions":list(navigation_transitions),
              "wall_seconds": duration, "processing_fps": processed / duration,
              "source_fps": fps, "mean_frame_ms": float(np.mean(timings)) if timings else None,
              "p95_frame_ms": float(np.percentile(timings, 95)) if timings else None,
              "directions": direction_counts, "settings": vars(args),
              "backend": backend,
              "note": "Display and inference FPS are separate. Old results never grant green permission in live playback."}
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    return next_mode


def parse_opt():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assist-mode", choices=("traffic", "lane", "fusion"), default="traffic")
    parser.add_argument('--navigation-roi',nargs=4,type=float,
                        help='white guidance arrow region, normalized within the selected front image; H selects again')
    parser.add_argument("--pedestrian-weights", help="optional RKNN person detector in fusion mode")
    parser.add_argument("--pedestrian-confidence", type=float, default=.45)
    parser.add_argument("--lane-weights", type=Path, default=ROOT / "lane.rknn")
    parser.add_argument("--lane-confidence", type=float, default=0.35)
    parser.add_argument("--weights", type=Path, default=PROJECT / "direction_artifacts/generalization_delivery/color.pt")
    parser.add_argument("--direction-weights", type=Path, default=PROJECT / "direction_artifacts/generalization_delivery/direction.pt")
    parser.add_argument("--camera-direction-weights", type=Path, help="Optional camera-only direction model; HDMI keeps direction-weights")
    parser.add_argument('--camera-review-dir',type=Path,help='Optional local camera training review output')
    parser.add_argument('--camera-review-frames',type=int,default=24)
    source_options = parser.add_mutually_exclusive_group()
    source_options.add_argument("--source", default=str(PROJECT / "raw_video/2.mp4"))
    source_options.add_argument("--choose-video", action="store_true", help="select a local video in a file dialog")
    source_options.add_argument("--pcie-capture", action="store_true", help="read the live FPGA 1280x720 mosaic")
    parser.add_argument("--pipeline-capture", action="store_true", help="overlap PCIe capture with image processing")
    parser.add_argument("--native-motion", action="store_true", help="use the app-local optical-flow and display code")
    parser.add_argument('--native-display',action='store_true',help='optional X11 window for the joint native preview')
    parser.add_argument("--capture-library", type=Path, default=ROOT / "pcie_capture/libtraffic_capture.so")
    parser.add_argument("--dma-delay-loops", type=int, default=2500, help="existing PCIe capture DMA wait loops")
    parser.add_argument("--dma-chunk-bytes", type=int, choices=(2560,3840), default=2560,
                        help="3840 requires the chunk-capable FPGA and matching capture library")
    parser.add_argument("--auxiliary-fps", type=float, default=0, help="lane/person update limit; zero runs continuously")
    parser.add_argument("--signal-fps", type=float, default=0, help="traffic detector update limit; zero runs continuously")
    parser.add_argument("--realtime-threads", action='store_true', help="measured RK3568 display/capture/model CPU layout")
    parser.add_argument("--front-channel", type=int, choices=range(4), default=0,
                        help="mosaic channel used for traffic decisions: 0 upper left, 1 upper right, 2 lower left, 3 lower right")
    parser.add_argument('--input-source', choices=tuple(INPUT_CHANNELS), type=str.upper,
                        help='live recognition input: CAM1 or CAM2 (OV5640), or HDMI; overrides front-channel')
    parser.add_argument("--output", default=str(PROJECT / "direction_artifacts/demo.mp4"))
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf-thres", type=float, default=0.75)
    parser.add_argument("--iou-thres", type=float, default=0.45)
    parser.add_argument("--direction-conf", type=float, default=0.90)
    parser.add_argument("--direction-margin", type=float, default=0.25)
    parser.add_argument("--device", default="", help="PC device; unused for RKNN")
    parser.add_argument("--runtime", type=Path, default=Path("/usr/lib/librknnrt.so"),
                        help="existing RKNN runtime library; no installation or system changes")
    parser.add_argument("--cpu-runtime", choices=("opencv", "onnxruntime"), default="opencv",
                        help="FP32 color backend when using ONNX color and RKNN direction models")
    parser.add_argument("--tiles", type=int, choices=(1, 2), default=2)
    parser.add_argument("--top-ratio", type=float, default=0.58)
    group_options = parser.add_mutually_exclusive_group()
    group_options.add_argument("--signal-roi", nargs=4, type=float,
                               help="normalized x1 y1 x2 y2 for the relevant signal group")
    group_options.add_argument("--pick-signal-group", action="store_true", help="select the signal group with the mouse")
    parser.add_argument("--circle-direction", choices=DIRECTIONS, help="explicit round-lamp mapping for this scene")
    parser.add_argument("--direction-centers", nargs=3, type=float,
                        help="explicit calibrated x centers for LEFT STRAIGHT RIGHT; overrides shape classification")
    parser.add_argument("--nav", choices=DIRECTIONS, default="STRAIGHT")
    parser.add_argument("--moving", action="store_true")
    parser.add_argument("--loop", action="store_true", help="repeat a local video until Q/Esc or max-frames")
    parser.add_argument("--confirm-frames", type=int, default=3)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--no-record", action="store_true", help="display without saving video or per-frame logs")
    parser.add_argument("--display-width", type=int, default=1024)
    parser.add_argument("--observation-panel", action="store_true", help="show a separate diagnostic model image")
    parser.add_argument("--cpu-threads", type=int, default=0,
                        help="OpenCV/ORT threads; 0 preserves the default thread counts")
    parser.add_argument("--display-height", type=int, default=600)
    parser.add_argument("--max-result-age", type=float, default=2.5,
                        help="maximum tracked-housing age; live color still uses this frame, expired results become UNKNOWN")
    parser.add_argument("--log-every", type=int, default=1, help="headless progress interval in inferred frames")
    parser.add_argument("--max-display-frames", type=int, default=0, help="limit interactive playback for validation")
    parser.add_argument("--fpga-pci", help="PCI device address for the matching traffic FPGA bitstream")
    parser.add_argument("--vid-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()
    if args.input_source is not None:
        if not args.pcie_capture:
            parser.error('input-source requires live PCIe capture')
        args.front_channel = INPUT_CHANNELS[args.input_source]
    if args.navigation_roi:
        from traffic_navigation import NavigationReader
        try:NavigationReader(args.navigation_roi)
        except ValueError as error:parser.error(str(error))
        if args.assist_mode!='fusion':parser.error('navigation requires fusion mode')
    if not 0 < args.top_ratio <= 1 or args.vid_stride < 1 or args.max_frames < 0 or args.confirm_frames < 1:
        parser.error("invalid top-ratio, vid-stride, max-frames or confirm-frames")
    if (not 0 <= args.cpu_threads <= 4 or args.display_width < 320 or args.display_height < 360 or args.max_result_age <= 0
            or not np.isfinite(args.max_result_age) or args.log_every < 1 or args.max_display_frames < 0):
        parser.error("invalid display dimensions, result age or progress/frame limit")
    if not all(0 <= value <= 1 for value in (args.conf_thres, args.iou_thres, args.direction_conf, args.direction_margin,
                                           args.lane_confidence,args.pedestrian_confidence)):
        parser.error("thresholds must be between zero and one")
    if args.pick_signal_group and not args.show:
        parser.error("pick-signal-group requires --show")
    if args.pipeline_capture and not (args.pcie_capture and args.show and args.fpga_pci):
        parser.error("pipeline-capture requires live PCIe, --show and --fpga-pci")
    if args.native_motion and not args.pipeline_capture:
        parser.error("native-motion requires pipeline-capture")
    if args.native_display and not (args.native_motion and args.show and args.pcie_capture and
                                   args.pipeline_capture and args.front_channel in (0,1) and args.assist_mode!='lane'):
        parser.error('native-display requires the joint camera native preview with --show')
    if args.realtime_threads and not (args.pipeline_capture and args.assist_mode=='fusion'):
        parser.error('realtime-threads requires pipeline capture in fusion mode')
    if args.dma_delay_loops < 0:
        parser.error("dma-delay-loops cannot be negative")
    if any(not np.isfinite(value) or value<0 for value in (args.auxiliary_fps,args.signal_fps)):
        parser.error('model update limits must be finite and nonnegative')
    if args.pcie_capture:
        if args.loop:
            parser.error("Live PCIe capture cannot loop or seek")
        from traffic_rknn import PCIeCapture
        try:
            validate_roi(args.signal_roi, PCIeCapture.channel_bounds(args.front_channel))
        except ValueError as error:
            parser.error(str(error))
        args.source = "/dev/pango_pci_driver"
    if args.choose_video:
        import tkinter as tk
        from tkinter import filedialog

        window = tk.Tk()
        window.withdraw()
        try:
            args.source = filedialog.askopenfilename(
                parent=window, title="选择交通路口视频", initialdir=str(PROJECT / "raw_video"),
                filetypes=[("视频文件", "*.mp4 *.avi *.mkv *.mov *.wmv"), ("所有文件", "*.*")])
        finally:
            window.destroy()
        if not args.source:
            parser.exit(message="Video selection cancelled.\n")
    if args.direction_centers and (not args.signal_roi or not 0 <= args.direction_centers[0] < args.direction_centers[1] < args.direction_centers[2] <= 1):
        parser.error("direction-centers require signal-roi and increasing normalized LEFT STRAIGHT RIGHT centers")
    return args


def main(options=None):
    options = parse_opt() if options is None else options
    while True:
        selection = run(options)
        if selection is None:
            break
        if isinstance(selection, int):
            options.front_channel = selection
            options.input_source = next(name for name, channel in INPUT_CHANNELS.items() if channel == selection)
            # A new camera has different pixels and coordinates; never retain the old source's permissions.
            options.signal_roi = options.navigation_roi = None
            options.direction_centers = options.circle_direction = None
            options.pick_signal_group = False
        else:
            options.assist_mode = selection
        options.output = str(Path(options.output).with_name(f"{selection}_{time.time_ns()}.mp4"))


if __name__ == "__main__":
    main()
