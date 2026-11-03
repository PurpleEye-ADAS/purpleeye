"""Front-view person detection and simulated emergency braking, using the existing RKNN runtime."""
import time
import ctypes as ct
from pathlib import Path
from collections import deque
from types import SimpleNamespace

import cv2
import numpy as np

from traffic_rknn import RKNNModel


class PersonModel:
    def __init__(self, weights, runtime, confidence=.45):
        self.model = RKNNModel(weights, runtime)
        self.person_class = self.model.config['classes'].index('person')
        self.options = SimpleNamespace(imgsz=self.model.config['input_size'], conf_thres=confidence, iou_thres=.45)

    def infer(self, frame, number, captured_at):
        tick = time.perf_counter()
        rows = self.model.detect(frame, self.options)
        return {'frame': number, 'captured_at': captured_at, 'shape': frame.shape[:2],
                'people': [row for row in rows if row[5] == self.person_class],
                'processing_ms': (time.perf_counter() - tick) * 1000}

    def close(self):
        self.model.close()


class PersonTracker:
    """Move model-confirmed boxes with the image, without requiring visible body corners."""
    def __init__(self, library=None):
        self.observation = self.gray = self.number = None
        self.cut_frame = 0
        # One-second inference delays must still have a motion chain at high capture rates.
        self.frames = deque(maxlen=64)
        self.lazy_flow = library is not None
        self.flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
        self.flow.setFinestScale(0)
        self.flow.setPatchStride(6)
        self.native = ct.CDLL(str(Path(library).resolve())) if library is not None else None
        if self.native is not None:
            self.native.traffic_move_people.argtypes = [ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_int,
                ct.c_void_p,ct.c_int,ct.c_int,ct.c_double,ct.c_int,ct.c_int,ct.c_void_p]
            self.native.traffic_move_people.restype = ct.c_int
            self.native.traffic_gray_flow.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_int,ct.c_int,
                                                    ct.c_void_p,ct.c_void_p]
            self.native.traffic_gray_flow.restype=ct.c_int

    def move_native(self, previous, current, flow, people, fraction, shape):
        if not people:
            return []
        boxes = np.ascontiguousarray(people, dtype=np.float64)
        output = np.empty_like(boxes)
        count = self.native.traffic_move_people(previous.ctypes.data,current.ctypes.data,flow.ctypes.data,
            current.shape[1],current.shape[0],boxes.ctypes.data,len(boxes),boxes.shape[1],fraction,
            shape[1],shape[0],output.ctypes.data)
        if count < 0:
            raise RuntimeError('Native person motion failed')
        return output[:count].tolist()

    @staticmethod
    def move_people(previous, current, flow, people, fraction, shape):
        aligned = []
        height, width = current.shape
        for person in people:
            box = np.asarray(person[:4], dtype=float)/2
            x1, y1, x2, y2 = box
            left, top = max(0, round(x1+(x2-x1)*.15)), max(0, round(y1+(y2-y1)*.08))
            right, bottom = min(width, round(x2-(x2-x1)*.15)), min(height, round(y2-(y2-y1)*.4))
            if right>left and bottom>top:
                motion = np.median(flow[top:bottom, left:right].reshape(-1, 2), axis=0)*fraction
                box += np.tile(motion, 2)
                core = previous[top:bottom, left:right]
                patch = cv2.getRectSubPix(current, (right-left, bottom-top),
                                         ((left+right-1)/2+float(motion[0]), (top+bottom-1)/2+float(motion[1])))
                score = float(cv2.matchTemplate(patch, core, cv2.TM_CCOEFF_NORMED)[0, 0])
                difference = cv2.norm(patch, core, cv2.NORM_L1)/core.size
                # Weak texture is not negative evidence; clear only a strong disappearance.
                if core.std()>3 and score<.2 and difference>50:
                    continue
            box *= 2
            clipped = np.clip(box, [0, 0, 0, 0], [shape[1], shape[0], shape[1], shape[0]])
            if clipped[2]-clipped[0]<max(2, (box[2]-box[0])*.4) or clipped[3]-clipped[1]<8:
                continue
            aligned.append([*clipped.tolist(), *person[4:]])
        return aligned

    def update(self, frame, result, number):
        need_flow = (not self.lazy_flow or bool(result and result['people']) or
                     bool(self.observation and self.observation['people']))
        if self.native is not None:
            gray=np.empty((frame.shape[0]//2,frame.shape[1]//2),np.uint8)
            flow=np.empty((*gray.shape,2),np.float32) if need_flow else None
            status=self.native.traffic_gray_flow(self.gray.ctypes.data if self.gray is not None and need_flow else None,
                frame.ctypes.data,frame.shape[1],frame.shape[0],frame.strides[0],gray.ctypes.data,
                flow.ctypes.data if flow is not None else None)
            if status<0:raise RuntimeError('Native person optical flow failed')
            cut=(status==2 or not need_flow and self.gray is not None and
                 cv2.norm(gray,self.gray,cv2.NORM_L1)/gray.size>35)
            if status!=1:flow=None
        else:
            gray=cv2.resize(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),(frame.shape[1]//2,frame.shape[0]//2))
            cut=self.gray is not None and cv2.norm(gray,self.gray,cv2.NORM_L1)/gray.size>35
            flow=self.flow.calc(self.gray,gray,None) if self.gray is not None and not cut and need_flow else None
        if cut:
            self.observation=None
            self.frames.clear()
            self.cut_frame=number
        self.frames.append([self.number, number, None if cut else self.gray, gray, flow])
        if result is not None and result['frame']>=self.cut_frame:
            self.observation = dict(result, tracking_lost=False)
            chain = [item for item in self.frames if item[1]>result['frame']]
            start = result['frame']
        else:
            chain, start = [self.frames[-1]], self.number
        if self.observation is not None:
            for item in chain:
                source, target, previous, current, flow = item
                if previous is None:
                    continue
                if not self.observation['people']:
                    break
                if flow is None:
                    flow=self.flow.calc(previous,current,None)
                    item[4]=flow
                fraction = min(1, (target-start)/max(1, target-source))
                people = self.observation['people']
                move = self.move_native if self.native is not None else self.move_people
                aligned = move(previous, current, flow, people, fraction, frame.shape)
                self.observation['tracking_lost'] |= len(aligned)!=len(people)
                self.observation['people'] = aligned
                start = target
        self.gray, self.number = gray, number
        return self.observation


def conflict_polygon(shape):
    height, width = shape[:2]
    return np.float32([[.40, .48], [.60, .48], [.82, .98], [.18, .98]]) * [width, height]


def in_vehicle_path(box, shape):
    height, width = shape[:2]
    if (not np.isfinite(box[:4]).all() or box[2] <= box[0] or
            box[3] - box[1] < height * .08):
        return False
    foot = ((box[0] + box[2]) / 2, min(float(box[3]), height * .98))
    return cv2.pointPolygonTest(conflict_polygon(shape).astype(np.float32), foot, False) >= 0


class PedestrianGuard:
    """Brake on a conflict; release only after fresh clear observations and a short hold."""
    def __init__(self, max_age=1., hold_seconds=1., clear_frames=2):
        self.max_age, self.hold_seconds, self.clear_frames = max_age, hold_seconds, clear_frames
        self.last_frame = None
        self.last_hazard = -float('inf')
        self.clear_count = 0
        self.braking = False

    def update(self, result, now):
        fresh = result is not None and 0 <= now - result['captured_at'] <= self.max_age
        people = result['people'] if fresh else []
        hazards = [row for row in people if in_vehicle_path(row, result['shape'])] if fresh else []
        new_observation = fresh and result['frame'] != self.last_frame
        lost = fresh and result.get('tracking_lost', False)
        if new_observation:
            self.last_frame = result['frame']
        if hazards:
            self.braking = True
            self.last_hazard = now
            self.clear_count = 0
        elif new_observation and not lost:
            self.clear_count += 1
            if self.clear_count >= self.clear_frames and now - self.last_hazard >= self.hold_seconds:
                self.braking = False
        status = ('BRAKE' if self.braking else 'CLEAR' if fresh and not lost and self.clear_count >= self.clear_frames else 'UNKNOWN')
        return {'status': status, 'people': people, 'hazards': hazards,
                'shape': result['shape'] if result is not None else None}

    @staticmethod
    def apply(simulation, observation):
        if simulation.manual_moving is False:
            return
        if observation['status'] == 'BRAKE':
            simulation.moving = False
            simulation.alarm = True
            simulation.decision = 'BRAKE_PEDESTRIAN'
        elif observation['status'] == 'UNKNOWN':
            simulation.moving = False
            if not simulation.alarm:
                simulation.decision = 'WAIT_PEDESTRIAN'


def draw_people(frame, observation):
    if observation['shape'] is None:
        return
    height, width = observation['shape']
    gain = np.array([frame.shape[1] / width, frame.shape[0] / height] * 2)
    for person in observation['people']:
        box = np.rint(np.asarray(person[:4]) * gain).astype(int)
        color = (20, 20, 255) if in_vehicle_path(person, observation['shape']) else (0, 210, 255)
        cv2.rectangle(frame, tuple(box[:2]), tuple(box[2:]), color, 2)
        cv2.putText(frame, 'PERSON', (box[0], max(15, box[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1, cv2.LINE_AA)
    if observation['status'] == 'BRAKE':
        cv2.putText(frame, 'PEDESTRIAN - BRAKE', (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .65, (20, 20, 255), 2, cv2.LINE_AA)
