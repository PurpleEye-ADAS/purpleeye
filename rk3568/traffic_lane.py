"""Lane assistance using actual instance masks and the existing traffic controls."""
import json
import ctypes as ct
import time
from pathlib import Path

import cv2
import numpy as np

from traffic_rknn import RKNNModel, letterbox_rgb, normalized_input, nms
from traffic_light_3way_v2_modified import (
    VehicleSimulation, LatestFrameInference, KEY_ACTIONS, draw_overlay,
    handle_mouse, CONTROLS, preview_dimensions,
)


def decode_instances(outputs, image_shape, size, gain, padding, threshold=.35):
    """Decode YOLO11's four classes and mask prototypes, including letterbox crop."""
    input_height, input_width = (size,size) if np.isscalar(size) else size
    if len(outputs) == 10:
        decoded = []
        for i, stride in enumerate((8, 16, 32)):
            distances, scores, coefficients = outputs[i*3:i*3+3]
            h,w=distances.shape[-2:]
            scores=1/(1+np.exp(-np.clip(scores,-80,80)))
            selected=np.flatnonzero(scores.max(axis=1).reshape(-1)>=threshold)
            if not selected.size:
                continue
            logits=distances.reshape(1,4,16,-1)[:,:,:,selected]
            probabilities=np.exp(logits-logits.max(2,keepdims=True))
            probabilities/=probabilities.sum(2,keepdims=True)
            distances=(probabilities*np.arange(16,dtype=np.float32).reshape(1,1,16,1)).sum(2)
            xx=(selected%w).astype(np.float32)[None]+.5
            yy=(selected//w).astype(np.float32)[None]+.5
            x1,y1=(xx-distances[:,0])*stride,(yy-distances[:,1])*stride
            x2,y2=(xx+distances[:,2])*stride,(yy+distances[:,3])*stride
            boxes=np.stack([(x1+x2)/2,(y1+y2)/2,x2-x1,y2-y1],axis=1)
            decoded.append(np.concatenate([boxes,scores.reshape(1,4,-1)[:,:,selected],
                coefficients.reshape(1,32,-1)[:,:,selected]],axis=1))
        outputs=[np.concatenate(decoded,axis=2) if decoded else np.empty((1,40,0),np.float32),outputs[-1]]
    predictions, prototypes = outputs
    if predictions.ndim != 3 or predictions.shape[1] != 40 or prototypes.shape[:2] != (1, 32):
        raise ValueError(f'Unexpected lane outputs: {[x.shape for x in outputs]}')
    rows = predictions[0].T
    classes = rows[:, 4:8].argmax(1)
    scores = rows[np.arange(len(rows)), 4 + classes]
    indices = np.flatnonzero(scores >= threshold)
    candidates = []
    for index in indices:
        cx, cy, width, height = rows[index, :4]
        candidates.append([float(cx-width/2), float(cy-height/2), float(cx+width/2),
                           float(cy+height/2), float(scores[index]), int(classes[index]), int(index)])
    kept = nms(candidates, .5, max_det=24)
    height, width = image_shape[:2]
    dx, dy = padding
    ph, pw = prototypes.shape[-2:]
    results = []
    for x1, y1, x2, y2, score, cls, index in kept:
        logits = rows[index, 8:] @ prototypes[0].reshape(32, -1)
        mask = logits.reshape(ph, pw)
        yy, xx = np.ogrid[:ph, :pw]
        mask = np.where((xx >= x1*pw/input_width) & (xx < x2*pw/input_width) &
                        (yy >= y1*ph/input_height) & (yy < y2*ph/input_height), mask, -100)
        # Interpolate logits before thresholding; thresholding the small mask
        # first erases narrow, distant dashed markings.
        mask = cv2.resize(mask, (input_width, input_height), interpolation=cv2.INTER_LINEAR)
        left, top = round(dx-.1), round(dy-.1)
        right, bottom = left+round(width*gain), top+round(height*gain)
        mask = cv2.resize(mask[top:bottom, left:right], (width, height), interpolation=cv2.INTER_LINEAR) > 0
        results.append({'mask': mask, 'kind': 'DASHED' if cls % 2 == 0 else 'SOLID',
                        'score': score, 'model_class': cls})
    return results


def ego_boundaries(instances, shape):
    """Select nearby left/right paint by geometry, independently of class side names."""
    height, width = shape[:2]
    candidates = {'LEFT': [], 'RIGHT': []}
    unique=[]
    for item in sorted(instances,key=lambda x:x['score'],reverse=True):
        area=int(item['mask'].sum())
        duplicate=False
        for previous,previous_area in unique:
            if previous['kind']==item['kind'] or previous['score']-item['score']<.15:
                continue
            if min(area,previous_area)>0 and min(area,previous_area)/max(area,previous_area)>.5:
                overlap=int((item['mask']&previous['mask']).sum())
                if overlap/min(area,previous_area)>.85:
                    duplicate=True
                    break
        if not duplicate:unique.append((item,area))
    for item,_ in unique:
        mask = item['mask'].copy()
        mask[:round(height*.43)] = False
        mask[round(height*.93):] = False
        yy, xx = np.nonzero(mask)
        if len(yy) < max(12, height*width*.00008):
            continue
        counts = np.bincount(yy, minlength=height)
        rows = np.flatnonzero(counts)
        if len(rows) < max(6, height*.015) or np.ptp(rows) < height*.025:
            continue
        # nonzero returns row-major order, so each row's x values are already
        # sorted. Select its middle pair without scanning the mask per row.
        offsets = np.cumsum(counts)-counts
        lower = offsets[rows]+(counts[rows]-1)//2
        upper = offsets[rows]+counts[rows]//2
        medians = (xx[lower].astype(np.float64)+xx[upper])*.5
        fit = np.polyfit(rows/height, medians/width, 1)
        if np.median(abs(np.polyval(fit, rows/height) - medians/width)) > .025:
            continue
        base_x = float(np.polyval(fit, .86))
        side = 'LEFT' if -.25 < base_x < .48 else 'RIGHT' if .52 < base_x < 1.25 else None
        if side is None or abs(fit[0]) < .06 or abs(fit[0]) > 3:
            continue
        # Lane paint extends away from the vehicle towards the vanishing region.
        if side == 'LEFT' and fit[0] > .2 or side == 'RIGHT' and fit[0] < -.2:
            continue
        candidates[side].append({**item, 'mask': mask, 'base_x': base_x,
                                 'slope': float(fit[0]), 'bottom': float(rows.max()/height)})
    result = {}
    for side, options in candidates.items():
        if not options:
            result[side] = {'kind': 'UNKNOWN', 'mask': np.zeros((height, width), bool), 'score': 0.}
            continue
        nearest = min(options, key=lambda x: abs(x['base_x']-.5) + .12*(.93-x['bottom']))
        group = [x for x in options if abs(x['base_x']-nearest['base_x']) < .055
                 and abs(x['slope']-nearest['slope']) < .25]
        votes = {kind: sum(x['score']*np.sqrt(x['mask'].sum()) for x in group if x['kind'] == kind)
                 for kind in ('SOLID', 'DASHED')}
        kind = max(votes, key=votes.get)
        dominance = votes[kind]/sum(votes.values())
        if dominance < .72:
            kind = 'UNKNOWN'
        evidence = np.logical_or.reduce([x['mask'] for x in group])
        result[side] = {'kind': kind, 'mask': evidence, 'score': max(x['score'] for x in group),
                        'base_x': nearest['base_x'], 'slope': nearest['slope'], 'dominance': dominance}
    return result


def paint_evidence(frame, slope, base_x, channels=None, paint_top=0, native=None):
    """Measure actual bright paint against adjacent asphalt in a tracked corridor."""
    height,width=frame.shape[:2]
    gray,hsv=channels if channels is not None else (cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),None)
    if native is not None:
        if (frame.ndim!=3 or frame.shape[2]!=3 or frame.dtype!=np.uint8 or
                gray.shape!=(height,width) or gray.dtype!=np.uint8 or
                hsv is not None and (hsv.shape!=frame.shape or hsv.dtype!=np.uint8)):
            raise ValueError('Native lane evidence requires matching uint8 BGR, gray and HSV images')
        pixels=frame if frame.strides[1:]==(3,1) else np.ascontiguousarray(frame)
        gray=np.ascontiguousarray(gray)
        hsv=np.ascontiguousarray(hsv) if hsv is not None else None
        mask=np.empty((height,width),np.uint8);metrics=np.empty(4,np.float64)
        status=native.traffic_lane_evidence(pixels.ctypes.data,gray.ctypes.data,
            hsv.ctypes.data if hsv is not None else None,width,height,pixels.strides[0],
            slope,base_x,paint_top,mask.ctypes.data,metrics.ctypes.data)
        if status!=1:raise RuntimeError(native.traffic_capture_error().decode(errors='replace'))
        return mask.view(np.bool_),*(bool(value) for value in metrics[:3]),float(metrics[3])
    ys=np.arange(round(height*.45),round(height*.92))
    centers=(base_x+slope*(ys/height-.86))*width
    radius=np.maximum(2,np.rint(width*(.002+.007*ys/height)).astype(int))
    offsets=np.arange(-8,9)
    xs=np.rint(centers[:,None]+offsets).astype(int)
    inside=(xs>=0)&(xs<width)
    clipped=np.clip(xs,0,width-1)
    luminance=gray[ys[:,None],clipped].astype(np.float32)
    flank_offsets=np.array([-15,-12,-10,10,12,15])
    flanks=np.clip(np.rint(centers[:,None]+flank_offsets).astype(int),0,width-1)
    background=np.median(gray[ys[:,None],flanks],axis=1)
    colors=(hsv[ys[:,None],clipped] if hsv is not None else
            cv2.cvtColor(frame[ys[:,None],clipped],cv2.COLOR_BGR2HSV))
    white=(colors[:,:,1]<120)&(colors[:,:,2]>60)
    yellow=(colors[:,:,0]>=12)&(colors[:,:,0]<=42)&(colors[:,:,1]>65)&(colors[:,:,2]>60)
    paint=inside&(abs(offsets)[None,:]<=radius[:,None])&(white|yellow)
    paint &= luminance-background[:,None]>=16
    mask=np.zeros((height,width),bool)
    mask[ys[:,None],clipped]=paint
    # Bright car edges and reflections may cross the corridor. Only connected
    # paint with a measurable roadward span is rendered as a lane marking.
    start,end=int(ys[0]),int(ys[-1])+1
    columns=clipped[paint]
    if columns.size:
        left,right=int(columns.min()),int(columns.max())+1
        _,labels,stats,_=cv2.connectedComponentsWithStats(mask[start:end,left:right].astype(np.uint8),connectivity=8)
        keep=(stats[:,cv2.CC_STAT_HEIGHT]>=max(3,round(height*.008))) & (
            stats[:,cv2.CC_STAT_AREA]>=max(4,round(height*width*.000018)))
        keep[0]=False
        expected=slope*width/height
        for label in np.flatnonzero(keep):
            x,y,w,h,_=stats[label]
            yy,xx=np.nonzero(labels[y:y+h,x:x+w]==label)
            dy=yy-yy.mean()
            measured=float(dy@(xx-xx.mean())/(dy@dy))
            keep[label]=(1+measured*expected)/np.sqrt((1+measured**2)*(1+expected**2))>=.9
        mask[start:end,left:right]=keep[labels]
    # Geometry can verify paint between model updates, but its extrapolation
    # beyond the observed marking must not be rendered onto distant objects.
    occupied=mask[ys].any(1)
    mask[:paint_top]=False
    changes=np.diff(np.r_[False,occupied,False].astype(np.int8))
    starts,ends=np.flatnonzero(changes==1),np.flatnonzero(changes==-1)
    runs=[(int(a),int(b)) for a,b in zip(starts,ends) if b-a>=max(3,round(height*.02))]
    visible=int(occupied.sum())>=max(6,round(height*.017))
    coverage=float(occupied.mean())
    # A continuous, visible boundary is a prohibition even when the network
    # momentarily calls its short crop dashed. Occlusion alone cannot grant a turn.
    solid=visible and coverage>=.74 and any(b-a>=height*.20 for a,b in runs)
    # One interruption can be a vehicle or a shadow on a solid line. Reversing
    # a confirmed prohibition requires multiple visible paint segments/gaps.
    dashed=visible and sum(runs[i+1][0]-runs[i][1]>=height*.018 for i in range(len(runs)-1))>=2
    return mask,visible,solid,dashed,coverage


def paint_boundaries(frame):
    """Propose missing lane geometry from nearby paint, without borrowing another source."""
    height,width=frame.shape[:2]
    scale=min(1.,320/width)
    image=cv2.resize(frame,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA) if scale<1 else frame
    h,w=image.shape[:2]
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    contrast=gray.astype(np.int16)-cv2.boxFilter(gray,-1,(15,15)).astype(np.int16)
    color=((hsv[:,:,1]<120)|((hsv[:,:,0]>=12)&(hsv[:,:,0]<=42)&(hsv[:,:,1]>65)))&(hsv[:,:,2]>60)
    mask=(color&(contrast>=12)).astype(np.uint8)*255
    mask[:round(h*.43)]=0;mask[round(h*.93):]=0
    lines=cv2.HoughLinesP(mask,1,np.pi/180,threshold=12,minLineLength=max(12,round(h*.12)),
                          maxLineGap=max(4,round(h*.06)))
    if lines is None:return {}
    candidates={'LEFT':[],'RIGHT':[]}
    for x1,y1,x2,y2 in lines.reshape(-1,4):
        if abs(y2-y1)<h*.12 or max(y1,y2)<h*.67:continue
        slope=(x2-x1)/(y2-y1)*h/w
        base=(x1+(h*.86-y1)*(x2-x1)/(y2-y1))/w
        horizon=base+slope*(.43-.86)
        if not .25<horizon<.75:continue
        side=('LEFT' if .08<base<.48 and -2.5<slope<-.12 else
              'RIGHT' if .52<base<.92 and .12<slope<2.5 else None)
        if side is not None:candidates[side].append((abs(base-.5),-abs(y2-y1),slope,base,min(y1,y2)))
    result={}
    for side,items in candidates.items():
        if not items:continue
        _,_,slope,base,top=min(items)
        paint,visible,solid,gaps,coverage=paint_evidence(image,slope,base,(gray,hsv),top)
        if not visible or not (solid or gaps):continue
        rows=np.flatnonzero(paint.any(1))
        centers=(base+slope*(rows/h-.86))*w
        flanks=np.clip(np.rint(centers[:,None]+np.array([-15,-12,-10,10,12,15])).astype(int),0,w-1)
        sides=np.maximum(np.median(gray[rows[:,None],flanks[:,:3]],axis=1),
                         np.median(gray[rows[:,None],flanks[:,3:]],axis=1))
        foreground=np.where(paint[rows],gray[rows],0).max(1).astype(float)
        # A broad white vehicle panel is bright on one flank; actual paint is narrow.
        if np.mean(foreground-sides>=16)<.8:continue
        result[side]={'kind':'SOLID' if solid else 'DASHED','mask':cv2.resize(paint.astype(np.uint8),
            (width,height),interpolation=cv2.INTER_NEAREST).astype(bool),'score':.51,
            'base_x':base,'slope':slope,'paint_coverage':coverage,'source':'current_paint_geometry'}
    return result


class LaneEvidenceTracker:
    """Confirm new dashed permissions and bridge misses using current paint evidence."""
    def __init__(self, library=None):
        self.history={}
        self.previous_gray=None
        self.cut_frame=0
        self.native=ct.CDLL(str(Path(library).resolve())) if library is not None else None
        if self.native is not None:
            self.native.traffic_lane_evidence.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_void_p,
                ct.c_int,ct.c_int,ct.c_int,ct.c_double,ct.c_double,ct.c_int,ct.c_void_p,ct.c_void_p]
            self.native.traffic_lane_evidence.restype=ct.c_int
            self.native.traffic_capture_error.argtypes=[]
            self.native.traffic_capture_error.restype=ct.c_char_p

    def update(self, frame, observations, observation_frame, current_frame, now, grayscale=None):
        channels=(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) if grayscale is None else grayscale,None)
        gray=cv2.resize(channels[0],(80,45))
        low_light=float(np.median(gray[18:42]))<50
        if self.previous_gray is not None:
            delta=abs(gray.astype(np.int16)-self.previous_gray.astype(np.int16))
            if float(delta[18:42].mean())>35:
                self.history.clear()
                self.cut_frame=current_frame
        self.previous_gray=gray
        if observation_frame is None or observation_frame<self.cut_frame:
            observations={}
        result={}
        for side in ('LEFT','RIGHT'):
            item=observations.get(side,{})
            state=self.history.get(side)
            fresh=state is None or observation_frame!=state.get('model_frame')
            kind=item.get('kind','UNKNOWN')
            model_span=0
            if item.get('mask',np.empty(0)).any() and 'base_x' in item:
                if state is None or abs(item['base_x']-state['base_x'])>.12:
                    state={'kind':'UNKNOWN','streak':0,'last_model':now,'model_frame':None}
                rows=np.flatnonzero(item['mask'].any(1))
                model_span=int(rows[-1]-rows[0]+1)
                state.update(base_x=item['base_x'],slope=item['slope'],
                             paint_top=int(rows[0]))
                if fresh and kind in ('SOLID','DASHED'):
                    state['last_model']=now
            if state is None or 'base_x' not in state or now-state['last_model']>2.:
                result[side]={'kind':'UNKNOWN','mask':np.zeros(frame.shape[:2],bool),'score':0.,'empty':True}
                continue
            mask,visible,solid,gaps,coverage=paint_evidence(
                frame,state['slope'],state['base_x'],channels,state['paint_top'],self.native)
            if state['kind']=='DASHED' and now-state.get('last_dashed',-float('inf'))>2.:
                state.update(kind='UNKNOWN',streak=0)
            short_solid=kind=='SOLID' and model_span<frame.shape[0]*.25
            uncertain_solid=(kind=='SOLID' and ((low_light or short_solid) and not solid or
                             item.get('score',0.)<.5 and (state['kind']=='DASHED' or not solid)))
            if kind=='SOLID' and not uncertain_solid or solid and state['kind']!='DASHED':
                state.update(kind='SOLID',streak=0)
                if fresh or solid:state['last_solid']=now
            elif fresh and kind=='DASHED' and item.get('score',0.)>=.5:
                state['last_dashed']=now
                state['streak']+=1
                required=3 if state['kind']=='SOLID' else 2
                if state['streak']>=required and visible and (state['kind']!='SOLID' or gaps):
                    state['kind']='DASHED'
            elif fresh and kind in ('UNKNOWN','DASHED'):
                state['streak']=0
            state['model_frame']=observation_frame
            self.history[side]=state
            confirmed=state['kind'] if visible and not uncertain_solid else 'UNKNOWN'
            if solid and kind!='DASHED' and state['kind']=='DASHED':
                confirmed='UNKNOWN'
            # Solid evidence remains a stop instruction if paint is briefly
            # occluded; a dashed permission always requires visible paint now.
            # Fresh dark/short model geometry cannot renew an unsupported solid prohibition.
            solid_seen=state.get('last_solid',-float('inf')) if low_light or short_solid else state['last_model']
            if state['kind']=='SOLID' and now-solid_seen<=.6:
                confirmed='SOLID'
            result[side]={'kind':confirmed,'mask':mask,'score':item.get('score',0.),
                          'base_x':state['base_x'],'slope':state['slope'],'paint_coverage':coverage,
                          'source':'current_paint' if confirmed!='UNKNOWN' else 'unconfirmed'}
        return result


class LaneModel:
    def __init__(self, weights, runtime, threshold=.35):
        self.weights = Path(weights)
        self.threshold = threshold
        if self.weights.suffix == '.rknn':
            self.model = RKNNModel(self.weights, runtime)
            self.size = self.model.config['input_size']
            self.shape = tuple(self.model.config.get('input_shape', [self.size,self.size]))
            self.backend = 'RKNN NPU'
        elif self.weights.suffix == '.onnx':
            import onnxruntime as ort
            options = ort.SessionOptions()
            options.intra_op_num_threads = 4
            self.model = ort.InferenceSession(str(self.weights), options, providers=['CPUExecutionProvider'])
            self.size = self.model.get_inputs()[0].shape[-1]
            self.shape = tuple(self.model.get_inputs()[0].shape[-2:])
            self.backend = 'ONNX CPU'
        else:
            from ultralytics import YOLO
            self.model = YOLO(str(self.weights))
            self.size = 640
            self.shape = (640,640)
            self.backend = 'PyTorch'

    def infer(self, frame, number=None, captured_at=None):
        started = time.perf_counter()
        if self.weights.suffix == '.pt':
            res = self.model.predict(frame, imgsz=self.size, conf=self.threshold, iou=.5, rect=False, verbose=False)[0]
            instances = []
            if res.masks is not None:
                for mask, cls, conf in zip(res.masks.data.cpu().numpy(), res.boxes.cls.cpu().numpy(),
                                            res.boxes.conf.cpu().numpy()):
                    mh, mw = mask.shape
                    gain = min(mh/frame.shape[0], mw/frame.shape[1])
                    dh, dw = round(frame.shape[0]*gain), round(frame.shape[1]*gain)
                    top, left = (mh-dh)//2, (mw-dw)//2
                    mask = cv2.resize(mask[top:top+dh, left:left+dw], (frame.shape[1], frame.shape[0])) > .5
                    instances.append({'mask': mask, 'kind': 'DASHED' if int(cls)%2 == 0 else 'SOLID',
                                      'score': float(conf), 'model_class': int(cls)})
        else:
            rgb, gain, padding = letterbox_rgb(frame, self.shape)
            outputs = (self.model.infer_rgb(rgb) if self.weights.suffix == '.rknn' else
                       self.model.run(None, {self.model.get_inputs()[0].name:
                                            normalized_input(rgb, {'input_scale': 255})}))
            instances = decode_instances(outputs, frame.shape, self.shape, gain, padding, self.threshold)
        boundaries=ego_boundaries(instances,frame.shape)
        if any('base_x' not in item for item in boundaries.values()):
            for side,item in paint_boundaries(frame).items():
                selected=boundaries[side]
                if ('base_x' not in selected or
                        abs(item['base_x']-.5)<abs(selected['base_x']-.5)):
                    boundaries[side]=item
        return {'boundaries': boundaries, 'instances': instances,
                'processing_ms': (time.perf_counter()-started)*1000,
                'frame':number, 'captured_at':captured_at}

    def close(self):
        if self.weights.suffix == '.rknn':
            self.model.close()


class LaneSimulation(VehicleSimulation):
    def update_lane(self, boundaries, action=None):
        if action is not None:
            self.command(action)
        self.alarm = False
        kind = boundaries.get(self.nav, {}).get('kind', 'UNKNOWN')
        known = any(x['kind'] in ('SOLID', 'DASHED') for x in boundaries.values())
        allowed = known if self.nav == 'STRAIGHT' else kind == 'DASHED'
        self.moving = allowed if self.manual_moving is None else self.manual_moving
        if self.manual_moving is False:
            self.decision = 'STOP_MANUAL'
        elif self.nav == 'STRAIGHT':
            self.decision = 'GO_STRAIGHT' if known else 'WAIT_UNKNOWN'
        elif kind == 'SOLID':
            self.decision = 'ALARM_SOLID'
            self.alarm = True
        elif kind == 'DASHED':
            self.decision = 'GO_'+self.nav
        elif self.manual_moving is True:
            self.decision = 'ALARM_UNKNOWN_LANE'
            self.alarm = True
            self.moving = False
        else:
            self.decision = 'WAIT_UNKNOWN'
        return self.decision


class FusionSimulation(LaneSimulation):
    """Keep a junction turn distinct from crossing a lane boundary."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lane_change=False
        self.navigation_enabled=False
        self.navigation_manual=False
        self.navigation_direction='UNKNOWN'
        self.navigation_roi=None
        self.navigation_peers=()

    @property
    def controls(self):
        controls=CONTROLS+('TURN' if self.lane_change else 'CHANGE',)
        return controls+('NAV OFF' if self.navigation_enabled else 'NAV ON',) if self.navigation_roi else controls

    def has_navigation_context(self):
        return self.navigation_enabled or any(peer.navigation_enabled for peer in self.navigation_peers)

    def command(self, action):
        if action in ('NAV ON','NAV OFF'):
            self.navigation_enabled=action=='NAV ON'
            self.navigation_manual=False
        elif action in ('TURN','CHANGE'):
            self.lane_change=action=='CHANGE'
        else:
            if self.has_navigation_context() and action in ('LEFT','STRAIGHT','RIGHT'):
                self.navigation_manual=True
            elif action=='AUTO':
                self.navigation_manual=False
            super().command(action)

    def update_fused(self, boundaries, signals, signal_active, action=None):
        route=self.navigation_enabled and not self.navigation_manual
        self.nav=(self.navigation_direction if route and self.navigation_direction in ('LEFT','STRAIGHT','RIGHT')
                  else self.requested_nav)
        self.update_lane(boundaries, action)
        lane_state=(self.decision,self.moving,self.alarm)
        if (route and not self.lane_change and not signal_active and self.manual_moving is not False
                and self.navigation_direction in ('LEFT','STRAIGHT','RIGHT')):
            self.decision='GO_'+self.nav
            self.moving=True
            self.alarm=False
        if signal_active:
            VehicleSimulation.update(self,signals,True)
            if self.guidance_visible and self.lane_change:
                self.decision,self.moving,self.alarm=lane_state
        if route and self.navigation_direction=='UNKNOWN' and not self.alarm and self.manual_moving is not False:
            self.decision='WAIT_NAVIGATION'
            if self.manual_moving is None:self.moving=False
        return self.decision


def draw_lanes(frame, boundaries, native=None):
    colors = {'SOLID': (0, 160, 255), 'DASHED': (100, 235, 65), 'UNKNOWN': (160, 160, 160)}
    for side, item in boundaries.items():
        if item.get('empty',False):
            continue
        mask=item['mask']
        if native is not None:
            import ctypes as ct
            function=native.traffic_paint_lane
            function.argtypes=[ct.c_void_p,ct.c_int,ct.c_int,ct.c_int,ct.c_void_p,ct.c_int,ct.c_int,
                               ct.c_int,ct.c_int,ct.c_int]
            function.restype=ct.c_int
            selected=np.ascontiguousarray(mask.view(np.uint8) if mask.dtype==np.bool_ else mask,dtype=np.uint8)
            status=function(frame.ctypes.data,frame.shape[1],frame.shape[0],frame.strides[0],selected.ctypes.data,
                            selected.shape[1],selected.shape[0],*colors[item['kind']])
            if status!=1:raise RuntimeError('Native lane display failed')
            continue
        if mask.shape!=frame.shape[:2]:
            mask=cv2.resize(mask.astype(np.uint8),(frame.shape[1],frame.shape[0]),
                            interpolation=cv2.INTER_NEAREST).astype(bool)
        color = colors[item['kind']]
        yy,xx=np.nonzero(mask)
        if not yy.size:
            continue
        frame[yy,xx] = (.4*frame[yy,xx]+.6*np.array(color)).astype(np.uint8)
        left,right,top,bottom=int(xx.min()),int(xx.max())+1,int(yy.min()),int(yy.max())+1
        contours, _ = cv2.findContours(mask[top:bottom,left:right].astype(np.uint8),cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)
        contours=[contour+np.array([[[left,top]]],dtype=contour.dtype) for contour in contours]
        cv2.drawContours(frame, contours, -1, color, 1)


def lane_canvas(mosaic, front, boundaries, simulation, channel, status, max_width, max_height):
    """Overlay the selected physical source once and reuse its mirrored view."""
    from traffic_rknn import PCIeCapture
    picture=front.copy()
    draw_lanes(picture,boundaries)
    if mosaic is None:
        canvas=cv2.copyMakeBorder(picture,0,70,0,0,cv2.BORDER_CONSTANT)
        draw_overlay(canvas,[],{},simulation,None)
        return canvas
    canvas=mosaic.copy()
    sources=status['source_channels']
    channels=[i for i,source in enumerate(sources)
              if source==sources[channel] and status['valid_mask']&(1<<i)]
    h,w=canvas.shape[:2]
    for i in channels:
        x1,y1,x2,y2=PCIeCapture.channel_bounds(i)
        canvas[round(y1*h):round(y2*h),round(x1*w):round(x2*w)]=cv2.resize(picture,(w//2,h//2))
    size=preview_dimensions(canvas.shape,max_width,max_height-70)
    canvas=cv2.resize(canvas,size)
    canvas=cv2.copyMakeBorder(canvas,0,70,0,0,cv2.BORDER_CONSTANT)
    ratio=size[1]/canvas.shape[0]
    def bounds(i):
        x1,y1,x2,y2=PCIeCapture.channel_bounds(i)
        return x1,y1*ratio,x2,y2*ratio
    draw_overlay(canvas,[],{},simulation,None,guidance_bounds=bounds(channel),video_shape=canvas.shape,
                 mirror_bounds=[bounds(i) for i in channels if i!=channel], input_channel=channel)
    return canvas


def run(args):
    from traffic_rknn import PCIeCapture
    from traffic_fpga import FPGAInteraction, CPUMotionKey
    cv2.setNumThreads(1)
    model = LaneModel(args.lane_weights, args.runtime, args.lane_confidence)
    cap = (PCIeCapture(args.capture_library, args.source, args.dma_delay_loops, args.fpga_pci)
           if args.pcie_capture else cv2.VideoCapture(int(args.source) if str(args.source).isdigit() else str(args.source)))
    if not cap.isOpened():
        model.close()
        raise RuntimeError(f'Cannot open lane source: {args.source}')
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    simulation = LaneSimulation(args.nav, args.moving)
    fpga = cpu = worker = writer = None
    processed = displayed = decoded = 0
    latest = None
    tracker = LaneEvidenceTracker()
    next_mode = None
    requested_input = [args.front_channel] if args.pcie_capture else None
    output = Path(args.output)
    if output.with_suffix('.json').exists() or not args.no_record and output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = last_log = time.perf_counter()
    timings = []
    observations = []
    state_counts={side:{kind:0 for kind in ('UNKNOWN','DASHED','SOLID')} for side in ('LEFT','RIGHT')}
    decision_counts={}
    def analyze(frame, number, captured_at):
        result = model.infer(frame)
        result.update(frame=number, captured_at=captured_at)
        return result
    try:
        if args.fpga_pci:
            fpga, cpu = FPGAInteraction(args.fpga_pci), CPUMotionKey()
        if args.show:
            worker = LatestFrameInference(analyze)
            cv2.namedWindow('Lane assistance', cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
        print(f'Lane assistance: {model.backend} size={model.size}; A/W/D or F_KEY1=direction, '
              'C_KEY1=move/stop; T=traffic mode, Q=exit. FPS below are actual processing rates.', flush=True)
        while True:
            if requested_input is not None and requested_input[0] != args.front_channel:
                next_mode = requested_input[0]
                break
            captured_at = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                if args.loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    latest = None
                    tracker = LaneEvidenceTracker()
                    continue
                break
            decoded += 1
            mosaic=frame if args.pcie_capture else None
            if args.pcie_capture:
                x1,y1,x2,y2=PCIeCapture.channel_bounds(args.front_channel)
                h,w=frame.shape[:2]
                frame=frame[round(y1*h):round(y2*h),round(x1*w):round(x2*w)]
            frame = cv2.resize(frame, (640,360))
            if worker is None:
                result = analyze(frame, decoded, captured_at)
            else:
                result = worker.take()
                worker.submit((frame.copy(), decoded, captured_at))
            if result is not None:
                latest = result
                processed += 1
                timings.append(result['processing_ms'])
            boundaries = {}
            age = time.perf_counter()-latest['captured_at'] if latest else 0
            if latest is not None and (latest['frame'] == decoded or age <= min(args.max_result_age, .7)):
                # These are line hypotheses only. The tracker draws and verifies
                # current-frame paint, so global scene flow must not veto them.
                boundaries = latest['boundaries']
            if args.pcie_capture and cap.video_status is not None:
                source = 2 if args.front_channel in (2,3) else args.front_channel
                if not cap.video_status['valid_mask'] & (1 << source):
                    boundaries = {}
                    tracker = LaneEvidenceTracker()
            boundaries=tracker.update(frame,boundaries,latest['frame'] if latest else None,decoded,
                                      time.perf_counter())
            if fpga is not None:
                controls = fpga.poll()
                if controls is not None:
                    simulation.nav = controls['navigation']
                    simulation.manual_moving = controls['manual_moving']
            action = cpu.poll() if cpu is not None else None
            simulation.update_lane(boundaries, action)
            for side,item in boundaries.items():state_counts[side][item['kind']]+=1
            decision_counts[simulation.decision]=decision_counts.get(simulation.decision,0)+1
            valid = any(item['kind'] in ('DASHED','SOLID') for item in boundaries.values())
            echo = fpga.publish(simulation, valid) if fpga is not None else None
            canvas=lane_canvas(mosaic,frame,boundaries,simulation,args.front_channel,
                               cap.video_status if args.pcie_capture else None,
                               args.display_width,args.display_height)
            observations.append({'frame':decoded,'inference_frame':latest['frame'] if latest else None,
                                 'states':{k:v['kind'] for k,v in boundaries.items()},
                                 'raw_states':{k:v['kind'] for k,v in latest['boundaries'].items()} if latest else {},
                                 'model_scores':{k:v['score'] for k,v in latest['boundaries'].items()} if latest else {},
                                 'paint_coverage':{k:v.get('paint_coverage') for k,v in boundaries.items()},
                                 'decision':simulation.decision,'alarm':simulation.alarm,'echo':echo})
            if len(observations)>120:
                observations.pop(0)
            if not args.no_record:
                if writer is None:
                    writer = cv2.VideoWriter(str(output),cv2.VideoWriter_fourcc(*'mp4v'),fps,
                                             (canvas.shape[1],canvas.shape[0]))
                    if not writer.isOpened():
                        raise RuntimeError(f'Cannot create {output}')
                writer.write(canvas)
            now = time.perf_counter()
            if now-last_log >= 1:
                duration=now-started
                print(f'display={displayed/duration:.1f} FPS inference={processed/duration:.2f} FPS | '
                      f'frame={decoded} last={timings[-1] if timings else 0:.0f}ms age={age:.2f}s '
                      f'{observations[-1]["states"]} {simulation.decision} moving={int(simulation.moving)} '
                      f'alarm={int(simulation.alarm)} fpga_echo={hex(echo) if echo is not None else "none"}',flush=True)
                last_log=now
            if args.show:
                cv2.setMouseCallback('Lane assistance',handle_mouse,
                                     (simulation,canvas.shape[:2],canvas.shape[:2],requested_input))
                cv2.imshow('Lane assistance',canvas)
                if displayed==0:
                    cv2.moveWindow('Lane assistance',40,40)
                displayed += 1
                key=cv2.waitKey(max(1,min(33,round((started+decoded/fps-time.perf_counter())*1000)))) & 0xff
                if key in (27,ord('q')) or cv2.getWindowProperty('Lane assistance',cv2.WND_PROP_VISIBLE)<1:
                    break
                if requested_input is not None and key in (ord('1'),ord('2'),ord('3')):
                    requested_input[0]=key-ord('1')
                    continue
                if key==ord('t'):
                    next_mode='traffic'
                    break
                if key==ord('b'):
                    next_mode='fusion'
                    break
                if key in KEY_ACTIONS:
                    simulation.command(KEY_ACTIONS[key])
            if args.max_frames and processed>=args.max_frames or args.max_display_frames and displayed>=args.max_display_frames:
                break
    finally:
        if worker is not None:worker.close()
        if fpga is not None:fpga.close()
        if cpu is not None:cpu.close()
        cap.release()
        model.close()
        if writer is not None:writer.release()
        if args.show:cv2.destroyAllWindows()
    duration=time.perf_counter()-started
    report={'decoded_frames':decoded,'inferred_frames':processed,'displayed_frames':displayed,
            'display_fps':displayed/duration,'inference_fps':processed/duration,
            'mean_inference_ms':float(np.mean(timings)) if timings else None,
            'p95_inference_ms':float(np.percentile(timings,95)) if timings else None,
            'backend':model.backend,'input_size':model.size,'observations':observations,
            'state_counts':state_counts,'decision_counts':decision_counts,
            'note':'No annotated accuracy measurement. Unknown boundaries never authorize lane changes.'}
    output.with_suffix('.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('LANE_REPORT',json.dumps({k:v for k,v in report.items() if k!='observations'}),flush=True)
    return next_mode
