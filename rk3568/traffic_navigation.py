"""Read a selected on-screen guidance arrow without another NPU model."""
import time
import cv2
import numpy as np

_camera_readers={}


def _read_camera_navigation(frame,roi=None):
    roi=tuple(roi) if roi is not None else None
    if roi not in _camera_readers:_camera_readers[roi]=NavigationReader(roi)
    started=time.perf_counter()
    direction,score=_camera_readers[roi].classify(frame)
    return direction,score,(time.perf_counter()-started)*1000


def arrow_mask(direction, size=96):
    if direction == 'STRAIGHT':
        points=[(48,4),(19,34),(35,34),(35,91),(61,91),(61,34),(77,34)]
    else:
        points=[(4,27),(32,4),(32,16),(58,16),(71,20),(79,31),(79,91),
                (55,91),(55,43),(51,40),(32,40),(32,52)]
        if direction == 'RIGHT':points=[(95-x,y) for x,y in points]
        elif direction != 'LEFT':raise ValueError('Unsupported navigation arrow')
    mask=np.zeros((96,96),np.uint8)
    cv2.fillPoly(mask,[np.array(points,np.int32)],255)
    return cv2.resize(mask,(size,size),interpolation=cv2.INTER_NEAREST)


def normalized_component(mask):
    points=cv2.findNonZero(mask)
    if points is None:return None
    x,y,w,h=cv2.boundingRect(points)
    scale=56/max(w,h)
    w2,h2=max(1,round(w*scale)),max(1,round(h*scale))
    small=cv2.resize(mask[y:y+h,x:x+w],(w2,h2),interpolation=cv2.INTER_AREA)>127
    result=np.zeros((64,64),bool)
    left,top=(64-w2)//2,(64-h2)//2
    result[top:top+h2,left:left+w2]=small
    return result


class NavigationReader:
    """Only confirmed, current arrow pixels can supply a route instruction."""
    def __init__(self, roi, interval=.2, executor=None):
        if roi is not None and (len(roi)!=4 or not all(np.isfinite(x) for x in roi) or not (
                0<=roi[0]<roi[2]<=1 and 0<=roi[1]<roi[3]<=1)):
            raise ValueError('Navigation ROI must be x1 y1 x2 y2 within the front image')
        self.roi=tuple(roi) if roi is not None else None
        self.interval=interval
        self.next_read=0.
        self.previous=None
        self.result={'direction':'UNKNOWN','score':0.,'captured_at':0.}
        self.count=0
        self.total_ms=0.
        self.executor=executor
        self.pending=None
        self.pending_captured_at=None
        self.templates={key:normalized_component(arrow_mask(key)) for key in ('LEFT','STRAIGHT','RIGHT')}

    def classify(self, image):
        if self.roi is None:
            scale=min(1.,320/image.shape[1])
            search=cv2.resize(image,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA) if scale<1 else image
            hsv=cv2.cvtColor(search,cv2.COLOR_BGR2HSV)
            matches=[]
            for low,high in ((35,79),(80,135)):
                mask=cv2.inRange(hsv,(low,85,30),(high,255,255))
                mask[round(mask.shape[0]*.65):]=0
                mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
                contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                for contour in sorted(contours,key=cv2.contourArea,reverse=True)[:6]:
                    x,y,w,h=cv2.boundingRect(contour)
                    if min(w,h)<16*scale or not .001< w*h/mask.size < .18 or not .4<w/h<2.8:
                        continue
                    if cv2.contourArea(contour)<w*h*.65:
                        continue
                    x1,y1=round(x/scale),round(y/scale)
                    x2,y2=round((x+w)/scale),round((y+h)/scale)
                    direction,score=self.classify_crop(image[y1:y2,x1:x2],camera=True)
                    if direction!='UNKNOWN':matches.append((score,direction))
            if not matches or len({key for _,key in matches})!=1:return 'UNKNOWN',0.
            score,direction=max(matches)
            return direction,score
        h,w=image.shape[:2]
        x1,y1,x2,y2=self.roi
        crop=image[round(y1*h):round(y2*h),round(x1*w):round(x2*w)]
        return self.classify_crop(crop)

    def classify_crop(self, crop,camera=False):
        if min(crop.shape[:2])<8:return 'UNKNOWN',0.
        scale=min(1.,160/max(crop.shape[:2]))
        if scale<1:
            crop=cv2.resize(crop,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
        # White guidance icons on a coloured/dark navigation card. Reject road texture/text.
        low=crop.min(axis=2);high=crop.max(axis=2)
        floor=max(110,min(170,float(np.quantile(high,.98))*.6)) if camera else 170
        mask=((low>=floor)&((high.astype(np.int16)-low)<65)).astype(np.uint8)
        count,labels,stats,_=cv2.connectedComponentsWithStats(mask,8)
        matches=[]
        for index in range(1,count):
            x,y,cw,ch,area=stats[index]
            if area<mask.size*.025 or ch<mask.shape[0]*.24 or not .27<cw/max(ch,1)<2:
                continue
            component=normalized_component((labels[y:y+ch,x:x+cw]==index).astype(np.uint8)*255)
            scores=sorted(((np.count_nonzero(component&template)/max(1,np.count_nonzero(component|template)),key)
                           for key,template in self.templates.items()),reverse=True)
            if scores[0][0]>=.72 and scores[0][0]-scores[1][0]>=.12:
                matches.append(scores[0])
        if not matches or len({key for score,key in matches})!=1:return 'UNKNOWN',0.
        score,direction=max(matches)
        return direction,float(score)

    def update(self, frame, captured_at, now=None):
        now=time.perf_counter() if now is None else now
        if now-captured_at>.35:
            self.previous=None
            self.result={'direction':'UNKNOWN','score':0.,'captured_at':captured_at}
            return self.result
        observation=None
        if self.pending is not None and self.pending.done():
            direction,score,elapsed=self.pending.result()
            observation=(direction,score,self.pending_captured_at,elapsed)
            self.pending=None
        if now>=self.next_read and self.pending is None:
            if self.executor is not None:
                self.pending=self.executor.submit(_read_camera_navigation,frame.copy(),self.roi)
                self.pending_captured_at=captured_at
            else:
                tick=time.perf_counter()
                direction,score=self.classify(frame)
                observation=(direction,score,captured_at,(time.perf_counter()-tick)*1000)
            self.next_read=now+self.interval
        if observation is not None:
            direction,score,observed_at,elapsed=observation
            if not 0<=now-observed_at<=.35:direction,score='UNKNOWN',0.
            confirmed=direction if direction==self.previous else 'UNKNOWN'
            self.previous=direction if direction!='UNKNOWN' else None
            self.result={'direction':confirmed,'score':score,'captured_at':observed_at}
            self.count+=1
            self.total_ms+=elapsed
        if now-self.result['captured_at']>.45:
            return {'direction':'UNKNOWN','score':0.,'captured_at':self.result['captured_at']}
        return self.result
