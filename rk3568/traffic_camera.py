# 姓名：丁黄炎
"""Joint-view fast matching, with camera refinement and the original HDMI fallback."""
import ctypes as ct
import time
from pathlib import Path
import numpy as np
import cv2
from traffic_pedestrian import PersonTracker


class CameraPersonTracker(PersonTracker):
    def __init__(self, library, camera_refinement=True):
        super().__init__(library)
        self.camera_refinement=camera_refinement
        self.camera = ct.CDLL(str(Path(__file__).resolve().parent/'native_camera/libcamera_threads.so'))
        self.camera.traffic_camera_move_people.argtypes = [ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_int,
            ct.c_void_p,ct.c_int,ct.c_int,ct.c_double,ct.c_int,ct.c_int,ct.c_void_p]
        self.camera.traffic_camera_move_people.restype = ct.c_int
        self.camera.traffic_camera_match_people.argtypes = [ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_int,
            ct.c_void_p,ct.c_int,ct.c_int,ct.c_double,ct.c_int,ct.c_int,ct.c_void_p]
        self.camera.traffic_camera_match_people.restype = ct.c_int
        self.camera.traffic_camera_dense_flow.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_int,ct.c_void_p]
        self.camera.traffic_camera_dense_flow.restype=ct.c_int

    def move_native(self, previous, current, flow, people, fraction, shape):
        if not self.camera_refinement:
            return super().move_native(previous,current,flow,people,fraction,shape)
        if not people:return []
        boxes=np.ascontiguousarray(people,dtype=np.float64)
        output=np.empty_like(boxes)
        count=self.camera.traffic_camera_move_people(previous.ctypes.data,current.ctypes.data,flow.ctypes.data,
            current.shape[1],current.shape[0],boxes.ctypes.data,len(boxes),boxes.shape[1],fraction,
            shape[1],shape[0],output.ctypes.data)
        if count<0:raise RuntimeError('Camera person motion refinement failed')
        return output[:count].tolist()

    def match_native(self, previous, current, people, fraction, shape):
        if np.array_equal(previous,current):return people
        boxes=np.ascontiguousarray(people,dtype=np.float64)
        output=np.empty_like(boxes)
        count=self.camera.traffic_camera_match_people(previous.ctypes.data,current.ctypes.data,
            current.shape[1],current.shape[0],boxes.ctypes.data,len(boxes),boxes.shape[1],fraction,
            shape[1],shape[0],output.ctypes.data)
        if count==-2:return None
        if count<0:raise RuntimeError('Camera person matching failed')
        return output[:count].tolist()

    def _advance(self, observation, chain, start, shape):
        for item in chain:
            source,target,previous,current,flow=item
            if previous is None:continue
            people=observation['people']
            if not people:break
            fraction=min(1,(target-start)/max(1,target-source))
            aligned=self.match_native(previous,current,people,fraction,shape)
            if aligned is None:
                if flow is None:
                    if self.native is not None:
                        flow=np.empty((*current.shape,2),np.float32)
                        status=self.camera.traffic_camera_dense_flow(previous.ctypes.data,current.ctypes.data,
                            current.shape[1],current.shape[0],flow.ctypes.data)
                        if status!=1:raise RuntimeError('Camera person optical flow failed')
                    else:
                        flow=self.flow.calc(previous,current,None)
                    item[4]=flow
                aligned=self.move_native(previous,current,flow,people,fraction,shape)
            observation['tracking_lost']|=len(aligned)!=len(people)
            observation['people']=aligned
            start=target
        return observation

    def align_detection(self, result):
        observation=dict(result,source_frame=result.get('source_frame',result['frame']),
                         people=[list(person) for person in result['people']],
                         tracking_lost=result.get('tracking_lost',False))
        if observation['source_frame']<self.cut_frame:return observation
        # Worker-owned entries keep lazily computed flows out of the live history.
        chain=[list(item) for item in list(self.frames) if item[1]>result['frame']]
        if chain:
            self._advance(observation,chain,result['frame'],result['shape'])
            observation['frame']=chain[-1][1]
        return observation

    def update(self, frame, result, number):
        if self.native is not None:
            gray=np.empty((frame.shape[0]//2,frame.shape[1]//2),np.uint8)
            status=self.native.traffic_gray_flow(None,frame.ctypes.data,frame.shape[1],frame.shape[0],
                frame.strides[0],gray.ctypes.data,None)
            if status<0:raise RuntimeError('Camera grayscale conversion failed')
        else:
            gray=cv2.resize(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),(frame.shape[1]//2,frame.shape[0]//2))
        self.frames.append([self.number,number,self.gray,gray,None])
        if result is not None and result.get('source_frame',result['frame'])>=self.cut_frame:
            self.observation=dict(result,tracking_lost=result.get('tracking_lost',False))
            chain=[item for item in self.frames if item[1]>result['frame']]
            start=result['frame']
        else:
            chain,start=[self.frames[-1]],self.number
        if self.observation is not None:
            self._advance(self.observation,chain,start,frame.shape)
        self.gray,self.number=gray,number
        return self.observation


class NativeDisplay:
    """Optional one-window native X11 preview with this application's HighGUI calls."""
    def __init__(self, library=None):
        self.library=ct.CDLL(str(library or Path(__file__).resolve().parent/'native_camera/libcamera_threads.so'))
        for name,arguments,result in (
            ('traffic_preview_open',[ct.c_char_p,ct.c_int,ct.c_int],ct.c_void_p),
            ('traffic_preview_show',[ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_int,ct.c_int,ct.c_void_p],ct.c_int),
            ('traffic_preview_poll',[ct.c_void_p,ct.c_void_p],ct.c_int),
            ('traffic_preview_move',[ct.c_void_p,ct.c_int,ct.c_int],ct.c_int),
            ('traffic_preview_close',[ct.c_void_p],None)):
            function=getattr(self.library,name);function.argtypes=arguments;function.restype=result
        self.library.traffic_camera_error.restype=ct.c_char_p
        self.handle=self.name=self.callback=self.parameter=self.position=None
        self.visible=True
        self.keys=[]

    def _check(self, valid):
        if not valid:raise RuntimeError(self.library.traffic_camera_error().decode(errors='replace'))

    def namedWindow(self, name, flags=0):
        if self.name is not None and self.name!=name:raise ValueError('Native preview supports one window')
        self.name=name

    def setMouseCallback(self, name, callback, parameter=None):
        if name!=self.name:raise ValueError('Unknown preview window')
        self.callback,self.parameter=callback,parameter

    def imshow(self, name, image, readback=None):
        if name!=self.name:raise ValueError('Unknown preview window')
        if image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3:
            raise ValueError('Native preview requires an 8-bit BGR image')
        image=np.ascontiguousarray(image)
        if readback is not None and (readback.shape!=image.shape or readback.dtype!=np.uint8 or not readback.flags.c_contiguous):
            raise ValueError('Preview readback must be a contiguous BGR image of the same size')
        if self.handle is None:
            self.handle=self.library.traffic_preview_open(name.encode('utf-8'),image.shape[1],image.shape[0])
            self._check(self.handle)
            if self.position is not None:self.moveWindow(name,*self.position)
        self._check(self.library.traffic_preview_show(self.handle,image.ctypes.data,image.shape[1],image.shape[0],
                    image.strides[0],None if readback is None else readback.ctypes.data)==1)

    def moveWindow(self, name, x, y):
        if name!=self.name:raise ValueError('Unknown preview window')
        self.position=(int(x),int(y))
        if self.handle is not None:self._check(self.library.traffic_preview_move(self.handle,*self.position)==1)

    def waitKey(self, delay=1):
        if self.handle is None:return -1
        fields=(ct.c_int*6)()
        for _ in range(256):
            status=self.library.traffic_preview_poll(self.handle,fields)
            self._check(status>=0)
            self.visible=bool(fields[5])
            if not status:break
            if fields[0]>=0 and self.callback is not None:
                self.callback(fields[0],fields[1],fields[2],fields[3],self.parameter)
            # A held key must not leave stale steering commands queued after release.
            if fields[4]>=0:self.keys[:]=[fields[4]]
        # Preserve an intentional frame-rate limit without HighGUI's minimum GUI wait.
        if delay>1 and not self.keys and self.visible:time.sleep(delay/1000)
        return self.keys.pop(0) if self.keys else -1

    def getWindowProperty(self, name, property_id):
        if name!=self.name or property_id!=cv2.WND_PROP_VISIBLE:raise ValueError('Unknown preview window property')
        return float(self.visible)

    def destroyAllWindows(self):
        if self.handle is not None:self.library.traffic_preview_close(self.handle)
        self.handle=None
        self.visible=False
