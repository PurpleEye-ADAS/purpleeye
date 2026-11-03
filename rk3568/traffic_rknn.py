"""RKNN C API inference with the same RGB preprocessing as the PC models."""

import ctypes as ct
import json
import mmap
import os
import struct
import time
from itertools import combinations
from pathlib import Path
from threading import RLock
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

_NPU_LOCK = RLock()


import cv2
import numpy as np


def source_scene_changed(previous, current, upper_only=True):
    """Reject a large pixel change only after supported motion fails to explain it."""
    if previous is None or previous.shape!=current.shape:
        return False
    height=max(1,current.shape[0]//2) if upper_only else current.shape[0]
    source,target=previous[:height],current[:height]
    if cv2.norm(source,target,cv2.NORM_L1)/target.size<=35:
        return False
    scale=min(1.,320/source.shape[1])
    small_source,small_target=(cv2.resize(image,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
                               if scale<1 else image for image in (source,target))
    orb=cv2.ORB_create(nfeatures=800,nlevels=1,edgeThreshold=15,fastThreshold=10)
    source_points,source_descriptors=orb.detectAndCompute(small_source,None)
    target_points,target_descriptors=orb.detectAndCompute(small_target,None)
    if source_descriptors is None or target_descriptors is None:
        return True
    matches=cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(source_descriptors,target_descriptors,k=2)
    matches=[pair[0] for pair in matches if len(pair)==2 and pair[0].distance<.75*pair[1].distance]
    if len(matches)<15:
        return True
    points=np.float32([source_points[match.queryIdx].pt for match in matches]).reshape(-1,1,2)
    moved=np.float32([target_points[match.trainIdx].pt for match in matches]).reshape(-1,1,2)
    matrix,inliers=cv2.estimateAffinePartial2D(points,moved,method=cv2.RANSAC,ransacReprojThreshold=3*scale)
    if matrix is None or inliers is None or not np.isfinite(matrix).all():
        return True
    supported=inliers.ravel().astype(bool)
    if supported.sum()<max(15,len(points)*.15):
        return True
    if min(cv2.contourArea(cv2.convexHull(values[supported])) for values in (points,moved))<small_source.size*.1:
        return True
    if not .8<=np.hypot(matrix[0,0],matrix[0,1])<=1.25:
        return True
    matrix[:,2]/=scale
    size=(source.shape[1],source.shape[0])
    # Keep the original pixel threshold and exclude newly exposed warp borders.
    mask=cv2.warpAffine(np.full(source.shape,255,np.uint8),matrix,size)
    mask[mask<255]=0
    overlap=cv2.countNonZero(mask)
    if overlap<source.size*.6:
        return True
    aligned=cv2.warpAffine(source,matrix,size)
    return cv2.norm(aligned,target,cv2.NORM_L1,mask=mask)/overlap>35


class NativeSignals:
    def __init__(self, library, camera=False):
        self.library=library
        self.camera_library=None
        if camera:
            self.camera_library=ct.CDLL(str(Path(__file__).resolve().parent/'native_camera/libcamera_threads.so'))
            self.camera_library.traffic_camera_configure_threads.argtypes=[]
            self.camera_library.traffic_camera_configure_threads.restype=ct.c_int
            if self.camera_library.traffic_camera_configure_threads()!=1:
                raise RuntimeError('Camera native thread configuration failed')
            self.camera_library.traffic_camera_color.argtypes=[ct.c_void_p,ct.c_int,ct.c_int,ct.c_int,
                                                             ct.POINTER(ct.c_double)]
            self.camera_library.traffic_camera_color.restype=ct.c_int
            self.camera_library.traffic_camera_match_boxes.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_int,
                ct.c_void_p,ct.c_int,ct.c_int,ct.c_double,ct.c_int,ct.c_int,ct.c_void_p,ct.c_double]
            self.camera_library.traffic_camera_match_boxes.restype=ct.c_int
        library.traffic_sparse_flow.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_int,ct.c_void_p,
            ct.c_int,ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_void_p]
        library.traffic_sparse_flow.restype=ct.c_int
        library.traffic_light_color.argtypes=[ct.c_void_p,ct.c_int,ct.c_int,ct.c_int,ct.POINTER(ct.c_double)]
        library.traffic_light_color.restype=ct.c_int
        library.traffic_alignment_open.argtypes=[];library.traffic_alignment_open.restype=ct.c_void_p
        library.traffic_alignment_close.argtypes=[ct.c_void_p];library.traffic_alignment_close.restype=None
        library.traffic_align_boxes.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_int,
            ct.c_int,ct.c_int,ct.c_uint64,ct.c_uint64,ct.c_void_p,ct.c_int,ct.c_int,ct.c_int,
            ct.c_void_p,ct.c_void_p,ct.c_void_p]
        library.traffic_align_boxes.restype=ct.c_int

    def align(self,source,current,lights,shape,cache):
        cache.pop('motion',None)
        boxes=np.ascontiguousarray([light['box'] for light in lights],np.float64)
        output=np.empty_like(boxes);kept=np.empty(len(lights),np.int32);motion=np.empty((3,3),np.float64)
        size=(current.shape[1]//2,current.shape[0]//2)
        half=(self.camera_library is not None and len(lights)>0 and
              ((boxes[:,2:]-boxes[:,:2])*np.array(size)/[shape[1],shape[0]]>=12).all())
        if (self.camera_library is not None and len(lights)>0 and source.shape==current.shape
                and source.ndim==2 and min(source.shape)>0 and shape[0]>0 and shape[1]>0
                and np.isfinite(boxes).all() and (boxes[:,:2]>=0).all()
                and (boxes[:,2:]<=[shape[1],shape[0]]).all() and (boxes[:,2:]>boxes[:,:2]).all()):
            dimensions=np.array([current.shape[1],current.shape[0]])
            start,end=dimensions.copy(),np.zeros(2,int)
            scales=(np.ones(2),np.array(size)/dimensions) if half else (np.ones(2),)
            for scale in scales:
                scaled=boxes.reshape(-1,2,2)*dimensions/[shape[1],shape[0]]*scale
                padding=max(24,int(np.max(scaled[:,1]-scaled[:,0])*2))
                start=np.minimum(start,np.floor((np.floor(scaled[:,0].min(0))-padding)/scale).astype(int)-2)
                end=np.maximum(end,np.ceil((np.ceil(scaled[:,1].max(0))+padding)/scale).astype(int)+2)
            previous=cache.get('_native_alignment_region')
            # Native tracking can retain its original ROI while boxes move; cover that ROI too.
            if previous is not None and previous[0]==current.shape:
                start=np.minimum(start,previous[1]);end=np.maximum(end,previous[2])
            start=np.maximum(start,0);end=np.minimum(end,dimensions)
            cache['_native_alignment_region']=(current.shape,start,end)
            x1,y1=start;x2,y2=end
            if x2>x1 and y2>y1 and np.array_equal(source[y1:y2,x1:x2],current[y1:y2,x1:x2]):
                cache['motion']=np.eye(3)
                return [{**light,'box':list(light['box'])} for light in lights]
            # A unique near-exact translation can reuse the existing native box matcher.
            # Any ambiguity, deformation or disagreement retains the original LK path.
            factor=np.tile(2*dimensions/[shape[1],shape[0]],2)
            scaled=np.ascontiguousarray(boxes*factor)
            matched=np.empty_like(scaled)
            count=self.camera_library.traffic_camera_match_boxes(source.ctypes.data,current.ctypes.data,
                current.shape[1],current.shape[0],scaled.ctypes.data,len(lights),4,1.,
                int(dimensions[0]*2),int(dimensions[1]*2),matched.ctypes.data,.98)
            if count==-1:raise RuntimeError('Native signal template matching failed')
            if count==len(lights):
                moved=matched/factor
                delta=(moved-boxes).reshape(-1,2)
                if np.max(abs(delta-delta[0]))<=.25:
                    motion=np.eye(3);motion[:2,2]=delta[0]
                    cache['motion']=motion
                    return [{**light,'box':box} for light,box in zip(lights,moved.tolist())]
        originals=(source,current)
        for reduced in ((True,False) if half else (False,)):
            key='_native_alignment_half' if reduced else '_native_alignment'
            if reduced:
                previous=cache.get('_native_alignment_half_frames',())
                frames=[]
                for image in originals:
                    pair=next((pair for pair in previous if pair[0] is image),None)
                    frames.append(pair if pair is not None else
                                  (image,cv2.resize(image,size,interpolation=cv2.INTER_AREA)))
                cache['_native_alignment_half_frames']=tuple(frames)
                source,current=(pair[1] for pair in frames)
            else:
                source,current=originals
                cache['_native_alignment_frames']=originals
            state=cache.get(key)
            if state is None:
                state=_NativeSignalAlignment(self.library)
                cache[key]=state
            count=self.library.traffic_align_boxes(state.handle,source.ctypes.data,current.ctypes.data,
                current.shape[1],current.shape[0],source.strides[0],current.strides[0],id(source),id(current),
                boxes.ctypes.data,len(lights),shape[1],shape[0],output.ctypes.data,kept.ctypes.data,motion.ctypes.data)
            if reduced and count!=len(lights):continue
            if count<0:raise RuntimeError('Native signal alignment failed')
            if count:cache['motion']=motion
            return [{**lights[index],'box':box} for index,box in zip(kept[:count],output[:count].tolist())]

    def flow(self, previous, current, points):
        moved=np.empty_like(points);returned=np.empty_like(points)
        forward=np.empty((len(points),1),np.uint8);backward=np.empty_like(forward)
        status=self.library.traffic_sparse_flow(previous.ctypes.data,current.ctypes.data,
            previous.shape[1],previous.shape[0],points.ctypes.data,len(points),moved.ctypes.data,
            returned.ctypes.data,forward.ctypes.data,backward.ctypes.data)
        if status!=1:raise RuntimeError('Native signal optical flow failed')
        return moved,returned,forward,backward

    def observe(self, crop, camera=False):
        confidence=ct.c_double()
        function=self.camera_library.traffic_camera_color if camera else self.library.traffic_light_color
        status=function(crop.ctypes.data,crop.shape[1],crop.shape[0],crop.strides[0],ct.byref(confidence))
        if status==-2:raise RuntimeError('Native signal color failed')
        return (('red','yellow','green')[status],confidence.value) if status>=0 else ('UNKNOWN',0.)

class _NativeSignalAlignment:
    def __init__(self,library):
        self.library=library;self.handle=library.traffic_alignment_open()
        if not self.handle:raise RuntimeError('Cannot allocate signal alignment')

    def __del__(self):
        if self.handle:
            self.library.traffic_alignment_close(self.handle);self.handle=None


def _camera_screen_pixels(gray,quads):
    edges=np.roll(quads,-1,axis=1)-quads
    normal=np.stack((-edges[:,:,1],edges[:,:,0]),axis=2)/np.maximum(np.linalg.norm(edges,axis=2,keepdims=True),1)
    points=quads[:,:,None,:]+edges[:,:,None,:]*np.linspace(.15,.85,24)[None,None,:,None]
    inside=np.rint(points+normal[:,:,None,:]*4).astype(int);outside=np.rint(points-normal[:,:,None,:]*4).astype(int)
    valid=((inside>=0).all((1,2,3))&(outside>=0).all((1,2,3))&
           (inside<[gray.shape[1],gray.shape[0]]).all((1,2,3))&(outside<[gray.shape[1],gray.shape[0]]).all((1,2,3)))
    inside=np.clip(inside,[0,0],[gray.shape[1]-1,gray.shape[0]-1])
    outside=np.clip(outside,[0,0],[gray.shape[1]-1,gray.shape[0]-1])
    return gray[inside[:,:,:,1],inside[:,:,:,0]].astype(float),gray[outside[:,:,:,1],outside[:,:,:,0]].astype(float),valid


def _camera_screen_quality(inner,outer,valid):
    contrast=np.median(inner-outer,axis=2)
    sides=(np.median(np.abs(np.diff(outer,axis=2)),axis=2)<=12)&(contrast>=12)
    fractions=np.mean(inner-outer>=6,axis=2)
    accepted=valid&(sides.sum(1)>=3)&sides[:,0]&sides[:,2]&(fractions[:,0]>=.6)&(fractions[:,2]>=.6)
    texture=(np.ptp(np.quantile(inner[:,2],[.1,.9],axis=1),axis=0),
             np.ptp(np.quantile(outer[:,2],[.1,.9],axis=1),axis=0))
    return accepted,np.minimum(contrast[:,0],contrast[:,2]),texture


def _camera_screen_border(gray,quad,reference=None):
    """Validate the current physical boundary; its exterior appearance survives video changes."""
    inner,outer,valid=_camera_screen_pixels(gray,np.asarray(quad)[None])
    accepted,contrast,texture=_camera_screen_quality(inner,outer,valid)
    if reference is not None:
        stable=np.median(np.abs(outer[0]-reference),axis=1)<=20
        accepted[0]=valid[0] and stable.sum()>=3 and (np.median(abs(inner[0]-outer[0]),axis=1)>=6).sum()>=2
    return bool(accepted[0]),outer[0].copy(),float(contrast[0]),(texture[0][0],texture[1][0])


def _camera_screen_shape(gray,quad):
    if not cv2.isContourConvex(quad) or (quad<3).any() or (quad>=[gray.shape[1]-3,gray.shape[0]-3]).any():return False
    lengths=np.linalg.norm(quad-np.roll(quad,-1,axis=0),axis=1)
    return (.15*gray.size<cv2.contourArea(quad)<.9*gray.size and lengths.min()>=40 and
            1.35<(lengths[0]+lengths[2])/(lengths[1]+lengths[3])<3.2)


def _camera_screen_lines(gray):
    """Join visible panel edges when reflection prevents a closed contour."""
    height,width=gray.shape
    edges=cv2.Canny(cv2.GaussianBlur(gray,(3,3),0),8,25)
    lines=cv2.HoughLinesP(edges,1,np.pi/180,threshold=max(10,width//24),
        minLineLength=max(10,height//12),maxLineGap=max(8,width//24))
    if lines is None:return None
    groups=[[],[]]
    for x1,y1,x2,y2 in lines.reshape(-1,4):
        dx,dy=x2-x1,y2-y1
        kind=(0 if abs(dy)<=abs(dx)*.3 and abs(dx)>width*.1 else
              1 if abs(dx)<=abs(dy)*.45 and abs(dy)>height*.15 else None)
        if kind is None:continue
        quality=1.
        if kind==0:
            points=np.array([x1,y1])+np.array([dx,dy])*np.linspace(.15,.85,24)[:,None]
            above=np.rint(points-[0,4]).astype(int);below=np.rint(points+[0,4]).astype(int)
            good=((above>=0).all(1)&(below>=0).all(1)&(above<[width,height]).all(1)&(below<[width,height]).all(1))
            if good.sum()<18:continue
            a=gray[above[good,1],above[good,0]].astype(float);b=gray[below[good,1],below[good,0]].astype(float)
            quality=max((float(np.median(inside-outside)) for outside,inside in ((a,b),(b,a))
                         if np.median(np.abs(np.diff(outside)))<=10),default=0.)
            if quality<12:continue
        line=np.cross([x1,y1,1.],[x2,y2,1.]);line/=np.linalg.norm(line[:2])
        intercept=-(line[0]*width/2+line[2])/line[1] if kind==0 else -(line[1]*height/2+line[2])/line[0]
        groups[kind].append((np.hypot(dx,dy)*quality,intercept,line,quality))
    selected=[]
    for index,group in enumerate(groups):
        kept=[]
        for item in sorted(group,key=lambda x:x[0],reverse=True):
            if not any(abs(item[1]-old[1])<5 and abs(np.dot(item[2][:2],old[2][:2]))>.99 for old in kept):kept.append(item)
            if len(kept)==(18 if index==0 else 10):break
        selected.append(sorted(kept,key=lambda x:x[1]))
    horizontal,vertical=selected
    if len(horizontal)<2 or len(vertical)<2:return None
    cross=np.cross(np.array([row[2] for row in horizontal])[:,None],np.array([row[2] for row in vertical])[None])
    points=np.divide(cross[:,:,:2],cross[:,:,2:],out=np.full_like(cross[:,:,:2],1e8),where=abs(cross[:,:,2:])>1e-6)
    hp=np.array([(t,b) for t,b in combinations(range(len(horizontal)),2)
                 if height*.2<horizontal[b][1]-horizontal[t][1]<height*.85],int)
    vp=np.array([(l,r) for l,r in combinations(range(len(vertical)),2)
                 if width*.35<vertical[r][1]-vertical[l][1]<width*.95],int)
    if not hp.size or not vp.size:return None
    hs=np.repeat(hp,len(vp),axis=0);vs=np.tile(vp,(len(hp),1));t,b=hs.T;l,r=vs.T
    quads=np.stack((points[t,l],points[t,r],points[b,r],points[b,l]),axis=1).astype(np.float32)
    delta=np.roll(quads,-1,axis=1)-quads;following=np.roll(delta,-1,axis=1)
    convex=(delta[:,:,0]*following[:,:,1]-delta[:,:,1]*following[:,:,0]>0).all(1)
    lengths=np.linalg.norm(delta,axis=2);ratio=(lengths[:,0]+lengths[:,2])/np.maximum(lengths[:,1]+lengths[:,3],1)
    area=np.abs((quads[:,:,0]*np.roll(quads[:,:,1],-1,axis=1)-quads[:,:,1]*np.roll(quads[:,:,0],-1,axis=1)).sum(1))*.5
    good=(convex&(quads>=3).all((1,2))&(quads<[width-3,height-3]).all((1,2))&
          (lengths.min(1)>=40)&(ratio>1.35)&(ratio<3.2)&(area>gray.size*.15)&(area<gray.size*.9))
    if not good.any():return None
    quads,area,l,r=quads[good],area[good],l[good],r[good]
    inner,outer,inside=_camera_screen_pixels(gray,quads)
    valid,contrast,texture=_camera_screen_quality(inner,outer,inside)
    valid&=texture[0]>=texture[1]+4
    if not valid.any():return None
    vl=np.array([row[2] for row in vertical]);vanishing=np.cross(vl[l],vl[r])
    vx=np.divide(vanishing[:,0],vanishing[:,2],out=np.full(len(l),width/2.),where=abs(vanishing[:,2])>1e-6)
    scores=np.where(valid,contrast*np.sqrt(area)/(1+3*abs(vx-width/2)/width),-np.inf)
    return quads[int(np.argmax(scores))]


def camera_screen_quad(image):
    """Find a complete widescreen panel; otherwise preserve the original camera view."""
    height, width = image.shape[:2]
    scale = min(1., 640 / width)
    small = cv2.resize(image, (round(width*scale), round(height*scale))) if scale < 1 else image
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 30, 80)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:8]:
        hull = cv2.convexHull(contour)
        polygon = cv2.approxPolyDP(hull, .025*cv2.arcLength(hull, True), True)
        area = cv2.contourArea(polygon)
        if len(polygon) != 4 or not .15*gray.size < area < .9*gray.size:
            continue
        points = polygon.reshape(4, 2).astype(np.float32)
        sums, differences = points.sum(axis=1), np.diff(points, axis=1).ravel()
        quad = points[[sums.argmin(), differences.argmin(), sums.argmax(), differences.argmax()]]
        if len(np.unique(quad, axis=0)) != 4:
            continue
        if not _camera_screen_shape(gray,quad):
            continue
        if not _camera_screen_border(gray,quad)[0]:
            continue
        return quad / [gray.shape[1], gray.shape[0]]
    quad=_camera_screen_lines(gray)
    return quad/[gray.shape[1],gray.shape[0]] if quad is not None else None


def _camera_screen_worker(affinity):
    cv2.setNumThreads(1)
    if affinity is not None:
        os.sched_setaffinity(0,affinity&{0,3} or affinity)


class NativeCameraWarp:
    def __init__(self,library):
        self.library=ct.CDLL(str(library));self.handle=None
        self.library.traffic_camera_warp_open.restype=ct.c_void_p
        self.library.traffic_camera_warp_close.argtypes=[ct.c_void_p]
        self.library.traffic_camera_error.restype=ct.c_char_p
        self.library.traffic_camera_warp.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_int,ct.c_int,ct.c_void_p,ct.c_void_p]

    def apply(self,image,inverse):
        if self.handle is None:
            self.handle=self.library.traffic_camera_warp_open()
            if not self.handle:raise RuntimeError(self.library.traffic_camera_error().decode())
        output=np.empty(image.shape,np.uint8)
        status=self.library.traffic_camera_warp(self.handle,image.ctypes.data,image.shape[1],image.shape[0],
            image.strides[0],inverse.ctypes.data,output.ctypes.data)
        if status!=1:raise RuntimeError(self.library.traffic_camera_error().decode())
        return output

    def close(self):
        if self.handle is not None:
            self.library.traffic_camera_warp_close(self.handle);self.handle=None


class CameraScreenView:
    def __init__(self, executor=None,warp=None):
        self.quad = self.candidate = None
        self.bezel = None
        self.next_update = 0.
        self.revision = 0
        self.remaps = {}
        self.executor=executor
        self.pending=None
        self.pending_gray=None
        self.warp=warp

    def update(self, image, now):
        if now < self.next_update:
            return
        self.next_update = now + 1
        height,width=image.shape[:2];scale=min(1.,640/width)
        small=cv2.resize(image,(round(width*scale),round(height*scale))) if scale<1 else image
        gray=cv2.cvtColor(small,cv2.COLOR_BGR2GRAY)
        if self.quad is not None:
            if _camera_screen_border(gray,self.quad*[gray.shape[1],gray.shape[0]],self.bezel)[0]:
                return
            self.quad=self.bezel=None
            self.remaps.clear()
            self.revision+=1
        if self.executor is not None:
            if self.pending is None:
                self.pending=self.executor.submit(camera_screen_quad,small.copy())
                self.pending_gray=gray.copy()
                return
            if not self.pending.done():return
            candidate=self.pending.result()
            self.pending=None
            if candidate is not None:
                reference=_camera_screen_border(self.pending_gray,candidate*[gray.shape[1],gray.shape[0]])[1]
                if not _camera_screen_border(gray,candidate*[gray.shape[1],gray.shape[0]],reference)[0]:candidate=None
            self.pending_gray=None
        else:
            candidate = camera_screen_quad(image)
        was_active = self.quad is not None
        if candidate is not None and (self.executor is not None or
                self.candidate is not None and np.max(np.abs(candidate-self.candidate)) < .025):
            self.quad = candidate
        else:
            self.quad = None
        self.candidate = candidate
        self.bezel=(_camera_screen_border(gray,self.quad*[gray.shape[1],gray.shape[0]])[1]
                    if self.quad is not None else None)
        self.remaps.clear()
        if was_active != (self.quad is not None):
            self.revision += 1

    def apply(self, image):
        if self.quad is None:
            return image
        height, width = image.shape[:2]
        if self.warp is not None:
            inverse=cv2.getPerspectiveTransform(
                np.array([[0,0],[width-1,0],[width-1,height-1],[0,height-1]],np.float32),
                (self.quad*[width,height]).astype(np.float32))
            return self.warp.apply(image,inverse)
        key=(width,height,self.quad.tobytes())
        if key not in self.remaps:
            matrix=cv2.getPerspectiveTransform(
                np.array([[0,0],[width-1,0],[width-1,height-1],[0,height-1]],np.float32),
                (self.quad*[width,height]).astype(np.float32))
            grid=np.stack(np.meshgrid(np.arange(width,dtype=np.float32),np.arange(height,dtype=np.float32)),axis=2)
            source=cv2.perspectiveTransform(grid.reshape(-1,1,2),matrix).reshape(height,width,2)
            self.remaps[key]=cv2.convertMaps(source,None,cv2.CV_16SC2)
        return cv2.remap(image,*self.remaps[key],interpolation=cv2.INTER_LINEAR)


class PCIeCapture:
    """Read the existing 1280x720 RGB565 DMA stream through its native capture code."""

    def __init__(self, library, device="/dev/pango_pci_driver", dma_delay_loops=2500, pci_device=None,
                 dma_chunk_bytes=2560, source_channel=2):
        if source_channel not in range(4):
            raise ValueError("PCIe channel must be 0, 1, 2 or 3")
        self.source_channel = min(source_channel, 2)
        self.front_channel = self.source_channel
        affinity=os.sched_getaffinity(0) if hasattr(os,'sched_getaffinity') else None
        self.screen_executor=(ProcessPoolExecutor(max_workers=2,mp_context=get_context('spawn'),
            initializer=_camera_screen_worker,initargs=(affinity,))
            if self.source_channel<2 else None)
        self.camera_warp=(NativeCameraWarp(Path(__file__).resolve().parent/'native_camera/libcamera_threads.so')
                          if self.source_channel<2 else None)
        self.camera_views = [CameraScreenView(self.screen_executor,self.camera_warp),CameraScreenView(self.screen_executor,self.camera_warp)] if self.source_channel < 2 else None
        self.camera_view = self.camera_views[self.front_channel] if self.camera_views else None
        if dma_delay_loops < 0:
            raise ValueError("DMA delay cannot be negative")
        if dma_chunk_bytes not in (2560,3840) or (dma_chunk_bytes==3840 and pci_device is None):
            raise ValueError('3840-byte DMA requires the verified PCI endpoint')
        self.lib = ct.CDLL(str(Path(library).resolve()))
        signatures = {
            "traffic_capture_open": ([ct.c_char_p, ct.c_int], ct.c_void_p),
            "traffic_capture_read": ([ct.c_void_p, ct.c_void_p, ct.c_size_t,
                                      ct.POINTER(ct.c_double), ct.POINTER(ct.c_double)], ct.c_int),
            "traffic_capture_close": ([ct.c_void_p], None),
            "traffic_capture_error": ([], ct.c_char_p),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(self.lib, name)
            function.argtypes, function.restype = arguments, result
        self.handle = self.lib.traffic_capture_open(str(device).encode(), dma_delay_loops)
        if not self.handle:
            raise RuntimeError(self.lib.traffic_capture_error().decode(errors="replace"))
        self.frames = 0
        self.dma_ms = self.convert_ms = 0.0
        self.status_map = None
        self.capture_map = self.capture_word = self.capture_mode = None
        self.native_source_word = None
        self.video_status = None
        if pci_device is not None:
            from traffic_fpga import read_video_status
            try:
                with (Path("/sys/bus/pci/devices") / pci_device / "resource0").open("rb", buffering=0) as handle:
                    self.status_map = mmap.mmap(handle.fileno(), 0x2000, access=mmap.ACCESS_READ)
                self.video_status = read_video_status(self.status_map)
                format_word = struct.unpack_from('<I', self.status_map, 0x20c)[0]
                if dma_chunk_bytes==3840:
                    if not format_word & 4096:
                        raise RuntimeError('FPGA firmware does not support chunked DMA')
                    configure = self.lib.traffic_capture_set_chunk
                    configure.argtypes = [ct.c_void_p, ct.c_int]
                    configure.restype = ct.c_int
                    if configure(self.handle, 3840) != 1:
                        raise RuntimeError(self.lib.traffic_capture_error().decode(errors='replace'))
                if self.video_status["full_hdmi_capture"]:
                    from traffic_fpga import CAPTURE_OFFSET
                    fd = os.open(Path("/sys/bus/pci/devices") / pci_device / "resource1", os.O_RDWR | os.O_SYNC)
                    try:
                        self.capture_map = mmap.mmap(fd, 0x1000, flags=mmap.MAP_SHARED,
                                                     prot=mmap.PROT_READ | mmap.PROT_WRITE)
                    finally:
                        os.close(fd)
                    self.capture_word = ct.c_uint32.from_buffer(self.capture_map, CAPTURE_OFFSET)
                    if self.video_status["full_camera_capture"]:
                        from traffic_fpga import NATIVE_OFFSET
                        self.native_source_word = ct.c_uint32.from_buffer(self.capture_map, NATIVE_OFFSET)
                        self.select_native_source(self.source_channel)
            except Exception:
                self.release()
                raise

    @staticmethod
    def channel_bounds(channel):
        if channel not in range(4):
            raise ValueError("PCIe channel must be 0, 1, 2 or 3")
        x, y = (channel % 2) / 2, (channel // 2) / 2
        return (x, y, x + .5, y + .5)

    def select_native_source(self, channel):
        from traffic_fpga import read_video_status
        if channel not in (0, 1, 2) or self.native_source_word is None:
            raise ValueError('Native source selection requires the matching camera firmware')
        self.native_source_word.value = channel
        deadline = time.monotonic() + 1
        while True:
            self.video_status = read_video_status(self.status_map)
            if self.video_status['native_source'] == channel:
                self.source_channel = channel
                return
            if time.monotonic() >= deadline:
                raise RuntimeError('FPGA did not acknowledge the native video source')
            time.sleep(.001)

    def isOpened(self):
        return bool(self.handle)

    def resize_preview(self, frame, size):
        function = self.lib.traffic_resize_bgr
        function.argtypes = [ct.c_void_p,ct.c_int,ct.c_int,ct.c_void_p,ct.c_int,ct.c_int]
        function.restype = ct.c_int
        output = np.empty((size[1],size[0],3),dtype=np.uint8)
        status = function(frame.ctypes.data,frame.shape[1],frame.shape[0],output.ctypes.data,*size)
        if status != 1:
            raise RuntimeError(f'Native preview resize failed: {status}')
        return output

    def get(self, property_id):
        return {cv2.CAP_PROP_FRAME_WIDTH: 1280, cv2.CAP_PROP_FRAME_HEIGHT: 720,
                cv2.CAP_PROP_FPS: 30, cv2.CAP_PROP_FRAME_COUNT: 0,
                cv2.CAP_PROP_POS_FRAMES: self.frames}.get(property_id, 0)

    def set_capture_mode(self, full_hdmi):
        from traffic_fpga import CAPTURE_OFFSET, CAPTURE_MAGIC
        mode = int(full_hdmi)
        if self.capture_mode == mode:
            return
        if self.capture_word is None:
            if mode:
                raise RuntimeError("FPGA firmware does not support full HDMI capture")
            return
        self.capture_word.value = mode
        deadline = time.monotonic() + 1
        while True:
            signature, state = struct.unpack_from("<2I", self.status_map, CAPTURE_OFFSET)
            if signature != CAPTURE_MAGIC:
                raise RuntimeError("Full HDMI capture register signature mismatch")
            if state & 7 == (7 if mode else 4):
                self.capture_mode = mode
                return
            if time.monotonic() >= deadline:
                raise RuntimeError(f"FPGA did not acknowledge capture mode {mode}: {state:#x}")
            time.sleep(.001)

    def read(self, full_hdmi=False):
        if not self.handle:
            return False, None
        self.set_capture_mode(full_hdmi)
        frame = np.empty((720, 1280, 3), dtype=np.uint8)
        before_mask = None
        if self.status_map is not None:
            from traffic_fpga import read_video_status
            before_mask = read_video_status(self.status_map)["valid_mask"]
        dma_ms, convert_ms = ct.c_double(), ct.c_double()
        if self.lib.traffic_capture_read(self.handle, frame.ctypes.data, frame.nbytes,
                                        ct.byref(dma_ms), ct.byref(convert_ms)) != 1:
            raise RuntimeError(self.lib.traffic_capture_error().decode(errors="replace"))
        self.frames += 1
        self.dma_ms, self.convert_ms = dma_ms.value, convert_ms.value
        if self.status_map is not None:
            self.video_status = read_video_status(self.status_map)
            self.video_status["valid_mask"] &= before_mask
            if full_hdmi:
                native_ok = (not self.video_status.get("full_camera_capture") or
                             (self.video_status["native_ready"] and
                              self.video_status["native_source"] == self.source_channel))
                if not native_ok or not self.video_status["valid_mask"] & (1 << self.source_channel):
                    frame[:] = 0
            else:
                for channel in range(4):
                    if not self.video_status["valid_mask"] & (1 << channel):
                        x, y = (channel % 2) * 640, (channel // 2) * 360
                        frame[y:y + 360, x:x + 640] = 0
        if self.camera_views is not None:
            if full_hdmi:
                if self.source_channel<2:
                    view = self.camera_views[self.source_channel]
                    view.update(frame, time.monotonic())
                    frame = view.apply(frame)
            else:
                for channel, view in enumerate(self.camera_views):
                    x = channel*640
                    tile = frame[:360, x:x+640]
                    if self.native_source_word is None:
                        view.update(tile, time.monotonic())
                    frame[:360, x:x+640] = view.apply(tile)
            if self.video_status is not None:
                self.video_status.update(camera_screen_revision=self.camera_view.revision,
                                         camera_screen_active=self.camera_view.quad is not None,
                                         camera_screen_revisions=tuple(view.revision for view in self.camera_views))
        return True, frame

    def release(self):
        if getattr(self,'screen_executor',None) is not None:
            self.screen_executor.shutdown(wait=True)
            self.screen_executor=None
        if getattr(self,'camera_warp',None) is not None:
            self.camera_warp.close()
        self.capture_word = None
        self.native_source_word = None
        if self.capture_map is not None:
            self.capture_map.close()
            self.capture_map = None
        if self.status_map is not None:
            self.status_map.close()
            self.status_map = None
        if self.handle:
            self.lib.traffic_capture_close(self.handle)
            self.handle = None


class TensorAttr(ct.Structure):
    _fields_ = [("index", ct.c_uint32), ("n_dims", ct.c_uint32), ("dims", ct.c_uint32 * 16),
                ("name", ct.c_char * 256), ("n_elems", ct.c_uint32), ("size", ct.c_uint32),
                ("fmt", ct.c_int), ("type", ct.c_int), ("qnt_type", ct.c_int), ("fl", ct.c_int8),
                ("zp", ct.c_int32), ("scale", ct.c_float), ("w_stride", ct.c_uint32),
                ("size_with_stride", ct.c_uint32), ("pass_through", ct.c_uint8), ("h_stride", ct.c_uint32)]


class Input(ct.Structure):
    _fields_ = [("index", ct.c_uint32), ("buf", ct.c_void_p), ("size", ct.c_uint32),
                ("pass_through", ct.c_uint8), ("type", ct.c_int), ("fmt", ct.c_int)]


class Output(ct.Structure):
    _fields_ = [("want_float", ct.c_uint8), ("is_prealloc", ct.c_uint8), ("index", ct.c_uint32),
                ("buf", ct.c_void_p), ("size", ct.c_uint32)]


def letterbox_rgb(image, size):
    height, width = image.shape[:2]
    target_height, target_width = (size, size) if np.isscalar(size) else size
    gain = min(target_height / height, target_width / width)
    resized = (round(width * gain), round(height * gain))
    dx, dy = (target_width - resized[0]) / 2, (target_height - resized[1]) / 2
    if (width, height) != resized:
        image = cv2.resize(image, resized, interpolation=cv2.INTER_LINEAR)
    image = cv2.copyMakeBorder(image, round(dy - .1), round(dy + .1), round(dx - .1), round(dx + .1),
                               cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), gain, (dx, dy)


def normalized_input(rgb, metadata):
    image = rgb.astype(np.float32) / metadata["input_scale"]
    if "mean" in metadata:
        image = (image - np.array(metadata["mean"], dtype=np.float32)) / np.array(metadata["std"], dtype=np.float32)
    return np.ascontiguousarray(image.transpose(2, 0, 1)[None])


def decode_heads(outputs, metadata, threshold):
    detections = []
    for output, anchors, stride in zip(outputs, metadata["anchors"], metadata["strides"]):
        _, channels, height, width = output.shape
        if channels != 3 * (5 + len(metadata["classes"])):
            raise ValueError(f"Unexpected detection head: {output.shape}")
        head = output.reshape(3, 5 + len(metadata["classes"]), height, width).transpose(0, 2, 3, 1)
        # Only decode plausible objects; most grid cells are background.
        probabilities = metadata.get("output_activation") == "sigmoid"
        minimum = (threshold if probabilities else
                   -np.inf if threshold == 0 else np.inf if threshold == 1 else np.log(threshold / (1 - threshold)))
        indices = np.where(head[..., 4] > minimum)
        rows = head[indices]
        if not len(rows):
            continue
        scores = rows if probabilities else 1 / (1 + np.exp(-np.clip(rows, -80, 80)))
        classes = scores[:, 5:].argmax(axis=1)
        confidence = scores[:, 4] * scores[np.arange(len(scores)), classes + 5]
        keep = confidence > threshold
        scores, classes, confidence = scores[keep], classes[keep], confidence[keep]
        anchor_ids, ys, xs = (index[keep] for index in indices)
        centers = (scores[:, :2] * 2 - .5 + np.column_stack((xs, ys))) * stride
        sizes = (scores[:, 2:4] * 2) ** 2 * np.asarray(anchors)[anchor_ids]
        boxes = np.column_stack((centers - sizes / 2, centers + sizes / 2, confidence, classes))
        detections.extend(boxes.tolist())
    return detections


def nms(detections, threshold, max_det=100):
    kept = []
    for row in sorted(detections, key=lambda value: value[4], reverse=True):
        reject = False
        for previous in kept:
            if row[5] != previous[5]:
                continue
            intersection = max(0, min(row[2], previous[2]) - max(row[0], previous[0])) * \
                max(0, min(row[3], previous[3]) - max(row[1], previous[1]))
            union = (row[2] - row[0]) * (row[3] - row[1]) + \
                (previous[2] - previous[0]) * (previous[3] - previous[1]) - intersection
            if union > 0 and intersection / union > threshold:
                reject = True
                break
        if not reject:
            kept.append(row)
            if len(kept) == max_det:
                break
    return kept


class RKNNModel:
    rgb_int8_input = False
    def __init__(self, weights, runtime):
        self.config = json.loads(Path(weights).with_suffix(".json").read_text(encoding="utf-8"))
        self.names = dict(enumerate(self.config["classes"]))
        self.lib = ct.CDLL(str(runtime))
        context = ct.c_uint64
        signatures = {
            "rknn_init": [ct.POINTER(context), ct.c_void_p, ct.c_uint32, ct.c_uint32, ct.c_void_p],
            "rknn_query": [context, ct.c_int, ct.c_void_p, ct.c_uint32],
            "rknn_inputs_set": [context, ct.c_uint32, ct.POINTER(Input)],
            "rknn_run": [context, ct.c_void_p],
            "rknn_outputs_get": [context, ct.c_uint32, ct.POINTER(Output), ct.c_void_p],
            "rknn_outputs_release": [context, ct.c_uint32, ct.POINTER(Output)],
            "rknn_destroy": [context],
        }
        for name, arguments in signatures.items():
            function = getattr(self.lib, name)
            function.argtypes, function.restype = arguments, ct.c_int
        self.context = context()
        payload = ct.create_string_buffer(Path(weights).read_bytes())
        self.check(self.lib.rknn_init(ct.byref(self.context), payload, len(payload) - 1, 0, None), "init")
        try:
            counts = (ct.c_uint32 * 2)()
            self.check(self.lib.rknn_query(self.context, 0, counts, ct.sizeof(counts)), "query I/O count")
            if counts[0] != 1:
                raise ValueError("Expected a single-input RKNN model")
            self.input_attr = TensorAttr(index=0)
            self.check(self.lib.rknn_query(self.context, 1, ct.byref(self.input_attr), ct.sizeof(TensorAttr)), "query input")
            self.outputs = []
            for index in range(counts[1]):
                attr = TensorAttr(index=index)
                self.check(self.lib.rknn_query(self.context, 2, ct.byref(attr), ct.sizeof(TensorAttr)), "query output")
                shape = tuple(attr.dims[:attr.n_dims])
                self.outputs.append(shape)
            expected = [tuple(shape) for shape in self.config["output_shapes"]]
            if self.outputs != expected:
                raise ValueError(f"RKNN output shapes {self.outputs} disagree with metadata {expected}")
            input_shape = self.config.get("input_shape", [self.config["input_size"]] * 2)
            if self.input_attr.n_elems != 3 * int(np.prod(input_shape)):
                raise ValueError("RKNN input size disagrees with metadata")
            self.config.setdefault("native_float16_input", self.input_attr.type == 1 and self.input_attr.fmt == 1
                                   and self.input_attr.size_with_stride == self.input_attr.n_elems * 2)
            self.rgb_int8_input = (self.input_attr.type == 2 and self.input_attr.fmt == 1
                and self.input_attr.qnt_type == 2 and self.input_attr.zp == -128
                and self.input_attr.scale == float(np.float32(1/255))
                and self.input_attr.size_with_stride == self.input_attr.n_elems
                and self.config['input_scale'] == 255 and 'mean' not in self.config)
            if self.config.get("native_float16_input", False) and (
                    self.input_attr.type != 1 or self.input_attr.fmt != 1
                    or self.input_attr.size_with_stride != self.input_attr.n_elems * 2):
                raise ValueError("Native FP16 input requires a contiguous NHWC model input")
            print(f"RKNN initialized: {Path(weights).name}, outputs={self.outputs}", flush=True)
        except Exception:
            self.close()
            raise

    @staticmethod
    def check(code, operation):
        if code != 0:
            raise RuntimeError(f"RKNN {operation} failed: {code}")

    def infer(self, tensor):
        tensor = np.ascontiguousarray(tensor, dtype=np.float32)
        expected = (1, 3, *self.config.get("input_shape", [self.config["input_size"]] * 2))
        if tensor.shape != expected:
            raise ValueError(f"Expected input {expected}, got {tensor.shape}")
        # Prepare CPU memory before reserving the shared NPU.
        native = self.config.get("native_float16_input", False)
        tensor = np.ascontiguousarray(tensor.transpose(0, 2, 3, 1), dtype=np.float16 if native else np.float32)
        return self._infer_prepared(tensor, int(native), int(native))

    def infer_rgb(self, rgb):
        if not self.rgb_int8_input:
            return self.infer(normalized_input(rgb, self.config))
        expected = tuple(self.config.get('input_shape',[self.config['input_size']]*2)) + (3,)
        if rgb.shape != expected or rgb.dtype != np.uint8:
            raise ValueError('Native RGB input must match the model shape and uint8 pixel type')
        # round((RGB/255)/(1/255)) - 128 is exactly RGB - 128 for 8-bit pixels.
        tensor = np.bitwise_xor(rgb[None], np.uint8(128)).view(np.int8)
        return self._infer_prepared(np.ascontiguousarray(tensor), 2, 1)

    def _infer_prepared(self, tensor, input_type, pass_through):
        with _NPU_LOCK:
            input_buffer = Input(0, tensor.ctypes.data, tensor.nbytes, pass_through, input_type, 1)
            self.check(self.lib.rknn_inputs_set(self.context, 1, ct.byref(input_buffer)), "set input")
            self.check(self.lib.rknn_run(self.context, None), "run")
            outputs = (Output * len(self.outputs))()
            for index, item in enumerate(outputs):
                item.want_float, item.index = 1, index
            self.check(self.lib.rknn_outputs_get(self.context, len(outputs), outputs, None), "get outputs")
            try:
                result = []
                for item, shape in zip(outputs, self.outputs):
                    count = int(np.prod(shape))
                    if item.size != count * 4 or not item.buf:
                        raise RuntimeError("Unexpected RKNN float output buffer")
                    result.append(np.ctypeslib.as_array(ct.cast(item.buf, ct.POINTER(ct.c_float)), (count,)).copy().reshape(shape))
                if not all(np.isfinite(array).all() for array in result):
                    raise RuntimeError("RKNN returned non-finite values")
                return result
            finally:
                self.check(self.lib.rknn_outputs_release(self.context, len(outputs), outputs), "release outputs")

    def detect(self, image, args, offset_x=0, offset_y=0):
        if args.imgsz != self.config["input_size"]:
            raise ValueError("--imgsz must match the fixed model input")
        rgb, gain, (dx, dy) = letterbox_rgb(image, self.config.get("input_shape", args.imgsz))
        heads = self.infer_rgb(rgb)
        detections = nms(decode_heads(heads, self.config, args.conf_thres), args.iou_thres)
        height, width = image.shape[:2]
        for row in detections:
            row[0] = round(np.clip((row[0] - dx) / gain, 0, width)) + offset_x
            row[1] = round(np.clip((row[1] - dy) / gain, 0, height)) + offset_y
            row[2] = round(np.clip((row[2] - dx) / gain, 0, width)) + offset_x
            row[3] = round(np.clip((row[3] - dy) / gain, 0, height)) + offset_y
            row[5] = int(row[5])
        return detections

    def close(self):
        if self.context.value:
            self.check(self.lib.rknn_destroy(self.context), "destroy")
            self.context.value = 0


class OpenCVDNNModel(RKNNModel):
    """Keep the color detector in FP32 when NPU precision checks fail."""

    def __init__(self, weights, runtime=None):
        self.config = json.loads(Path(weights).with_suffix(".json").read_text(encoding="utf-8"))
        self.names = dict(enumerate(self.config["classes"]))
        self.outputs = [tuple(shape) for shape in self.config["output_shapes"]]
        self.net = cv2.dnn.readNetFromONNX(str(weights))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.output_names = self.net.getUnconnectedOutLayersNames()
        print(f"OpenCV FP32 CPU initialized: {Path(weights).name}", flush=True)

    def infer(self, tensor):
        tensor = np.ascontiguousarray(tensor, dtype=np.float32)
        if tensor.shape != (1, 3, *self.config.get("input_shape", [self.config["input_size"]] * 2)):
            raise ValueError("OpenCV input size disagrees with metadata")
        self.net.setInput(tensor)
        outputs = self.net.forward(self.output_names)
        if [array.shape for array in outputs] != self.outputs or not all(np.isfinite(array).all() for array in outputs):
            raise RuntimeError("OpenCV returned invalid detection heads")
        return outputs

    def close(self):
        self.net = None


class ONNXRuntimeCPUModel(RKNNModel):
    """Isolated FP32 color inference without changing the board environment."""

    def __init__(self, weights, runtime=None, threads=4):
        import onnxruntime as ort

        ort.disable_telemetry_events()
        self.config = json.loads(Path(weights).with_suffix(".json").read_text(encoding="utf-8"))
        self.names = dict(enumerate(self.config["classes"]))
        self.outputs = [tuple(shape) for shape in self.config["output_shapes"]]
        options = ort.SessionOptions()
        options.intra_op_num_threads, options.inter_op_num_threads = threads, 1
        self.session = ort.InferenceSession(str(weights), options, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        print(f"ONNX Runtime FP32 CPU initialized: {Path(weights).name}", flush=True)

    def infer(self, tensor):
        tensor = np.ascontiguousarray(tensor, dtype=np.float32)
        if tensor.shape != (1, 3, *self.config.get("input_shape", [self.config["input_size"]] * 2)):
            raise ValueError("ONNX Runtime input size disagrees with metadata")
        outputs = self.session.run(None, {self.input_name: tensor})
        if [array.shape for array in outputs] != self.outputs or not all(np.isfinite(array).all() for array in outputs):
            raise RuntimeError("ONNX Runtime returned invalid detection heads")
        return outputs

    def close(self):
        self.session = None
