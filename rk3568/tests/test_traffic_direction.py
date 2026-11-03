import importlib.util
import unittest
import json
import tempfile
import io
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import torch
from PIL import Image
from traffic_rknn import decode_heads, letterbox_rgb, nms, normalized_input

from prepare_direction_data import square_crop, public_semantics, prepare_generalization, save_prepared_image
from traffic_light_3way_v2_modified import (
    DirectionClassifier, DirectionHistory, StableState, VehicleSimulation, assign_calibrated_direction, control_rectangles, direction_states, filter_signal_group,
    detect_roi, draw_driving_arrow, draw_overlay, handle_mouse, make_decision, parse_opt, select_signal_group, validate_roi,
)


class PCIeCaptureTests(unittest.TestCase):
    def test_verified_camera_red_housing_keeps_identity_without_reclassifying_round_lamp(self):
        from types import SimpleNamespace
        from traffic_light_3way_v2_modified import FrameAnalyzer
        args=SimpleNamespace(pcie_capture=True,front_channel=0,camera_generation=0,signal_roi=None,
            confirm_frames=1,top_ratio=1,tiles=1,show=False,conf_thres=.65,iou_thres=.45,
            direction_centers=None,circle_direction=None)
        classifier=MagicMock();classifier.predict.return_value=[('RIGHT',.99)]
        analyzer=FrameAnalyzer(SimpleNamespace(names={0:'red'}),classifier,None)
        analyzer.generation=(0,0,0);analyzer.group_roi=None
        analyzer.filters={key:StableState(1) for key in ('LEFT','STRAIGHT','RIGHT')}
        analyzer.direction_history=DirectionHistory(1)
        box=[50,20,80,100,.42,0]
        analyzer.direction_history.update([box],[('LEFT',.99)],preserve_identity=True)
        with patch('traffic_light_3way_v2_modified.detect_roi',return_value=[box]), \
             patch.object(analyzer,'reuse_live_tracks',return_value=True), \
             patch('traffic_light_3way_v2_modified.observe_light_color',return_value=('red',1.)):
            result=analyzer(np.zeros((140,200,3),np.uint8),args,2,0,0,1)
        classifier.predict.assert_not_called()
        self.assertEqual([(x['direction'],x['box']) for x in result['lights']],[('LEFT',box[:4])])

    def test_verified_camera_green_housing_is_reclassified(self):
        from types import SimpleNamespace
        from traffic_light_3way_v2_modified import FrameAnalyzer
        args=SimpleNamespace(pcie_capture=True,front_channel=0,camera_generation=0,signal_roi=None,
            confirm_frames=1,top_ratio=1,tiles=1,show=False,conf_thres=.65,iou_thres=.45,
            direction_centers=None,circle_direction=None)
        classifier=MagicMock();classifier.predict.return_value=[('RIGHT',.99)]
        analyzer=FrameAnalyzer(SimpleNamespace(names={0:'green'}),classifier,None)
        analyzer.generation=(0,0,0);analyzer.group_roi=None
        analyzer.filters={key:StableState(1) for key in ('LEFT','STRAIGHT','RIGHT')}
        analyzer.direction_history=DirectionHistory(1)
        box=[50,20,80,100,.42,0]
        analyzer.direction_history.update([box],[('LEFT',.99)],preserve_identity=True)
        with patch('traffic_light_3way_v2_modified.detect_roi',return_value=[box]), \
             patch.object(analyzer,'reuse_live_tracks',return_value=True), \
             patch('traffic_light_3way_v2_modified.observe_light_color',return_value=('green',1.)):
            result=analyzer(np.zeros((140,200,3),np.uint8),args,2,0,0,1)
        classifier.predict.assert_called_once()
        self.assertEqual(result['lights'][0]['direction'],'RIGHT')

    def test_unverified_camera_geometry_cannot_reuse_a_red_direction(self):
        from types import SimpleNamespace
        from traffic_light_3way_v2_modified import FrameAnalyzer
        args=SimpleNamespace(pcie_capture=True,front_channel=0,camera_generation=0,signal_roi=None,
            confirm_frames=1,top_ratio=1,tiles=1,show=False,conf_thres=.65,iou_thres=.45,
            direction_centers=None,circle_direction=None)
        classifier=MagicMock();classifier.predict.return_value=[('UNKNOWN',.7)]
        analyzer=FrameAnalyzer(SimpleNamespace(names={0:'red'}),classifier,None)
        analyzer.generation=(0,0,0);analyzer.group_roi=None
        analyzer.filters={key:StableState(1) for key in ('LEFT','STRAIGHT','RIGHT')}
        analyzer.direction_history=DirectionHistory(1)
        box=[50,20,80,100,.42,0]
        analyzer.direction_history.update([box],[('LEFT',.99)],preserve_identity=True)
        with patch('traffic_light_3way_v2_modified.detect_roi',return_value=[box]), \
             patch.object(analyzer,'reuse_live_tracks',return_value=False):
            result=analyzer(np.zeros((140,200,3),np.uint8),args,2,0,0,1)
        classifier.predict.assert_called_once()
        self.assertEqual(result['lights'],[])

    def test_only_strong_current_detection_refreshes_a_false_other_housing(self):
        from types import SimpleNamespace
        from traffic_light_3way_v2_modified import FrameAnalyzer
        for confidence,expected in ((.9,'LEFT'),(.42,None)):
            with self.subTest(confidence=confidence):
                args=SimpleNamespace(pcie_capture=True,front_channel=0,camera_generation=0,signal_roi=None,
                    confirm_frames=1,top_ratio=1,tiles=1,show=False,conf_thres=.65,iou_thres=.45,
                    direction_centers=None,circle_direction=None)
                classifier=MagicMock();classifier.predict.return_value=[('OTHER',.8)]
                analyzer=FrameAnalyzer(SimpleNamespace(names={0:'green'}),classifier,None)
                analyzer.generation=(0,0,0);analyzer.group_roi=None
                analyzer.filters={key:StableState(1) for key in ('LEFT','STRAIGHT','RIGHT')}
                analyzer.direction_history=DirectionHistory(1)
                box=[50,20,80,100,confidence,0]
                analyzer.direction_history.update([box],[('LEFT',.99)],preserve_identity=True)
                with patch('traffic_light_3way_v2_modified.detect_roi',return_value=[box]), \
                     patch.object(analyzer,'reuse_live_tracks',return_value=True), \
                     patch('traffic_light_3way_v2_modified.observe_light_color',return_value=('green',1.)):
                    result=analyzer(np.zeros((140,200,3),np.uint8),args,2,0,0,1)
                self.assertEqual([x['direction'] for x in result['lights']],['LEFT'])
                self.assertEqual(analyzer.direction_history.tracks[0]['last_seen'],
                                 analyzer.direction_history.frame if expected is not None else 1)

    def test_camera_fresh_confirmed_housing_suppresses_overlapping_old_track(self):
        from types import SimpleNamespace
        from traffic_light_3way_v2_modified import FrameAnalyzer
        args=SimpleNamespace(pcie_capture=True,front_channel=0,camera_generation=0,signal_roi=None,
            confirm_frames=1,top_ratio=1,tiles=1,show=False,conf_thres=.65,iou_thres=.45,
            direction_centers=None,circle_direction=None)
        classifier=MagicMock();classifier.predict.return_value=[('STRAIGHT',.99)]
        analyzer=FrameAnalyzer(SimpleNamespace(names={0:'red'}),classifier,None)
        analyzer.generation=(0,0,0);analyzer.group_roi=None
        analyzer.filters={key:StableState(1) for key in ('LEFT','STRAIGHT','RIGHT')}
        analyzer.direction_history=DirectionHistory(1)
        analyzer.direction_history.update([[50,20,150,120,.9,0]],[('STRAIGHT',.99)],preserve_identity=True)
        fresh=[80,35,120,85,.9,0]
        with patch('traffic_light_3way_v2_modified.detect_roi',return_value=[fresh]), \
             patch.object(analyzer.direction_history,'advance',return_value=None):
            result=analyzer(np.zeros((140,200,3),np.uint8),args,2,0,0,1)
        self.assertEqual(len(result['lights']),1)
        self.assertEqual(result['lights'][0]['box'],fresh[:4])




    def test_camera_weak_detection_recovers_confirmed_housing_without_admitting_new_false_boxes(self):
        from types import SimpleNamespace
        from traffic_light_3way_v2_modified import FrameAnalyzer
        args=SimpleNamespace(pcie_capture=True,front_channel=0,signal_roi=None,confirm_frames=1,
            top_ratio=1,tiles=1,show=False,iou_thres=.45,conf_thres=.65,direction_centers=None,circle_direction=None)
        classifier=MagicMock()
        analyzer=FrameAnalyzer(SimpleNamespace(names={0:'red'}),classifier,None)
        analyzer.generation=(0,0,0);analyzer.group_roi=None
        analyzer.filters={key:StableState(1) for key in ('LEFT','STRAIGHT','RIGHT')}
        analyzer.direction_history=DirectionHistory(1)
        analyzer.direction_history.update([[100,80,130,150,.9,0]],[('LEFT',.99)],preserve_identity=True)
        weak=[101,81,131,151,.42,0];new_false=[200,80,230,150,.55,0]
        new_valid=[300,80,330,150,.45,0]
        classifier.predict.side_effect=lambda image,boxes:[('RIGHT',.92) if d==new_valid else ('UNKNOWN',.7) for d in boxes]
        with patch('traffic_light_3way_v2_modified.detect_roi',return_value=[weak,new_false,new_valid]) as detect, \
             patch.object(analyzer.direction_history,'advance',return_value=None):
            result=analyzer(np.zeros((720,1280,3),np.uint8),args,2,0,0,1)
        self.assertEqual(detect.call_args.args[2].conf_thres,.35)
        self.assertEqual(args.conf_thres,.65)
        self.assertCountEqual(classifier.predict.call_args.args[1],[weak,new_false,new_valid])
        self.assertCountEqual([item['direction'] for item in result['lights']],['LEFT','RIGHT'])

    def test_camera_confident_direction_recovers_immediately_but_ambiguous_and_hdmi_still_confirm(self):
        box=[10,10,34,80,.8,0]
        for camera,score,expected in ((True,.95,'LEFT'),(True,.85,'UNKNOWN'),(False,.95,'UNKNOWN')):
            history=DirectionHistory(confirm_frames=2)
            self.assertEqual(history.update([box],[('LEFT',score)],preserve_identity=camera)[0][0],expected)
            self.assertEqual(history.update([box],[('LEFT',score)],preserve_identity=camera)[0][0],'LEFT')

    def test_camera_housing_cell_detections_do_not_create_duplicate_tracks_or_merge_neighbours(self):
        from traffic_light_3way_v2_modified import camera_housing_overlap
        whole=[10,10,34,80,.8,0];cell=[12,12,32,32,.99,0];neighbour=[45,10,69,80,.9,0]
        self.assertTrue(camera_housing_overlap(whole,cell))
        self.assertFalse(camera_housing_overlap(whole,neighbour))
        for camera,expected in ((False,2),(True,1)):
            history=DirectionHistory(confirm_frames=1)
            history.update([whole],[('LEFT',.99)],preserve_identity=camera)
            history.update([cell],[('LEFT',.99)],preserve_identity=camera)
            self.assertEqual(len(history.tracks),expected)

    def test_hdmi_never_loads_camera_library_and_observations_use_their_own_profile(self):
        from traffic_rknn import NativeSignals
        with patch('traffic_rknn.ct.CDLL') as load:
            legacy=NativeSignals(MagicMock())
            load.assert_not_called()
        control=MagicMock();control.traffic_camera_color.return_value=2
        control.traffic_camera_configure_threads.return_value=1
        library=MagicMock();library.traffic_light_color.return_value=0
        with patch('traffic_rknn.ct.CDLL',return_value=control):
            camera=NativeSignals(library,camera=True)
            crop=np.zeros((6,6,3),np.uint8)
            self.assertEqual(camera.observe(crop,camera=True)[0],'green')
            self.assertEqual(camera.observe(crop)[0],'red')
        control.traffic_camera_color.assert_called_once()
        control.traffic_camera_configure_threads.assert_called_once_with()
        library.traffic_light_color.assert_called_once()

    def native_signal_alignment(self,camera=True):
        import ctypes as ct
        from itertools import count
        from traffic_rknn import NativeSignals
        library=MagicMock();library.traffic_alignment_open.side_effect=count(1)
        def align(handle,source,current,width,height,source_stride,current_stride,source_id,current_id,
                  boxes,number,original_width,original_height,output,kept,motion):
            def array(address,kind,size):
                return np.ctypeslib.as_array((kind*size).from_address(address))
            array(output,ct.c_double,number*4)[:]=array(boxes,ct.c_double,number*4)+np.tile([2.,1.,2.,1.],number)
            array(kept,ct.c_int,number)[:]=np.arange(number)
            array(motion,ct.c_double,9)[:]=[1,0,2,0,1,1,0,0,1]
            return number
        library.traffic_align_boxes.side_effect=align
        signals=NativeSignals(library)
        if camera:signals.camera_library=MagicMock()
        return signals,library

    def test_native_camera_half_alignment_preserves_original_boxes_pixels_and_color_crop(self):
        signals,library=self.native_signal_alignment()
        source=np.full((120,160),30,np.uint8);current=np.full_like(source,90)
        light={'box':[40,20,100,100],'direction':'LEFT','color':'red'};cache={}
        with patch('traffic_rknn.cv2.resize',wraps=cv2.resize) as resize:
            result=signals.align(source,current,[light],(240,320),cache)
            signals.align(source,current,[light],(240,320),cache)
            self.assertEqual(resize.call_count,2)
        self.assertEqual(library.traffic_align_boxes.call_args.args[3:5],(80,60))
        self.assertEqual(result,[{**light,'box':[42.,21.,102.,101.]}])
        self.assertIs(cache['_native_alignment_half_frames'][0][0],source)
        self.assertIs(cache['_native_alignment_half_frames'][1][0],current)
        small_current=cache['_native_alignment_half_frames'][1][1]
        with patch('traffic_rknn.cv2.resize',wraps=cv2.resize) as resize:
            signals.align(current,current+1,[light],(240,320),cache)
            self.assertEqual(resize.call_count,1)
        self.assertIs(cache['_native_alignment_half_frames'][0][1],small_current)
        np.testing.assert_array_equal(source,np.full_like(source,30))
        np.testing.assert_array_equal(current,np.full_like(current,90))
        np.testing.assert_array_equal(cache['motion'],[[1,0,2],[0,1,1],[0,0,1]])
        crop=np.full((40,30,3),(0,0,255),np.uint8)
        signals.camera_library.traffic_camera_color.return_value=0
        self.assertEqual(signals.observe(crop,camera=True)[0],'red')
        self.assertEqual(signals.camera_library.traffic_camera_color.call_args.args[:4],
                         (crop.ctypes.data,30,40,crop.strides[0]))
        np.testing.assert_array_equal(crop,np.full_like(crop,(0,0,255)))

    def test_native_small_lights_and_single_hdmi_keep_full_alignment(self):
        source=np.zeros((120,160),np.uint8);current=source+1
        for camera,boxes in ((True,[[40,20,84,100]]),
                             (True,[[40,20,100,100],[110,20,130,100]]),
                             (False,[[40,20,100,100]])):
            with self.subTest(camera=camera,boxes=boxes):
                signals,library=self.native_signal_alignment(camera)
                lights=[{'box':box} for box in boxes]
                with patch('traffic_rknn.cv2.resize',side_effect=AssertionError('full resolution required')):
                    self.assertEqual(len(signals.align(source,current,lights,(240,320),{})),len(lights))
                self.assertEqual(library.traffic_align_boxes.call_args.args[3:5],(160,120))

    def test_native_half_alignment_failure_or_dropped_box_falls_back_to_separate_full_state(self):
        source=np.zeros((120,160),np.uint8);current=source+1
        lights=[{'box':[40,20,100,100]},{'box':[120,20,180,100]}]
        for status in (-1,0,1):
            with self.subTest(status=status):
                signals,library=self.native_signal_alignment();cache={}
                native=library.traffic_align_boxes.side_effect
                library.traffic_align_boxes.side_effect=lambda *args: status if args[3]==80 else native(*args)
                self.assertEqual(len(signals.align(source,current,lights,(240,320),cache)),2)
                calls=library.traffic_align_boxes.call_args_list
                self.assertEqual([call.args[3:5] for call in calls],[(80,60),(160,120)])
                self.assertNotEqual(calls[0].args[0],calls[1].args[0])
                self.assertIs(cache['_native_alignment_frames'][0],source)
                self.assertIs(cache['_native_alignment_frames'][1],current)

    def test_unchanged_camera_signal_roi_keeps_small_boxes_when_hdmi_changes(self):
        signals,library=self.native_signal_alignment()
        source=np.full((240,640),50,np.uint8);current=source.copy();current[200:]=100
        light={'box':[90,40,102,70],'direction':'LEFT','color':'red'};cache={}
        with patch('traffic_rknn.cv2.resize',side_effect=AssertionError('small light must not be resized')):
            self.assertEqual(signals.align(source,current,[light],source.shape,cache),[light])
        library.traffic_align_boxes.assert_not_called()
        np.testing.assert_array_equal(cache['motion'],np.eye(3))
        hdmi,hdmi_library=self.native_signal_alignment(camera=False)
        hdmi.align(source,current,[light],source.shape,{})
        hdmi_library.traffic_align_boxes.assert_called_once()
        crop=np.full((30,12,3),(0,255,0),np.uint8)
        signals.camera_library.traffic_camera_color.return_value=2
        self.assertEqual(signals.observe(crop,camera=True)[0],'green')
        self.assertEqual(signals.camera_library.traffic_camera_color.call_args.args[0],crop.ctypes.data)

    def test_single_changed_pixel_in_current_or_retained_native_roi_uses_original_alignment(self):
        signals,library=self.native_signal_alignment();cache={}
        source=np.zeros((240,640),np.uint8);current=source.copy();current[90,70]=1
        light={'box':[100,50,120,80]}
        signals.align(source,current,[light],source.shape,cache)
        self.assertEqual(library.traffic_align_boxes.call_count,1)
        next_frame=current.copy();next_frame[90,70]=2
        signals.align(current,next_frame,[{'box':[300,50,320,80]}],source.shape,cache)
        self.assertEqual(library.traffic_align_boxes.call_count,2)
        for box in ([float('nan'),10,30,50],[-1,10,30,50],[630,10,650,50],[30,10,20,50]):
            library.traffic_align_boxes.reset_mock()
            signals.align(source,source.copy(),[{'box':box}],source.shape,{})
            library.traffic_align_boxes.assert_called()

    def test_parallel_camera_observations_use_one_frame_and_keep_red_green_states_separate(self):
        from concurrent.futures import ThreadPoolExecutor
        from traffic_light_3way_v2_modified import current_observation
        frame=np.full((720,1280,3),25,np.uint8)
        boxes=([200,90,230,160],[840,90,870,160])
        cv2.circle(frame,(215,104),6,(5,5,230),-1)
        cv2.circle(frame,(855,104),6,(5,230,5),-1)
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        results=[{'frame':1,'loop':0,'group_revision':0,'captured_at':1,
                  'states':{},'lights':[{'box':box,'direction':'STRAIGHT','color':'UNKNOWN'}]} for box in boxes]
        with patch('traffic_light_3way_v2_modified.NATIVE_SIGNALS',None), ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(current_observation,result,0,0,1.1,2,[(1,0,0,gray)],(720,1280),None,
                                 frame,{key:StableState(1) for key in ('LEFT','STRAIGHT','RIGHT')},None,camera_color=True) for result in results]
            observed=[future.result() for future in futures]
        self.assertEqual([item[1]['STRAIGHT'] for item in observed],['red','green'])
        self.assertEqual([item[0][0]['box'] for item in observed],list(boxes))

    def test_camera_motion_pyramid_keeps_full_frame_box_coordinates(self):
        from traffic_light_3way_v2_modified import align_light_boxes
        source=cv2.GaussianBlur(np.random.default_rng(81).integers(0,256,(720,1280),np.uint8),(5,5),0)
        moved=cv2.warpAffine(source,np.float32([[1,0,6],[0,1,4]]),(1280,720))
        for offset in (0,640):
            box=[200+offset,120,240+offset,200]
            light={'box':box,'direction':'LEFT','color':'red'}
            for scale in (1.,.5):
                first=cv2.resize(source,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
                second=cv2.resize(moved,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
                with patch('traffic_light_3way_v2_modified.NATIVE_SIGNALS',None):
                    aligned=align_light_boxes(first,second,[light],(720,1280),{})
                self.assertEqual(len(aligned),1)
                np.testing.assert_allclose(aligned[0]['box'],np.array(box)+[6,4,6,4],atol=1.2)

    def test_native_camera_motion_cache_is_converted_from_its_own_tile_coordinates(self):
        from types import SimpleNamespace
        from traffic_light_3way_v2_modified import FrameAnalyzer
        from traffic_rknn import PCIeCapture
        detector=SimpleNamespace(config={'detection_context':'full_frame'},names={0:'red'})
        classifier=MagicMock();classifier.predict.return_value=[('LEFT',.99)]
        global_motion=np.array([[1,.01,4],[-.02,1,-2],[0,0,1]],float)
        gray=np.zeros((720,1280),np.uint8)
        for channel in (0,1):
            args=SimpleNamespace(signal_roi=None,confirm_frames=1,top_ratio=1,tiles=1,show=False,conf_thres=.65,
                                 iou_thres=.45,direction_centers=None,circle_direction=None,
                                 pcie_capture=True,front_channel=channel)
            with patch('traffic_light_3way_v2_modified.detect_roi',return_value=[[100,80,130,150,.99,0]]), \
                 patch('traffic_light_3way_v2_modified.DirectionHistory.advance',return_value=None) as advance:
                FrameAnalyzer(detector,classifier,None)(np.zeros((720,1280,3),np.uint8),args,2,0,0,1,
                    frames=[(1,0,0,gray),(2,0,0,gray)],motions={(1,2,0,0):global_motion},
                    native_bounds=PCIeCapture.channel_bounds(channel))
            converted=advance.call_args.args[3][(1,2)]
            expected=np.array([[1,.01,8],[-.02,1,-4-25.6*channel],[0,0,1]])
            np.testing.assert_allclose(converted,expected,atol=1e-9)

    def test_camera_pair_scheduler_does_not_wait_for_a_disconnected_camera(self):
        from traffic_light_3way_v2_modified import LiveFrameCapture
        live=LiveFrameCapture.__new__(LiveFrameCapture)
        live.capture=MagicMock();live.capture.capture_mode=0;live.capture.source_channel=0
        live.capture.video_status={'valid_mask':1,'full_hdmi_capture':True,'full_camera_capture':True,
                                   'native_source':0,'native_ready':True}
        live.capture.read.return_value=(True,np.full((720,1280,3),90,np.uint8))
        live.signal_worker=MagicMock();live.signal_worker.ready.return_value=True
        live.both_cameras=True;live.next_native_channel=1;live.front_channel=0;live.native_inflight=False
        live.captured_frames=0;live.mosaic=np.zeros((720,1280,3),np.uint8)
        for _ in range(2):
            packet=live._capture()
            self.assertEqual(packet['native_channel'],0)
            self.assertIsNotNone(packet['native'])
        live.capture.select_native_source.assert_not_called()

    def test_dual_camera_native_packets_alternate_sources_and_never_copy_to_the_other_camera(self):
        import time
        from traffic_light_3way_v2_modified import LiveFrameCapture
        class Capture:
            def __init__(self):
                self.source_channel=0;self.capture_mode=0;self.dma_ms=self.convert_ms=0.
                self.video_status={'full_hdmi_capture':True,'full_camera_capture':True,
                                   'native_ready':True,'native_source':0,'valid_mask':15}
                self.selected=[]
            def select_native_source(self,channel):
                self.source_channel=channel;self.selected.append(channel)
                self.video_status=dict(self.video_status,native_source=channel)
            def read(self,full_hdmi=False):
                time.sleep(.003)
                self.capture_mode=int(full_hdmi)
                frame=np.zeros((720,1280,3),np.uint8)
                if full_hdmi:frame[:]=101 if self.source_channel==0 else 202
                else:
                    frame[:360,:640]=10;frame[:360,640:]=20;frame[360:]=30
                return True,frame
            def release(self):pass
        source=Capture();worker=MagicMock();worker.ready.return_value=True
        live=LiveFrameCapture(source,worker,0,both_cameras=True)
        found=[]
        try:
            for _ in range(30):
                ok,frame=live.read();self.assertTrue(ok)
                if live.native_frame is None:continue
                channel=live.native_frame_channel;found.append(channel)
                self.assertTrue(np.all(live.native_frame==(101 if channel==0 else 202)))
                self.assertEqual(int(frame[100,100 if channel==0 else 740,0]),101 if channel==0 else 202)
                self.assertEqual(int(frame[100,740 if channel==0 else 100,0]),20 if channel==0 else 10)
                self.assertTrue(np.all(frame[360:]==30))
                if len(found)==2:break
            self.assertEqual(found,[0,1])
            self.assertIn(1,source.selected)
        finally:live.release()

    def test_camera_dim_green_is_separate_from_hdmi_and_still_requires_lamp_contrast(self):
        from traffic_light_3way_v2_modified import observe_light_color
        frame=np.full((40,20,3),25,np.uint8)
        cv2.rectangle(frame,(7,5),(12,10),(8,165,8),-1)
        with patch('traffic_light_3way_v2_modified.NATIVE_SIGNALS',None):
            self.assertEqual(observe_light_color(frame,[0,0,20,40])[0],'UNKNOWN')
            self.assertEqual(observe_light_color(frame,[0,0,20,40],camera=True)[0],'green')
            cv2.rectangle(frame,(7,5),(12,10),(8,130,8),-1)
            self.assertEqual(observe_light_color(frame,[0,0,20,40],camera=True)[0],'UNKNOWN')
            frame[:]=(8,165,8)
            self.assertEqual(observe_light_color(frame,[0,0,20,40],camera=True)[0],'UNKNOWN')

    def test_three_native_sources_rotate_and_only_hdmi_is_mirrored(self):
        from traffic_light_3way_v2_modified import LiveFrameCapture
        from types import SimpleNamespace
        source=SimpleNamespace(source_channel=0,capture_mode=0,dma_ms=0.,convert_ms=0.,
            video_status=dict(full_hdmi_capture=True,full_camera_capture=True,native_ready=True,native_source=0,valid_mask=15))
        def select(channel):
            source.source_channel=channel;source.video_status=dict(source.video_status,native_source=channel)
        def read(full_hdmi=False):
            source.capture_mode=int(full_hdmi)
            return True,np.full((720,1280,3),50+source.source_channel if full_hdmi else 10,np.uint8)
        source.select_native_source=select;source.read=read
        live=LiveFrameCapture.__new__(LiveFrameCapture)
        live.capture=source;live.both_cameras=True;live.native_channels=(0,1,2)
        live.next_native_channel=live.front_channel=live.captured_frames=0;live.native_inflight=False
        live.signal_worker=MagicMock();live.signal_worker.ready.return_value=True
        live.mosaic=np.full((720,1280,3),10,np.uint8)
        found=[]
        for _ in range(6):
            packet=live._capture()
            if packet['native'] is None:continue
            channel=packet['native_channel'];found.append(channel);frame=packet['frame']
            if channel==2:
                self.assertEqual(packet['native'].shape,(360,640,3))
                self.assertTrue(np.all(frame==10))
                self.assertNotEqual(source.source_channel,2)
            else:
                self.assertTrue(np.all(frame[:360,channel*640:(channel+1)*640]==50+channel))
                self.assertTrue(np.all(frame[360:]==10))
            if len(found)==3:break
        self.assertEqual(found,[0,1,2])

    def test_camera_keeps_confirmed_housing_direction_through_unknown_pixels_only_when_requested(self):
        box=[10,20,30,60,.99,0]
        for preserve, expected in ((False,'UNKNOWN'),(True,'LEFT')):
            history=DirectionHistory(confirm_frames=2)
            history.update([box],[('LEFT',.99)])
            history.update([box],[('LEFT',.99)])
            history.update([box],[('UNKNOWN',.4)],preserve_identity=preserve)
            result=history.update([box],[('UNKNOWN',.4)],preserve_identity=preserve)
            self.assertEqual(result[0][0],expected)

    def test_camera_screen_requires_stable_dark_bezel_and_drops_removed_screen(self):
        from traffic_rknn import CameraScreenView, camera_screen_quad
        image = np.full((360, 640, 3), 110, np.uint8)
        cv2.rectangle(image, (90,45), (550,315), (10,10,10), -1)
        cv2.rectangle(image, (100,55), (540,305), (180,150,120), -1)
        quad = camera_screen_quad(image)
        self.assertIsNotNone(quad)
        view = CameraScreenView()
        view.update(image, 0)
        self.assertIsNone(view.quad)
        view.update(image, 1.1)
        self.assertEqual(view.revision, 1)
        self.assertIsNotNone(view.quad)
        output = view.apply(image)
        self.assertEqual(output.shape, image.shape)
        self.assertGreater(output[180,5].min(), 90)
        blank = np.full_like(image, 180)
        self.assertIsNone(camera_screen_quad(blank))
        view.update(blank, 2.2)
        self.assertIsNone(view.quad)
        self.assertEqual(view.revision, 2)
        self.assertIs(view.apply(blank), blank)

    def test_camera_screen_accepts_vertical_foreshortening_but_keeps_bezel_validation(self):
        from traffic_rknn import camera_screen_quad,CameraScreenView
        image=np.full((360,640,3),110,np.uint8)
        cv2.rectangle(image,(90,65),(550,245),(10,10,10),-1)
        cv2.rectangle(image,(100,75),(540,235),(180,150,120),-1)
        found=camera_screen_quad(image)
        self.assertIsNotNone(found)
        np.testing.assert_allclose(found*[640,360],[[100,75],[540,75],[540,235],[100,235]],atol=5)
        view=CameraScreenView();view.update(image,0);view.update(image,1.1)
        self.assertIsNotNone(view.quad)
        self.assertGreater(view.apply(image)[180,5].min(),90)
        natural=np.full_like(image,110)
        cv2.rectangle(natural,(100,75),(540,235),(115,115,115),-1)
        self.assertIsNone(camera_screen_quad(natural))

    def test_screen_border_allows_dark_video_patches_without_accepting_missing_bezel(self):
        from traffic_rknn import _camera_screen_border
        image=np.full((360,640),100,np.uint8)
        cv2.rectangle(image,(90,65),(550,245),10,-1)
        cv2.rectangle(image,(100,75),(540,235),180,-1)
        quad=np.float32([[100,75],[540,75],[540,235],[100,235]])
        image[75:90,210:290]=10
        image[220:236,210:290]=10
        self.assertTrue(_camera_screen_border(image,quad)[0])
        image[:]=180
        self.assertFalse(_camera_screen_border(image,quad)[0])

    def test_camera_screen_holds_current_valid_bezel_when_video_breaks_the_contour(self):
        from traffic_rknn import CameraScreenView,camera_screen_quad
        image=np.full((360,640,3),110,np.uint8)
        cv2.rectangle(image,(90,45),(550,315),(10,10,10),-1)
        cv2.rectangle(image,(100,55),(540,305),(180,150,120),-1)
        quad=camera_screen_quad(image);view=CameraScreenView()
        with patch('traffic_rknn.camera_screen_quad',side_effect=[quad,quad,None]):
            view.update(image,0);view.update(image,1.1)
            original=view.quad.copy()
            view.update(image,2.2)
            np.testing.assert_array_equal(view.quad,original)
            self.assertEqual(view.revision,1)
            changed=image.copy()
            cv2.rectangle(changed,(100,55),(540,85),(14,14,14),-1)
            view.update(changed,3.3)
            np.testing.assert_array_equal(view.quad,original)
            self.assertEqual(view.revision,1)

    def test_camera_rectification_preserves_perspective_and_reuses_each_resolution(self):
        from traffic_rknn import CameraScreenView
        view=CameraScreenView()
        view.quad=np.array([[.16,.12],[.85,.15],[.91,.83],[.1,.86]])
        for width,height in ((640,360),(1280,720)):
            y,x=np.indices((height,width))
            image=np.stack((x%256,y%256,(x+y)%256),axis=2).astype(np.uint8)
            matrix=cv2.getPerspectiveTransform((view.quad*[width,height]).astype(np.float32),
                np.float32([[0,0],[width-1,0],[width-1,height-1],[0,height-1]]))
            expected=cv2.warpPerspective(image,matrix,(width,height))
            actual=view.apply(image)
            # Fixed-point interpolation can differ by one interpolation step at sharp edges.
            self.assertLess(np.abs(actual.astype(float)-expected).mean(),.05)
            self.assertEqual(np.count_nonzero(np.max(np.abs(actual.astype(float)-expected),axis=2)>8),0)
            with patch('traffic_rknn.cv2.perspectiveTransform',side_effect=AssertionError('Repeated mapping')):
                np.testing.assert_array_equal(view.apply(image),actual)
        self.assertEqual(len(view.remaps),2)

    def test_camera_screen_accepts_a_reflective_bright_bezel_using_relative_contrast(self):
        from traffic_rknn import camera_screen_quad,CameraScreenView
        image=np.full((360,640,3),110,np.uint8)
        cv2.rectangle(image,(90,45),(550,315),(95,95,95),-1)
        cv2.rectangle(image,(100,55),(540,305),(200,200,200),-1)
        quad=camera_screen_quad(image)
        self.assertIsNotNone(quad)
        view=CameraScreenView();view.update(image,0);view.update(image,1.1)
        view.update(image,2.2)
        self.assertEqual(view.revision,1)
        darker=image.copy()
        cv2.rectangle(darker,(100,55),(540,305),(20,20,20),-1)
        view.update(darker,3.3)
        self.assertEqual(view.revision,1)
        self.assertIsNotNone(view.quad)

    def test_confirmed_screen_uses_the_verified_boundary_instead_of_averaging_away_from_it(self):
        from traffic_rknn import CameraScreenView,camera_screen_quad
        image=np.full((360,640,3),110,np.uint8)
        cv2.rectangle(image,(90,45),(550,315),(10,10,10),-1)
        cv2.rectangle(image,(100,55),(540,305),(180,150,120),-1)
        quad=camera_screen_quad(image)
        previous=quad+np.array([8/640,0])
        with patch('traffic_rknn.camera_screen_quad',side_effect=[previous,quad]):
            view=CameraScreenView();view.update(image,0);view.update(image,1.1)
            np.testing.assert_array_equal(view.quad,quad)
            view.update(image,2.2)
            np.testing.assert_array_equal(view.quad,quad)

    def test_async_screen_activates_after_source_and_current_border_checks(self):
        from concurrent.futures import Future
        from unittest.mock import MagicMock
        from traffic_rknn import CameraScreenView,camera_screen_quad
        image=np.full((360,640,3),110,np.uint8)
        cv2.rectangle(image,(90,45),(550,315),(10,10,10),-1)
        cv2.rectangle(image,(100,55),(540,305),(180,150,120),-1)
        quad=camera_screen_quad(image);pending=Future();executor=MagicMock()
        executor.submit.return_value=pending
        view=CameraScreenView(executor);view.update(image,0)
        self.assertIsNone(view.quad)
        pending.set_result(quad)
        view.update(image,1.1)
        np.testing.assert_array_equal(view.quad,quad)
        self.assertEqual(view.revision,1)
        view.update(np.full_like(image,180),2.2)
        self.assertIsNone(view.quad)

    def test_camera_screen_search_does_not_block_capture_and_rejects_a_moved_panel(self):
        from concurrent.futures import ThreadPoolExecutor
        from threading import Event
        from traffic_rknn import CameraScreenView,camera_screen_quad
        image=np.full((360,640,3),110,np.uint8)
        cv2.rectangle(image,(90,45),(550,315),(10,10,10),-1)
        cv2.rectangle(image,(100,55),(540,305),(180,150,120),-1)
        quad=camera_screen_quad(image);entered=Event();resume=Event()
        def search(frame):
            entered.set()
            if not resume.wait(2):raise TimeoutError('screen test did not resume')
            return quad
        with ThreadPoolExecutor(max_workers=1) as executor,patch('traffic_rknn.camera_screen_quad',side_effect=search):
            view=CameraScreenView(executor)
            try:
                view.update(image,0)
                self.assertTrue(entered.wait(1))
                view.update(image,1.1)
                self.assertIsNone(view.quad)
                self.assertFalse(view.pending.done())
            finally:resume.set()
            view.pending.result(timeout=2)
            view.update(np.full_like(image,180),2.2)
            self.assertIsNone(view.candidate)
            self.assertIsNone(view.quad)

    def test_camera_screen_search_runs_in_a_spawned_process_without_hardware(self):
        from concurrent.futures import ProcessPoolExecutor
        from multiprocessing import get_context
        from traffic_rknn import camera_screen_quad,_camera_screen_worker
        image=np.full((360,640,3),110,np.uint8)
        cv2.rectangle(image,(90,45),(550,315),(10,10,10),-1)
        cv2.rectangle(image,(100,55),(540,305),(180,150,120),-1)
        with ProcessPoolExecutor(max_workers=1,mp_context=get_context('spawn'),
                                 initializer=_camera_screen_worker,initargs=(None,)) as executor:
            quad=executor.submit(camera_screen_quad,image).result(timeout=20)
        self.assertIsNotNone(quad)
        np.testing.assert_allclose(quad*[640,360],[[100,55],[540,55],[540,305],[100,305]],atol=5)

    def test_camera_screen_joins_edges_when_glare_breaks_the_closed_panel_outline(self):
        from traffic_rknn import camera_screen_quad
        image=np.full((360,640,3),110,np.uint8)
        outer=np.int32([[109,43],[521,43],[565,303],[65,303]])
        cv2.fillConvexPoly(image,outer,(95,95,95))
        quad=np.float32([[120,55],[510,55],[550,290],[80,290]])
        road=np.empty((360,640,3),np.uint8)
        road[:180]=(220,200,180)
        road[180:]=np.linspace(115,155,640,dtype=np.uint8)[None,:,None]
        cv2.line(road,(230,359),(300,180),(240,240,240),9)
        cv2.line(road,(470,359),(350,180),(0,210,245),9)
        matrix=cv2.getPerspectiveTransform(np.float32([[0,0],[639,0],[639,359],[0,359]]),quad)
        content=cv2.warpPerspective(road,matrix,(640,360))
        mask=np.zeros((360,640),np.uint8);cv2.fillConvexPoly(mask,quad.astype(int),255)
        image[mask>0]=content[mask>0]
        cv2.rectangle(image,(500,35),(639,155),(245,245,245),-1)
        found=camera_screen_quad(image)
        self.assertIsNotNone(found)
        np.testing.assert_allclose(found*[640,360],quad,atol=35)

    def test_one_camera_view_change_does_not_invalidate_the_other_camera(self):
        from traffic_light_3way_v2_modified import changed_camera_channels,FrameAnalyzer
        from types import SimpleNamespace
        self.assertEqual(changed_camera_channels((0,0),(0,1),3,3),{1})
        self.assertEqual(changed_camera_channels((0,1),(0,1),3,2),{0})
        args=SimpleNamespace(pcie_capture=True,front_channel=0,camera_generation=0,signal_roi=None,
            confirm_frames=1,top_ratio=1,tiles=1,show=False,conf_thres=.65,iou_thres=.45,
            direction_centers=None,circle_direction=None)
        classifier=MagicMock();classifier.predict.return_value=[('LEFT',.99)]
        analyzer=FrameAnalyzer(SimpleNamespace(names={0:'red'}),classifier,None)
        with patch('traffic_light_3way_v2_modified.detect_roi',return_value=[[20,20,40,70,.9,0]]), \
             patch('traffic_light_3way_v2_modified.DirectionHistory.advance',return_value=None):
            image=np.zeros((120,200,3),np.uint8)
            analyzer(image,args,1,0,0,1);first=analyzer.direction_history
            analyzer(image,args,2,0,0,2)
            self.assertIs(analyzer.direction_history,first)
            args.camera_generation=1;analyzer(image,args,3,0,0,3)
            self.assertIsNot(analyzer.direction_history,first)

    def test_camera_native_selection_only_writes_its_own_register(self):
        import struct
        from traffic_rknn import PCIeCapture
        class Bar(bytearray):
            def close(self):
                self.closed = True
        status, commands = Bar(0x2000), Bar(0x1000)
        struct.pack_into('<4I', status, 0x200, 0x34444956, 720 << 16 | 1280, 15, 2 | 2048 | 8192)
        struct.pack_into('<3I', status, 0x270, 0x4E415456, 1 | 1 << 2 | 16, 9)
        struct.pack_into('<I', commands, 0x1c0, 0x12345678)
        native = MagicMock()
        native.traffic_capture_open.return_value = 42
        with patch('traffic_rknn.ct.CDLL', return_value=native), patch('pathlib.Path.open'), \
             patch('traffic_rknn.os.O_SYNC', 0, create=True), \
             patch('traffic_rknn.mmap.MAP_SHARED', 1, create=True), \
             patch('traffic_rknn.mmap.PROT_READ', 1, create=True), \
             patch('traffic_rknn.mmap.PROT_WRITE', 2, create=True), \
             patch('traffic_rknn.os.open', return_value=99), patch('traffic_rknn.os.close'), \
             patch('traffic_rknn.mmap.mmap', side_effect=[status, commands]):
            capture = PCIeCapture('capture.so', pci_device='0002:21:00.0', source_channel=1)
            self.assertEqual(capture.video_status['native_source'], 1)
            self.assertEqual(struct.unpack_from('<I', commands, 0x270)[0], 1)
            self.assertEqual(struct.unpack_from('<I', commands, 0x1c0)[0], 0x12345678)
            capture.release()
        self.assertTrue(status.closed and commands.closed)
        native.traffic_capture_close.assert_called_once_with(42)

    def test_native_source_status_and_selected_preview_preserve_other_tiles(self):
        import struct
        from traffic_fpga import read_video_status, NATIVE_OFFSET, NATIVE_MAGIC
        from traffic_light_3way_v2_modified import native_capture_available, update_native_preview
        view = bytearray(0x2000)
        struct.pack_into('<4I', view, 0x200, 0x34444956, 720 << 16 | 1280, 15, 2 | 2048 | 8192)
        mosaic = np.full((720, 1280, 3), 17, np.uint8)
        raw = np.full_like(mosaic, 94)
        for channel in (0, 1, 2):
            struct.pack_into('<3I', view, NATIVE_OFFSET, NATIVE_MAGIC, channel | channel << 2 | 16, 7)
            status = read_video_status(view)
            self.assertEqual(status['native_good_frames'], 7)
            for selected in (0, 1, 2):
                self.assertEqual(native_capture_available(status, selected), selected == channel)
            frame = update_native_preview(mosaic, raw, channel)
            expected = mosaic.copy()
            if channel < 2:
                expected[:360, channel*640:(channel+1)*640] = 94
            else:
                expected[360:] = 94
            np.testing.assert_array_equal(frame, expected)
            frame[:] = 123
            self.assertTrue(np.all(mosaic == 17))
            self.assertTrue(np.all(raw == 94))
        struct.pack_into('<I', view, NATIVE_OFFSET, 0)
        with self.assertRaisesRegex(RuntimeError, 'Native video source'):
            read_video_status(view)

    def test_camera_detection_uses_only_its_pixels_and_maps_boxes_to_its_own_tile(self):
        from types import SimpleNamespace
        from traffic_light_3way_v2_modified import FrameAnalyzer
        from traffic_rknn import PCIeCapture
        classifier = MagicMock()
        classifier.predict.return_value = [('STRAIGHT', .99)]
        detector = SimpleNamespace(config={'detection_context': 'full_frame'}, names={0: 'red'})
        for channel in (0, 1):
            bounds = PCIeCapture.channel_bounds(channel)
            frame = np.full((360, 640, 3), 80 + channel * 40, np.uint8)
            args = SimpleNamespace(signal_roi=bounds, confirm_frames=1, top_ratio=1, tiles=1, show=True,
                                   iou_thres=.45, direction_centers=None, circle_direction=None)
            with patch('traffic_light_3way_v2_modified.detect_roi', return_value=[[100, 60, 150, 150, .99, 0]]) as detect:
                result = FrameAnalyzer(detector, classifier, None)(frame, args, 0, 0, 0, 1,
                                                                  native_bounds=bounds, display_shape=(720, 1280))
            np.testing.assert_array_equal(detect.call_args.args[1], frame)
            np.testing.assert_allclose(result['lights'][0]['box'], [100+640*channel, 60, 150+640*channel, 150])
            np.testing.assert_array_equal(result['image'][:360, 640*channel:640*(channel+1)], frame)
            self.assertEqual(result['signal_roi'], bounds)
            self.assertFalse(result['image'][360:].any())
            self.assertFalse(result['image'][:360, 640*(1-channel):640*(2-channel)].any())

    def test_named_live_sources_override_launcher_default_and_reject_file_input(self):
        for name, channel in (('cam1', 0), ('CAM2', 1), ('HDMI', 2)):
            with patch('sys.argv', ['traffic', '--pcie-capture', '--front-channel', '2', '--input-source', name]):
                self.assertEqual(parse_opt().front_channel, channel)
        with patch('sys.argv', ['traffic', '--input-source', 'CAM1']), redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_opt()

    def test_camera_buttons_change_input_without_changing_driving_command(self):
        import traffic_light_3way_v2_modified as traffic
        car = VehicleSimulation('RIGHT')
        car.command('STOP')
        selected = [2]
        actions = traffic.CONTROLS + tuple(traffic.INPUT_CHANNELS)
        for action, (x1, y1, x2, y2) in control_rectangles((430, 640), actions):
            if action in traffic.INPUT_CHANNELS:
                handle_mouse(cv2.EVENT_LBUTTONDOWN, (x1+x2)//2, (y1+y2)//2, 0,
                             (car, (430, 640), (430, 640), selected))
                self.assertEqual(selected[0], traffic.INPUT_CHANNELS[action])
                self.assertEqual(car.nav, 'RIGHT')
                self.assertIs(car.manual_moving, False)

    def test_source_restart_discards_coordinates_and_keeps_same_models_and_assist_mode(self):
        import copy
        import traffic_light_3way_v2_modified as traffic
        with patch('sys.argv', ['traffic', '--pcie-capture', '--assist-mode', 'fusion', '--front-channel', '2',
                                '--signal-roi', '.1', '.6', '.3', '.8',
                                '--navigation-roi', '.02', '.02', '.2', '.27']):
            args = parse_opt()
        args.circle_direction = 'LEFT'
        args.direction_centers = [.1, .5, .9]
        args.pick_signal_group = True
        seen = []
        def run(options):
            seen.append(copy.deepcopy(vars(options)))
            return (0, 1, 2, None)[len(seen)-1]
        with patch.object(traffic, 'run', side_effect=run):
            traffic.main(args)
        self.assertEqual([x['front_channel'] for x in seen], [2, 0, 1, 2])
        for options in seen[1:]:
            for name in ('signal_roi', 'navigation_roi', 'circle_direction', 'direction_centers'):
                self.assertIsNone(options[name])
            self.assertFalse(options['pick_signal_group'])
            for name in ('assist_mode', 'source', 'weights', 'direction_weights', 'lane_weights',
                         'pedestrian_weights', 'runtime', 'pipeline_capture'):
                self.assertEqual(options[name], seen[0][name])
        self.assertEqual(len({x['output'] for x in seen}), 4)

    def test_rate_limited_inference_waits_for_and_processes_the_latest_frame(self):
        import time
        from threading import Event
        from traffic_light_3way_v2_modified import LatestFrameInference
        first,second=Event(),Event();calls=[]
        def analyze(number):
            calls.append((number,time.perf_counter()))
            (first if len(calls)==1 else second).set()
            return number
        worker=LatestFrameInference(analyze,max_fps=10)
        try:
            worker.submit((0,));self.assertTrue(first.wait(2))
            self.assertFalse(worker.ready())
            for number in range(1,30):worker.submit((number,))
            self.assertTrue(second.wait(2))
            self.assertEqual([row[0] for row in calls],[0,29])
            self.assertGreaterEqual(calls[1][1]-calls[0][1],.095)
        finally:worker.close()

    def test_larger_dma_requires_firmware_capability_and_propagates_configuration_errors(self):
        import struct
        from traffic_rknn import PCIeCapture
        class Bar(bytearray):
            closed=False
            def close(self):self.closed=True
        for capability,success in ((False,True),(True,True),(True,False)):
            with self.subTest(capability=capability,success=success):
                view=Bar(0x2000)
                struct.pack_into('<4I',view,0x200,0x34444956,720<<16|1280,15,2|(4096 if capability else 0))
                native=MagicMock();native.traffic_capture_open.return_value=42
                native.traffic_capture_set_chunk.return_value=int(success)
                native.traffic_capture_error.return_value=b'invalid DMA chunk'
                with patch('traffic_rknn.ct.CDLL',return_value=native), \
                     patch('pathlib.Path.open'), patch('traffic_rknn.mmap.mmap',return_value=view):
                    if success:
                        capture=PCIeCapture('capture.so',pci_device='0002:21:00.0',
                                            dma_chunk_bytes=3840 if capability else 2560)
                        capture.release()
                    else:
                        with self.assertRaisesRegex(RuntimeError,'invalid DMA chunk'):
                            PCIeCapture('capture.so',pci_device='0002:21:00.0',dma_chunk_bytes=3840)
                if capability:native.traffic_capture_set_chunk.assert_called_once_with(42,3840)
                else:native.traffic_capture_set_chunk.assert_not_called()
                native.traffic_capture_close.assert_called_once_with(42)
                self.assertTrue(view.closed)

    def test_live_capture_preserves_native_results_pixels_status_and_frame_order(self):
        self._check_live_capture()

    def test_prepared_frames_keep_their_exact_motion_observation_and_native_image(self):
        def prepare(packet):
            packet['person_observation']={'frame':packet['number']}
            packet['lane_observation']={'frame':packet['number']}
            packet['front_gray']=packet['frame'][360:,:640,0].copy()
            return packet
        self._check_live_capture(prepare)

    def _check_live_capture(self,prepare=None):
        import time
        from threading import Event
        from traffic_light_3way_v2_modified import LiveFrameCapture
        class Capture:
            def __init__(self):
                self.capture_mode=0;self.video_status={'full_hdmi_capture':True}
                self.dma_ms=self.convert_ms=0.;self.count=0
                self.waiting,self.resume=Event(),Event();self.closed=False
            def read(self,full_hdmi=False):
                self.count+=1
                if self.count==5:
                    self.waiting.set()
                    if not self.resume.wait(2):raise TimeoutError('test capture blocked')
                time.sleep(.002)
                self.capture_mode=int(full_hdmi)
                self.video_status={'full_hdmi_capture':True,'sample':self.count}
                return True,np.full((720,1280,3),self.count%251,np.uint8)
            def release(self):self.closed=True
        source=Capture();signal=MagicMock();signal.ready.return_value=True
        live=LiveFrameCapture(source,signal,2,prepare)
        try:
            self.assertTrue(source.waiting.wait(2))
            ok,frame=live.read();self.assertTrue(ok)
            self.assertEqual(live.frame_id,0)
            self.assertEqual(live.video_status['sample'],2)
            self.assertTrue(np.all(live.native_frame==2))
            self.assertTrue(np.all(frame[:360]==1))
            self.assertTrue(np.all(frame[360:]==2))
            original_native=live.native_frame
            frame[:]=231
            ok,frame=live.read();self.assertTrue(ok)
            self.assertEqual(live.frame_id,2)
            self.assertEqual(live.video_status['sample'],4)
            self.assertTrue(np.all(frame==4))
            self.assertTrue(np.all(original_native==2))
            source.resume.set();last=live.frame_id
            for _ in range(6):
                _,frame=live.read()
                self.assertGreater(live.frame_id,last)
                if prepare is not None:
                    self.assertEqual(live.person_observation['frame'],live.frame_id)
                    self.assertEqual(live.lane_observation['frame'],live.frame_id)
                    self.assertEqual(int(live.front_gray[40,20]),(live.frame_id+2)%251)
                self.assertEqual(int(frame[400,20,0]),(live.frame_id+2)%251)
                self.assertFalse(np.any(frame==231))
                last=live.frame_id;frame[:]=231
        finally:
            source.resume.set();live.release()
        self.assertTrue(source.closed)
        self.assertFalse(live.thread.is_alive())
        if prepare is not None:self.assertFalse(live.prepare_thread.is_alive())

    def test_live_capture_failure_is_reported_and_the_source_is_released(self):
        from traffic_light_3way_v2_modified import LiveFrameCapture
        source=MagicMock();source.capture_mode=0
        source.video_status={'full_hdmi_capture':False}
        source.read.side_effect=[(True,np.zeros((720,1280,3),np.uint8)),ValueError('bad DMA frame')]
        live=LiveFrameCapture(source,MagicMock(),2)
        try:
            with self.assertRaisesRegex(RuntimeError,'capture failed') as caught:live.read()
            self.assertIsInstance(caught.exception.__cause__,ValueError)
        finally:live.release()
        source.release.assert_called_once()

    def test_preview_optimization_preserves_every_output_pixel(self):
        from traffic_light_3way_v2_modified import display_canvas,preview_dimensions
        rng=np.random.default_rng(7)
        for shape,limits in (((720,1280),(640,430)),((201,321),(1024,600)),((360,640),(641,361))):
            frame=rng.integers(0,256,(*shape,3),dtype=np.uint8)
            w,h=preview_dimensions(shape,*limits)
            cw,ch=max(w,min(640,limits[0])),max(h,min(360,limits[1]))
            dx,dy=(cw-w)//2,(ch-h)//2
            expected=np.full((ch,cw,3),(34,31,27),dtype=np.uint8)
            expected[dy:dy+h,dx:dx+w]=(frame if (w,h)==(shape[1],shape[0]) else cv2.resize(frame,(w,h)))
            actual,_,_=display_canvas(frame,[],None,*limits)
            np.testing.assert_array_equal(actual,expected)

    def test_fusion_keeps_both_models_active_with_or_without_group_and_in_both_intents(self):
        import traffic_light_3way_v2_modified as traffic
        from traffic_rknn import PCIeCapture
        from traffic_lane import FusionSimulation

        for channel in (0, 1, 2):
            for selected in (False, True):
                for change in (False, True):
                    with self.subTest(channel=channel, selected=selected, change=change), \
                         tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
                        root=Path(folder);weights=root/'color.rknn';weights.touch()
                        bounds=PCIeCapture.channel_bounds(channel)
                        roi=(bounds[0]+.1,bounds[1]+.1,bounds[2]-.1,bounds[3]-.1)
                        options=['traffic','--assist-mode','fusion','--pcie-capture','--front-channel',str(channel),
                                 '--show','--no-record','--weights',str(weights),
                                 '--direction-weights',str(root/'direction.rknn'),'--output',str(root/'out.mp4')]
                        if selected:options+=['--signal-roi',*(str(x) for x in roi)]
                        stack.enter_context(patch('sys.argv',options))
                        stack.enter_context(redirect_stdout(io.StringIO()))
                        args=traffic.parse_opt()
                        args.realtime_threads=True
                        stack.enter_context(patch.object(traffic.os,'sched_getaffinity',create=True,return_value={0,1,2,3}))
                        restore=stack.enter_context(patch.object(traffic.os,'sched_setaffinity',create=True))
                        mosaic=np.full((720,1280,3),43,np.uint8)
                        native=np.full_like(mosaic,199)
                        capture=MagicMock();capture.capture_mode=None;capture.dma_ms=1.
                        capture.screen_executor=None
                        capture.video_status={'full_hdmi_capture':True,'valid_mask':15,
                                              'source_channels':(0,1,2,2),'good_frames':(1,1,1,1)}
                        capture.get.side_effect=lambda key:{cv2.CAP_PROP_FRAME_WIDTH:1280,
                                                           cv2.CAP_PROP_FRAME_HEIGHT:720,cv2.CAP_PROP_FPS:30}.get(key,0)
                        def read(full_hdmi=False):
                            capture.capture_mode=int(full_hdmi)
                            return True,(native if full_hdmi else mosaic).copy()
                        capture.read.side_effect=read
                        factory=stack.enter_context(patch('traffic_rknn.PCIeCapture',return_value=capture))
                        factory.channel_bounds.side_effect=PCIeCapture.channel_bounds
                        stack.enter_context(patch('traffic_rknn.RKNNModel'))
                        stack.enter_context(patch.object(traffic,'DirectionClassifier'))
                        stack.enter_context(patch('traffic_lane.LaneModel'))
                        tracker=stack.enter_context(patch('traffic_lane.LaneEvidenceTracker')).return_value
                        tracker.update.return_value={}
                        simulation=FusionSimulation('LEFT')
                        if change:simulation.command('CHANGE')
                        stack.enter_context(patch('traffic_lane.FusionSimulation',return_value=simulation))
                        lane_worker,signal_worker=MagicMock(),MagicMock()
                        for worker in (lane_worker,signal_worker):
                            worker.ready.return_value=True;worker.take.return_value=None
                        stack.enter_context(patch.object(traffic,'LatestFrameInference',side_effect=[lane_worker,signal_worker]))
                        for name in ('namedWindow','moveWindow','setMouseCallback','imshow','destroyAllWindows'):
                            stack.enter_context(patch('cv2.'+name))
                        stack.enter_context(patch('cv2.getWindowProperty',return_value=1))
                        stack.enter_context(patch('cv2.waitKey',side_effect=[-1,ord('q')]))
                        stack.enter_context(patch.object(traffic,'draw_overlay'))
                        stack.enter_context(patch('traffic_lane.draw_lanes',side_effect=lambda image,boundaries:image.fill(231)))
                        traffic.run(args)
                        restore.assert_called_once_with(0,{0,1,2,3})
                        self.assertEqual(lane_worker.submit.call_count,2)
                        self.assertGreater(signal_worker.submit.call_count,0)
                        job=signal_worker.submit.call_args.args[0]
                        expected_roi = roi if selected else bounds if channel == 2 else None
                        self.assertEqual(tuple(job[1].signal_roi) if job[1].signal_roi is not None else None, expected_roi)
                        self.assertTrue(np.all(job[0]==(199 if channel==2 else 43)))
                        if channel in (0,1):
                            self.assertEqual(job[0].shape[:2],(360,640))
                            self.assertEqual(job[-2:],(bounds,(720,1280)))
                        self.assertEqual(tuple(args.signal_roi) if args.signal_roi is not None else None,roi if selected else None)
                        self.assertEqual(simulation.lane_change,change)

    def test_confirmed_signal_selection_preserves_mosaic_for_first_native_frame(self):
        import traffic_light_3way_v2_modified as traffic
        from traffic_rknn import PCIeCapture

        for channel in (2, 3):
            with self.subTest(channel=channel), tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
                root = Path(folder)
                weights = root / 'color.rknn'
                weights.touch()
                options = ['traffic', '--pcie-capture', '--front-channel', str(channel), '--show',
                           '--pick-signal-group', '--no-record', '--weights', str(weights),
                           '--direction-weights', str(root / 'direction.rknn'), '--output', str(root / 'out.mp4')]
                stack.enter_context(patch('sys.argv', options))
                stack.enter_context(redirect_stdout(io.StringIO()))
                args = traffic.parse_opt()
                mosaic = np.full((720, 1280, 3), 43, np.uint8)
                native = np.full_like(mosaic, 199)
                capture = MagicMock()
                capture.capture_mode = None
                capture.dma_ms = 1.0
                capture.video_status = {'full_hdmi_capture': True, 'valid_mask': 15,
                                        'source_channels': (0, 1, 2, 2), 'good_frames': (1, 1, 1, 1)}
                capture.get.side_effect = lambda key: {cv2.CAP_PROP_FRAME_WIDTH: 1280,
                                                       cv2.CAP_PROP_FRAME_HEIGHT: 720,
                                                       cv2.CAP_PROP_FPS: 30}.get(key, 0)

                def read(full_hdmi=False):
                    capture.capture_mode = int(full_hdmi)
                    return True, native if full_hdmi else mosaic

                capture.read.side_effect = read
                factory = stack.enter_context(patch('traffic_rknn.PCIeCapture', return_value=capture))
                factory.channel_bounds.side_effect = PCIeCapture.channel_bounds
                stack.enter_context(patch('traffic_rknn.RKNNModel'))
                stack.enter_context(patch.object(traffic, 'DirectionClassifier'))
                worker = stack.enter_context(patch.object(traffic, 'LatestFrameInference')).return_value
                worker.ready.return_value, worker.take.return_value = True, None
                stack.enter_context(patch('cv2.selectROI', return_value=(64, 36, 320, 180)))
                for name in ('destroyWindow', 'namedWindow', 'moveWindow', 'setMouseCallback', 'destroyAllWindows'):
                    stack.enter_context(patch('cv2.' + name))
                shown = stack.enter_context(patch('cv2.imshow'))
                stack.enter_context(patch('cv2.waitKey', return_value=ord('q')))
                stack.enter_context(patch.object(traffic, 'draw_overlay'))
                traffic.run(args)
                self.assertEqual(capture.read.call_args_list, [unittest.mock.call(), unittest.mock.call(full_hdmi=True)])
                image = shown.call_args.args[1]
                self.assertTrue(np.all(image[:image.shape[0] // 2] == 43))
                self.assertTrue(np.all(image[image.shape[0] // 2:] == 199))
                self.assertIs(worker.submit.call_args.args[0][0], native)
                capture.release.assert_called_once()

    def test_native_capture_waits_for_active_mode_and_prefetch(self):
        from traffic_rknn import PCIeCapture
        from traffic_fpga import CAPTURE_MAGIC
        from types import SimpleNamespace
        capture = PCIeCapture.__new__(PCIeCapture)
        capture.capture_mode, capture.capture_word = None, SimpleNamespace(value=0)
        capture.status_map = bytearray(0x2000)
        with patch('traffic_rknn.struct.unpack_from', side_effect=[(CAPTURE_MAGIC, 1), (CAPTURE_MAGIC, 3), (CAPTURE_MAGIC, 7)]):
            capture.set_capture_mode(True)
        self.assertEqual((capture.capture_mode, capture.capture_word.value), (1, 1))
        with patch('traffic_rknn.struct.unpack_from', return_value=(CAPTURE_MAGIC, 4)):
            capture.set_capture_mode(False)
        self.assertEqual((capture.capture_mode, capture.capture_word.value), (0, 0))

    def test_native_detection_keeps_original_pixels_and_maps_back_to_selected_tile(self):
        from traffic_light_3way_v2_modified import FrameAnalyzer
        from types import SimpleNamespace
        args = SimpleNamespace(signal_roi=(.2, .65, .35, .8), direction_centers=None,
                               confirm_frames=1, top_ratio=1, tiles=1, iou_thres=.45,
                               circle_direction=None, show=True)
        frame = np.full((720, 1280, 3), 123, np.uint8)
        classifier = MagicMock()
        classifier.predict.return_value = [('LEFT', .98)]
        detector = SimpleNamespace(names={0: 'green'})
        analyzer = FrameAnalyzer(detector, classifier, None)
        with patch('traffic_light_3way_v2_modified.detect_roi', return_value=[[520, 230, 540, 280, .95, 0]]) as detect:
            result = analyzer(frame, args, 10, 0, 0, 1.5, native_bounds=(0, .5, .5, 1))
        self.assertEqual(detect.call_args.args[1].shape[:2], (346, 512))
        self.assertIs(classifier.predict.call_args.args[0], frame)
        np.testing.assert_allclose(result['lights'][0]['box'], [260, 475, 270, 500])
        self.assertEqual(result['signal_roi'], args.signal_roi)
        self.assertTrue(np.all(result['image'][360:, :640] == 123))
        self.assertTrue(np.all(result['image'][:360] == 0))

    def test_selected_group_keeps_context_without_accepting_neighbor_lights(self):
        from traffic_light_3way_v2_modified import FrameAnalyzer
        from types import SimpleNamespace

        args = SimpleNamespace(signal_roi=None, direction_centers=None,
                               confirm_frames=1, top_ratio=1, tiles=1, iou_thres=.45,
                               circle_direction=None, show=False)
        frame = np.full((240, 320, 3), 123, np.uint8)
        inside = [145, 100, 160, 130, .95, 0]
        neighbor = [80, 60, 100, 85, .98, 0]
        selections = [((.4, .4, .6, .6), (64, 32, 256, 208)),
                      ((.375, .375, .625, .625), (56, 26, 264, 214)),
                      ((.42, .42, .58, .58), (70, 36, 250, 204))]
        for roi, (x1, y1, x2, y2) in selections:
            with self.subTest(roi=roi):
                args.signal_roi = roi
                classifier = MagicMock()
                classifier.predict.return_value = [('LEFT', .99)]
                analyzer = FrameAnalyzer(SimpleNamespace(names={0: 'green'}), classifier, None)
                with patch('traffic_light_3way_v2_modified.detect_roi', return_value=[inside, neighbor]) as detect:
                    result = analyzer(frame, args, 0, 0, 0, 1.5)
                self.assertEqual(detect.call_args.args[3:], (x1, y1))
                np.testing.assert_array_equal(detect.call_args.args[1], frame[y1:y2, x1:x2])
                classifier.predict.assert_called_once_with(frame, [inside])
                self.assertEqual([light['box'] for light in result['lights']], [inside[:4]])

    def test_full_frame_model_keeps_scale_and_filters_selected_group(self):
        from traffic_light_3way_v2_modified import FrameAnalyzer
        from types import SimpleNamespace
        args=SimpleNamespace(signal_roi=(.4,.4,.6,.6),direction_centers=None,
                             confirm_frames=1,top_ratio=1,tiles=1,iou_thres=.45,
                             circle_direction=None,show=False)
        frame=np.zeros((240,320,3),np.uint8)
        inside=[145,100,160,130,.95,0]
        neighbor=[80,60,100,85,.98,0]
        detector=SimpleNamespace(names={0:'green'},config={'detection_context':'full_frame'})
        classifier=MagicMock();classifier.predict.return_value=[('LEFT',.99)]
        with patch('traffic_light_3way_v2_modified.detect_roi',return_value=[inside,neighbor]) as detect:
            result=FrameAnalyzer(detector,classifier,None)(frame,args,0,0,0,1.)
        self.assertEqual(detect.call_count,1)
        self.assertEqual(detect.call_args.args[3:],(0,0))
        self.assertEqual(detect.call_args.args[1].shape,frame.shape)
        classifier.predict.assert_called_once_with(frame,[inside])
        self.assertEqual(len(result['lights']),1)

    def test_native_direction_tracking_uses_intermediate_hdmi_frames_during_camera_motion(self):
        from traffic_light_3way_v2_modified import FrameAnalyzer
        from types import SimpleNamespace

        image = cv2.GaussianBlur(np.random.default_rng(32).integers(0, 256, (240, 320, 3), dtype=np.uint8), (3, 3), 0)
        box = [60, 80, 72, 140, .95, 0]
        shifted = [100, 80, 112, 140, .95, 0]
        args = SimpleNamespace(signal_roi=(.1, .6, .4, .95), direction_centers=None,
                               confirm_frames=2, top_ratio=1, tiles=1, iou_thres=.45,
                               circle_direction=None, show=False)
        classifier = MagicMock()
        classifier.predict.side_effect = [[('LEFT', .98)], [('LEFT', .98)], [('UNKNOWN', .3)]]
        analyzer = FrameAnalyzer(SimpleNamespace(names={0: 'red'}), classifier, None)
        frames = []
        with patch('traffic_light_3way_v2_modified.detect_roi', side_effect=[[box], [box], [shifted]]):
            for index in range(12):
                delta = max(0, index - 1) * 4
                moved = cv2.warpAffine(image, np.float32([[1, 0, delta], [0, 1, 0]]), (320, 240))
                mosaic_gray = np.full((240, 320), (index * 23) % 256, np.uint8)
                mosaic_gray[120:, :160] = cv2.cvtColor(moved[::2, ::2], cv2.COLOR_BGR2GRAY)
                frames.append((index, 0, 0, mosaic_gray))
                if index in (0, 1, 11):
                    result = analyzer(moved, args, index, 0, 0, index, frames=frames,
                                      native_bounds=(0, .5, .5, 1))
        self.assertEqual(result['lights'][0]['direction'], 'LEFT')
        np.testing.assert_allclose(result['lights'][0]['box'], [50, 160, 56, 190])

    def test_selected_group_follows_pan_without_accepting_neighbor_at_old_selection(self):
        from traffic_light_3way_v2_modified import FrameAnalyzer
        from types import SimpleNamespace

        image = cv2.GaussianBlur(np.random.default_rng(42).integers(0, 256, (240, 320, 3), np.uint8), (3, 3), 0)
        args = SimpleNamespace(signal_roi=(.4, .35, .5, .65), direction_centers=None,
                               confirm_frames=2, top_ratio=1, tiles=1, iou_thres=.45,
                               circle_direction=None, show=False)
        box = [133, 100, 150, 140, .98, 0]
        classifier = MagicMock()
        classifier.predict.return_value = [('LEFT', .99)]
        analyzer = FrameAnalyzer(SimpleNamespace(names={0: 'red'}), classifier, None)
        frames = []
        with patch('traffic_light_3way_v2_modified.detect_roi') as detect:
            for index in range(12):
                shift = max(0, index - 1) * 5
                frame = cv2.warpAffine(image, np.float32([[1, 0, shift], [0, 1, 0]]), (320, 240))
                frames.append((index, 0, 0, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
                if index in (0, 1, 11):
                    moved = [box[0]+shift, box[1], box[2]+shift, box[3], *box[4:]]
                    detect.return_value = [moved] if index < 11 else [moved, box]
                    result = analyzer(frame, args, index, 0, 0, index, frames=frames)
        self.assertEqual([light['box'] for light in result['lights']], [moved[:4]])
        self.assertEqual(result['lights'][0]['direction'], 'LEFT')
        classifier.predict.assert_called_with(frame, [moved])
        self.assertGreater(analyzer.group_roi[0], args.signal_roi[2])
        np.testing.assert_allclose(analyzer.group_roi, [.55625, .35, .65625, .65], atol=.005)

    def test_scene_cut_and_new_selection_clear_tracked_group_identity(self):
        from traffic_light_3way_v2_modified import FrameAnalyzer
        from types import SimpleNamespace

        rng = np.random.default_rng(43)
        image = cv2.GaussianBlur(rng.integers(0, 256, (240, 320, 3), np.uint8), (3, 3), 0)
        args = SimpleNamespace(signal_roi=(.4, .35, .5, .65), direction_centers=None,
                               confirm_frames=2, top_ratio=1, tiles=1, iou_thres=.45,
                               circle_direction=None, show=False)
        box = [133, 100, 150, 140, .98, 0]
        classifier = MagicMock()
        classifier.predict.return_value = [('LEFT', .99)]
        analyzer = FrameAnalyzer(SimpleNamespace(names={0: 'green'}), classifier, None)
        with patch('traffic_light_3way_v2_modified.detect_roi', return_value=[box]):
            for index in range(2):
                result = analyzer(image, args, index, 0, 0, index)
            self.assertEqual(result['lights'][0]['direction'], 'LEFT')
            cut = rng.integers(0, 256, image.shape, np.uint8)
            result = analyzer(cut, args, 2, 0, 0, 2)
            self.assertEqual(result['lights'][0]['direction'], 'UNKNOWN')
            self.assertEqual(result['states']['LEFT'], 'UNKNOWN')
            np.testing.assert_allclose(analyzer.group_roi, args.signal_roi)
            args.signal_roi = (.1, .1, .2, .3)
            result = analyzer(cut, args, 3, 0, 1, 3)
            self.assertEqual(result['lights'], [])
            np.testing.assert_allclose(analyzer.group_roi, args.signal_roi)

    def test_first_inference_uses_the_frame_on_which_user_drew_selection(self):
        from traffic_light_3way_v2_modified import FrameAnalyzer
        from types import SimpleNamespace

        image = cv2.GaussianBlur(np.random.default_rng(44).integers(0, 256, (240, 320, 3), np.uint8), (3, 3), 0)
        args = SimpleNamespace(signal_roi=(.4, .35, .5, .65), direction_centers=None,
                               confirm_frames=2, top_ratio=1, tiles=1, iou_thres=.45,
                               circle_direction=None, show=False)
        moved = cv2.warpAffine(image, np.float32([[1, 0, 30], [0, 1, 0]]), (320, 240))
        frames = [(-1, 0, 0, cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)),
                  (0, 0, 0, cv2.cvtColor(moved, cv2.COLOR_BGR2GRAY))]
        box = [163, 100, 180, 140, .98, 0]
        classifier = MagicMock()
        classifier.predict.return_value = [('LEFT', .99)]
        analyzer = FrameAnalyzer(SimpleNamespace(names={0: 'red'}), classifier, None)
        with patch('traffic_light_3way_v2_modified.detect_roi', return_value=[box]):
            result = analyzer(moved, args, 0, 0, 0, 0, frames=frames)
        self.assertEqual([light['box'] for light in result['lights']], [box[:4]])
        np.testing.assert_allclose(analyzer.group_roi, [.49375, .35, .59375, .65], atol=.005)

    def test_four_input_status_requires_matching_firmware_and_geometry(self):
        import struct
        from traffic_fpga import read_video_status, VIDEO_OFFSET, VIDEO_MAGIC
        view = bytearray(0x2000)
        struct.pack_into('<4I', view, VIDEO_OFFSET, VIDEO_MAGIC, (720 << 16) | 1280, 5, 1)
        struct.pack_into('<4I', view, VIDEO_OFFSET + 16, 12, 0, 24, 0)
        struct.pack_into('<4I', view, VIDEO_OFFSET + 32, 230400, 0, 230400, 0)
        status = read_video_status(view)
        self.assertEqual(status['valid_mask'], 5)
        self.assertEqual(status['good_frames'], (12, 0, 24, 0))
        self.assertEqual(status['source_channels'], (0, 1, 2, 3))
        struct.pack_into('<I', view, VIDEO_OFFSET + 12, 0x302)
        status = read_video_status(view)
        self.assertEqual(status['source_channels'], (0, 1, 2, 2))
        self.assertEqual(status['hdmi_flags'], {'initialized': True, 'chip_id_ok': True, 'i2c_error': False})
        struct.pack_into('<I', view, VIDEO_OFFSET + 4, 0)
        with self.assertRaisesRegex(RuntimeError, 'VID4'):
            read_video_status(view)

    def test_missing_or_newly_connected_channel_never_reuses_stale_dma_picture(self):
        from traffic_rknn import PCIeCapture
        import ctypes as ct
        capture = PCIeCapture.__new__(PCIeCapture)
        capture.handle, capture.frames, capture.status_map = 42, 0, bytearray(0x2000)
        capture.capture_mode = capture.capture_word = None
        capture.camera_view = None
        capture.camera_views = None
        capture.lib = MagicMock()

        def fill(handle, pixels, size, dma_ms, convert_ms):
            ct.memset(pixels, 255, size)
            return 1

        capture.lib.traffic_capture_read.side_effect = fill
        with patch('traffic_fpga.read_video_status', side_effect=[{'valid_mask': 5}, {'valid_mask': 15}]):
            ok, frame = capture.read()
        self.assertTrue(ok)
        self.assertTrue(np.all(frame[:360, :640] == 255))
        self.assertTrue(np.all(frame[360:, :640] == 255))
        self.assertTrue(np.all(frame[:, 640:] == 0))
        self.assertEqual(capture.video_status['valid_mask'], 5)

    def test_camera_calibration_reads_native_pixels_and_does_not_resubmit_aliased_mosaic(self):
        import ctypes as ct
        from traffic_rknn import PCIeCapture
        capture = PCIeCapture.__new__(PCIeCapture)
        capture.handle, capture.frames, capture.status_map = 42, 0, None
        capture.video_status = None
        capture.native_source_word = object()
        capture.source_channel = 1
        capture.set_capture_mode = MagicMock()
        capture.camera_views = [MagicMock(), MagicMock()]
        for view in capture.camera_views:
            view.apply.side_effect = lambda image: image
        capture.lib = MagicMock()
        def fill(handle, pixels, size, dma_ms, convert_ms):
            ct.memset(pixels, 123, size)
            return 1
        capture.lib.traffic_capture_read.side_effect = fill
        _, native = capture.read(full_hdmi=True)
        capture.camera_views[0].update.assert_not_called()
        self.assertIs(capture.camera_views[1].update.call_args.args[0], native)
        self.assertEqual(native.shape, (720, 1280, 3))
        capture.read()
        capture.camera_views[0].update.assert_not_called()
        self.assertEqual(capture.camera_views[1].update.call_count, 1)
        capture.native_source_word = None
        capture.read()
        for view in capture.camera_views:
            self.assertEqual(view.update.call_args.args[0].shape, (360, 640, 3))
        counts=[view.apply.call_count for view in capture.camera_views]
        capture.source_channel=2
        _,hdmi=capture.read(full_hdmi=True)
        self.assertEqual([view.apply.call_count for view in capture.camera_views],counts)
        self.assertTrue(np.all(hdmi==123))

    def test_capture_uses_native_bgr_buffer_and_preserves_previous_frame(self):
        import ctypes as ct
        from traffic_rknn import PCIeCapture
        native = MagicMock()
        native.traffic_capture_open.return_value = 42
        counter = [0]

        def fill(handle, pixels, size, dma_ms, convert_ms):
            self.assertEqual((handle, size), (42, 1280 * 720 * 3))
            counter[0] += 1
            ct.memset(pixels, counter[0], size)
            ct.cast(dma_ms, ct.POINTER(ct.c_double))[0] = 12.5
            ct.cast(convert_ms, ct.POINTER(ct.c_double))[0] = .3
            return 1

        native.traffic_capture_read.side_effect = fill
        with patch("traffic_rknn.ct.CDLL", return_value=native):
            capture = PCIeCapture("libtraffic_capture.so")
        native.traffic_capture_open.assert_called_once_with(b"/dev/pango_pci_driver", 2500)
        self.assertTrue(capture.isOpened())
        self.assertEqual(capture.get(cv2.CAP_PROP_FRAME_WIDTH), 1280)
        self.assertEqual(capture.get(cv2.CAP_PROP_FRAME_HEIGHT), 720)
        self.assertEqual(capture.get(cv2.CAP_PROP_FRAME_COUNT), 0)
        _, first = capture.read()
        _, second = capture.read()
        self.assertEqual(first.shape, (720, 1280, 3))
        self.assertTrue(np.all(first == 1))
        self.assertTrue(np.all(second == 2))
        self.assertEqual((capture.dma_ms, capture.convert_ms), (12.5, .3))
        capture.release()
        capture.release()
        native.traffic_capture_close.assert_called_once_with(42)
        self.assertFalse(capture.isOpened())
        self.assertEqual(capture.read(), (False, None))

    def test_driver_errors_never_return_a_previous_frame(self):
        from traffic_rknn import PCIeCapture
        native = MagicMock()
        native.traffic_capture_open.return_value = 42
        native.traffic_capture_read.return_value = 0
        native.traffic_capture_error.return_value = b"PCI_DMA_WRITE_CMD failed"
        with patch("traffic_rknn.ct.CDLL", return_value=native):
            capture = PCIeCapture("libtraffic_capture.so")
        with self.assertRaisesRegex(RuntimeError, "PCI_DMA_WRITE_CMD"):
            capture.read()
        self.assertEqual(capture.frames, 0)
        capture.release()
        native.traffic_capture_open.return_value = None
        native.traffic_capture_error.return_value = b"Driver not found"
        with patch("traffic_rknn.ct.CDLL", return_value=native):
            with self.assertRaisesRegex(RuntimeError, "Driver not found"):
                PCIeCapture("libtraffic_capture.so")

    def test_all_four_channels_keep_their_own_roi_coordinates(self):
        from traffic_rknn import PCIeCapture
        frame = np.zeros((720, 1280, 3), np.uint8)
        for channel in range(4):
            x, y = (channel % 2) * 640, (channel // 2) * 360
            frame[y:y + 360, x:x + 640] = channel * 50
        for channel in range(4):
            bounds = PCIeCapture.channel_bounds(channel)
            with patch("cv2.selectROI", return_value=(64, 36, 320, 180)) as select, patch("cv2.destroyWindow"), patch("cv2.namedWindow"):
                roi = select_signal_group(frame, bounds=bounds)
            self.assertEqual(select.call_args.args[1].shape[:2], (360, 640))
            self.assertTrue(np.all(select.call_args.args[1] == channel * 50))
            np.testing.assert_allclose(roi, [bounds[0] + .05, bounds[1] + .05,
                                          bounds[0] + .3, bounds[1] + .3])
            validate_roi(roi, bounds)
        with self.assertRaises(ValueError):
            PCIeCapture.channel_bounds(4)
        with self.assertRaisesRegex(ValueError, "front channel"):
            validate_roi([.4, .1, .7, .2], PCIeCapture.channel_bounds(0))

    def test_live_capture_cannot_loop_or_read_video_file(self):
        with patch("sys.argv", ["traffic", "--pcie-capture", "--front-channel", "3", "--show"]):
            args = parse_opt()
        self.assertTrue(args.pcie_capture)
        self.assertEqual(args.source, "/dev/pango_pci_driver")
        for options in (["--loop"], ["--source", "video.mp4"], ["--dma-delay-loops", "-1"],
                        ["--signal-roi", ".1", ".1", ".7", ".2"]):
            with patch("sys.argv", ["traffic", "--pcie-capture"] + options), redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    parse_opt()
            self.assertEqual(error.exception.code, 2)

    def test_guidance_arrow_stays_in_front_channel_even_with_display_padding(self):
        from traffic_rknn import PCIeCapture
        for shape in ((720, 1280), (360, 320)):
            for channel in range(4):
                frame = np.zeros((*shape, 3), np.uint8)
                car = VehicleSimulation()
                car.update({"STRAIGHT": "green"}, True)
                with patch("traffic_light_3way_v2_modified.draw_driving_arrow") as arrow:
                    draw_overlay(frame, [], {}, car, None, guidance_bounds=PCIeCapture.channel_bounds(channel),
                                 video_shape=(720, 1280))
                tile = arrow.call_args.args[0]
                self.assertTrue(np.shares_memory(frame, tile))
                self.assertEqual(tile.shape[:2], (360, 640) if shape == (720, 1280) else (90, 160))
                x, y = (channel % 2) * tile.shape[1], (channel // 2) * tile.shape[0]
                if shape == (360, 320):
                    y += 90
                self.assertEqual(tile.ctypes.data, frame[y:y + tile.shape[0], x:x + tile.shape[1]].ctypes.data)


    def test_mirrored_view_copies_video_and_guidance_together(self):
        from traffic_rknn import PCIeCapture
        frame = np.full((720, 1280, 3), 50, np.uint8)
        frame[360:, 640:] = 10
        car = VehicleSimulation()
        car.update({"STRAIGHT": "green"}, True)
        lights = [{"box": [280, 470, 298, 520], "direction": "STRAIGHT", "color": "green",
                   "color_conf": .99}]
        draw_overlay(frame, lights, {"STRAIGHT": "green"}, car, None,
                     guidance_bounds=PCIeCapture.channel_bounds(2),
                     mirror_bounds=[PCIeCapture.channel_bounds(3)])
        np.testing.assert_array_equal(frame[360:650, :640], frame[360:650, 640:])
        self.assertTrue(np.any(frame[360:650, :640] != 50))
        self.assertTrue(np.all(frame[:360] == 50))


class DirectionDecisionTests(unittest.TestCase):
    def test_onnxruntime_cpu_preserves_fp32_and_disables_telemetry(self):
        from types import SimpleNamespace
        from traffic_rknn import ONNXRuntimeCPUModel

        with tempfile.TemporaryDirectory() as temporary:
            weights = Path(temporary) / "color.onnx"
            weights.with_suffix(".json").write_text(json.dumps({"classes": ["red", "yellow", "green", "off"],
                "input_size": 16, "output_shapes": [[1, 27, 2, 2]]}))
            ort = MagicMock()
            session = ort.InferenceSession.return_value
            session.get_inputs.return_value = [SimpleNamespace(name="images")]
            session.run.return_value = [np.zeros((1, 27, 2, 2), dtype=np.float32)]
            with patch.dict("sys.modules", {"onnxruntime": ort}):
                model = ONNXRuntimeCPUModel(weights)
            tensor = np.full((1, 3, 16, 16), .5, dtype=np.float32)
            self.assertEqual(model.infer(tensor)[0].dtype, np.float32)
            np.testing.assert_array_equal(session.run.call_args[0][1]["images"], tensor)
            self.assertEqual(ort.InferenceSession.call_args[1]["providers"], ["CPUExecutionProvider"])
            ort.disable_telemetry_events.assert_called_once()
            model.close()

    def test_cpu_color_and_npu_direction_entry_does_not_load_pytorch(self):
        from types import SimpleNamespace
        from traffic_light_3way_v2_modified import inference_runtime

        args = SimpleNamespace(weights=Path("color_raw.onnx"), direction_weights=Path("direction.rknn"))
        with patch("traffic_light_3way_v2_modified.load_pc_runtime") as load_pc:
            result = inference_runtime(lambda settings: settings.weights)(args)
        self.assertEqual(result, args.weights)
        load_pc.assert_not_called()

    def test_cpu_color_keeps_fp32_input_and_checks_detection_outputs(self):
        from traffic_rknn import OpenCVDNNModel

        with tempfile.TemporaryDirectory() as temporary:
            weights = Path(temporary) / "color.onnx"
            weights.with_suffix(".json").write_text(json.dumps({"classes": ["red", "yellow", "green", "off"],
                "input_size": 16, "output_shapes": [[1, 27, 2, 2]]}))
            net = MagicMock()
            net.getUnconnectedOutLayersNames.return_value = ("head",)
            net.forward.return_value = [np.zeros((1, 27, 2, 2), dtype=np.float32)]
            with patch("traffic_rknn.cv2.dnn.readNetFromONNX", return_value=net):
                model = OpenCVDNNModel(weights)
            tensor = np.full((1, 3, 16, 16), .5, dtype=np.float32)
            self.assertEqual(model.infer(tensor)[0].shape, (1, 27, 2, 2))
            np.testing.assert_array_equal(net.setInput.call_args[0][0], tensor)
            net.setPreferableTarget.assert_called_once_with(cv2.dnn.DNN_TARGET_CPU)
            net.forward.return_value[0][0, 0, 0, 0] = np.nan
            with self.assertRaisesRegex(RuntimeError, "invalid detection heads"):
                model.infer(tensor)
            model.close()

    def test_board_api_receives_normalized_rgb_pixels_in_nhwc_order(self):
        import ctypes as ct
        from traffic_rknn import Input, RKNNModel

        model = RKNNModel.__new__(RKNNModel)
        model.config = {"input_size": 2}
        model.outputs = [(1, 5)]
        model.context = ct.c_uint64(1)
        model.lib = MagicMock()
        tensor = np.arange(12, dtype=np.float32).reshape(1, 3, 2, 2) / 12
        expected_pixels = np.array([[[[0, 4, 8], [1, 5, 9]],
                                     [[2, 6, 10], [3, 7, 11]]]], dtype=np.float32) / 12
        logits = np.arange(5, dtype=np.float32)

        def check_input(context, count, inputs):
            item = ct.cast(inputs, ct.POINTER(Input)).contents
            self.assertEqual((count, item.type, item.fmt, item.pass_through), (1, 0, 1, 0))
            self.assertEqual(item.size, expected_pixels.nbytes)
            pixels = np.ctypeslib.as_array(ct.cast(item.buf, ct.POINTER(ct.c_float)), (12,))
            np.testing.assert_array_equal(pixels.reshape(1, 2, 2, 3), expected_pixels)
            return 0

        def return_logits(context, count, outputs, extend):
            outputs[0].buf, outputs[0].size = logits.ctypes.data, logits.nbytes
            return 0

        model.lib.rknn_inputs_set.side_effect = check_input
        model.lib.rknn_run.return_value = 0
        model.lib.rknn_outputs_get.side_effect = return_logits
        model.lib.rknn_outputs_release.return_value = 0
        np.testing.assert_array_equal(model.infer(tensor)[0], logits.reshape(1, 5))
        model.lib.rknn_outputs_release.assert_called_once()

        model.config["native_float16_input"] = True

        def check_half_input(context, count, inputs):
            item = ct.cast(inputs, ct.POINTER(Input)).contents
            self.assertEqual((count, item.type, item.fmt, item.pass_through), (1, 1, 1, 1))
            self.assertEqual(item.size, expected_pixels.size * 2)
            pixels = np.ctypeslib.as_array(ct.cast(item.buf, ct.POINTER(ct.c_uint16)), (12,)).view(np.float16)
            np.testing.assert_array_equal(pixels.reshape(1, 2, 2, 3), expected_pixels.astype(np.float16))
            return 0

        model.lib.rknn_inputs_set.side_effect = check_half_input
        np.testing.assert_array_equal(model.infer(tensor)[0], logits.reshape(1, 5))

    def test_native_signed_rgb_input_preserves_all_256_pixel_values(self):
        from traffic_rknn import RKNNModel
        model=RKNNModel.__new__(RKNNModel)
        model.rgb_int8_input=True;model.config={'input_size':16,'input_scale':255}
        model._infer_prepared=MagicMock(return_value=['output'])
        pixels=np.arange(768,dtype=np.uint8).reshape(16,16,3)
        self.assertEqual(model.infer_rgb(pixels),['output'])
        tensor,input_type,pass_through=model._infer_prepared.call_args.args
        expected=np.round((pixels.astype(np.float32)/255)/np.float32(1/255)).astype(np.int16)-128
        np.testing.assert_array_equal(tensor[0],expected.astype(np.int8))
        self.assertTrue(tensor.flags.c_contiguous)
        self.assertEqual((input_type,pass_through),(2,1))

    @unittest.skipUnless(importlib.util.find_spec("utils") is not None, "Requires the local YOLOv5 source/configuration")
    def test_rknn_preprocessing_matches_pc_normalization_and_letterbox(self):
        from utils.augmentations import classify_transforms, letterbox

        frame = np.random.default_rng(1).integers(0, 256, (43, 65, 3), dtype=np.uint8)
        rgb, _, _ = letterbox_rgb(frame, 128)
        expected = letterbox(frame, (128, 128), auto=False)[0][:, :, ::-1]
        np.testing.assert_array_equal(rgb, expected)
        config = {"input_scale": 255, "mean": [.485, .456, .406], "std": [.229, .224, .225]}
        gray = np.asarray(Image.fromarray(rgb).convert("L").convert("RGB"))
        pc = classify_transforms(128)(gray[:, :, ::-1].copy()).numpy()[None]
        np.testing.assert_allclose(normalized_input(gray, config), pc, atol=1e-7)

    def test_rknn_raw_heads_keep_class_and_anchor_coordinates(self):
        metadata = {"classes": ["red", "yellow", "green", "off"], "anchors": [[[10, 20]] * 3], "strides": [8]}
        head = np.full((1, 27, 2, 2), -30, dtype=np.float32)
        head[0, :4, 1, 0] = 0
        head[0, 4, 1, 0] = 10
        head[0, 7, 1, 0] = 10
        rows = decode_heads([head], metadata, .75)
        self.assertEqual(len(rows), 1)
        np.testing.assert_allclose(rows[0][:4], [-1, 2, 9, 22])
        self.assertEqual(rows[0][5], 2)
        self.assertEqual(decode_heads([head], metadata, 1), [])
        self.assertEqual(len(nms(rows + rows, .45)), 1)

    def test_probability_heads_match_raw_logits_without_second_sigmoid(self):
        metadata = {"classes": ["red", "yellow", "green", "off"], "anchors": [[[10, 20]] * 3], "strides": [8]}
        logits = np.full((1, 27, 2, 2), -30, dtype=np.float32)
        logits[0, :4, 1, 0] = [-.5, .2, 1., -.3]
        logits[0, 4, 1, 0] = 5
        logits[0, 7, 1, 0] = 6
        probability = 1 / (1 + np.exp(-logits))
        for threshold in (0, .75, 1):
            expected = decode_heads([logits], metadata, threshold)
            actual = decode_heads([probability], dict(metadata, output_activation="sigmoid"), threshold)
            np.testing.assert_allclose(actual, expected)

    def test_board_cli_does_not_load_pytorch(self):
        import subprocess
        import sys

        process = subprocess.run([sys.executable, "-c", "import sys; import traffic_light_3way_v2_modified; "
                                  "assert 'torch' not in sys.modules"], capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)

    @unittest.skipUnless(importlib.util.find_spec("models.common") is not None, "Requires the local YOLOv5 source/configuration")
    def test_cropped_detection_maps_both_axes_to_video(self):
        model = MagicMock()
        model.device, model.stride, model.fp16 = torch.device("cpu"), 32, False
        settings = MagicMock(imgsz=64, conf_thres=.3, iou_thres=.45)
        boxes = torch.tensor([[4., 8., 20., 32., .9, 1.]])
        region = np.zeros((48, 32, 3), dtype=np.uint8)
        with patch("traffic_light_3way_v2_modified.letterbox", return_value=(region,)), \
             patch("traffic_light_3way_v2_modified.non_max_suppression", return_value=[boxes]), \
             patch("traffic_light_3way_v2_modified.scale_boxes", side_effect=lambda shape, b, original: b):
            detected = detect_roi(model, region, settings, offset_x=40, offset_y=60)
        self.assertEqual(detected[0][:4], [44, 68, 60, 92])
        self.assertEqual(detected[0][5], 1)
        self.assertAlmostEqual(detected[0][4], .9)

    @unittest.skipUnless(importlib.util.find_spec("models.common") is not None, "Requires the local YOLOv5 source/configuration")
    def test_default_classifier_rejects_uncertain_direction(self):
        classes = ["circle", "left", "other", "right", "straight"]
        model = MagicMock()
        model.names, model.pt = dict(enumerate(classes)), True
        model.return_value = torch.tensor([[.85, .0375, .0375, .0375, .0375]]).log()
        with tempfile.TemporaryDirectory() as temporary:
            weights = Path(temporary) / "direction.pt"
            weights.touch()
            weights.with_suffix(".json").write_text(json.dumps({
                "input_size": 128, "classes": classes, "grayscale": True}))
            frame = np.zeros((60, 60, 3), dtype=np.uint8)
            detections = [[10, 10, 30, 45, .9, 0]]
            with patch("traffic_light_3way_v2_modified.DetectMultiBackend", return_value=model):
                conservative = DirectionClassifier(weights, torch.device("cpu"))
                self.assertEqual(conservative.predict(frame, detections)[0][0], "UNKNOWN")
                explicit = DirectionClassifier(weights, torch.device("cpu"), threshold=.8)
                self.assertEqual(explicit.predict(frame, detections)[0][0], "CIRCLE")

    @unittest.skipUnless(importlib.util.find_spec("models.common") is not None, "Requires the local YOLOv5 source/configuration")
    def test_four_view_small_housing_still_requires_confident_direction(self):
        classes = ["circle", "left", "other", "right", "straight"]
        model = MagicMock()
        model.names, model.pt = dict(enumerate(classes)), True
        with tempfile.TemporaryDirectory() as temporary:
            weights = Path(temporary) / "direction.pt"
            weights.touch()
            weights.with_suffix(".json").write_text(json.dumps({
                "input_size": 128, "classes": classes, "grayscale": True}))
            with patch("traffic_light_3way_v2_modified.DetectMultiBackend", return_value=model):
                classifier = DirectionClassifier(weights, torch.device("cpu"))
                frame = np.zeros((60, 60, 3), dtype=np.uint8)
                housing = [[10, 10, 18, 24, .99, 0]]
                for values, expected in (([.01, .01, .01, .01, .96], "STRAIGHT"),
                                         ([.05, .05, .05, .05, .80], "UNKNOWN"),
                                         ([.01, .01, .96, .01, .01], "OTHER"),
                                         ([.15, .15, .40, .15, .15], "UNKNOWN")):
                    model.return_value = torch.tensor([values]).log()
                    self.assertEqual(classifier.predict(frame, housing)[0][0], expected)

    def test_non_signal_candidate_is_removed_before_tracking_or_drawing(self):
        from traffic_light_3way_v2_modified import FrameAnalyzer
        from types import SimpleNamespace
        args=SimpleNamespace(signal_roi=(.2,.2,.8,.8),direction_centers=None,
                             confirm_frames=1,top_ratio=1,tiles=1,iou_thres=.45,
                             circle_direction=None,show=False)
        classifier=MagicMock()
        analyzer=FrameAnalyzer(SimpleNamespace(names={0:'green'}),classifier,None)
        rejected=[80,70,90,90,.9,0];housing=[145,100,160,130,.95,0]
        classifier.predict.side_effect=lambda frame,boxes:[('OTHER',.6) if box==rejected else ('LEFT',.98) for box in boxes]
        with patch('traffic_light_3way_v2_modified.detect_roi',return_value=[rejected,housing]):
            result=analyzer(np.zeros((240,320,3),np.uint8),args,0,0,0,1.)
        self.assertEqual([x['box'] for x in result['lights']],[housing[:4]])
        self.assertEqual(len(analyzer.direction_history.tracks),1)

    def test_blinking_housing_survives_dark_inference_but_never_reuses_old_green(self):
        from traffic_light_3way_v2_modified import DIRECTIONS, FrameAnalyzer, current_observation
        from types import SimpleNamespace
        args=SimpleNamespace(signal_roi=(.2,.2,.8,.8),direction_centers=None,
                             confirm_frames=1,top_ratio=1,tiles=1,iou_thres=.45,
                             circle_direction=None,show=False)
        classifier=MagicMock();classifier.predict.side_effect=lambda frame,boxes:[('LEFT',.98)]*len(boxes)
        analyzer=FrameAnalyzer(SimpleNamespace(names={0:'red'}),classifier,None)
        frame=np.zeros((240,320,3),np.uint8);frame[105:113,147:155]=(0,0,255)
        box=[145,100,160,145,.95,0]
        with patch('traffic_light_3way_v2_modified.detect_roi',side_effect=[[box]]+[[]]*9), \
             patch.object(DirectionHistory,'advance',return_value=args.signal_roi):
            first=analyzer(frame,args,0,0,0,0.)
            self.assertEqual(first['lights'][0]['direction'],'LEFT')
            recovered=analyzer(frame,args,1,0,0,.1)
            self.assertEqual(recovered['lights'][0]['direction_source'],'tracked_model')
            self.assertEqual(recovered['lights'][0]['color'],'red')
            dark_frame=np.zeros_like(frame)
            dark=analyzer(dark_frame,args,2,0,0,.2)
            self.assertEqual(dark['lights'][0]['color'],'UNKNOWN')
            bright=frame.copy();bright[105:113,147:155]=(0,255,0)
            filters={direction:StableState(1) for direction in DIRECTIONS}
            lights,states=current_observation(dark,0,0,.21,1.,original_shape=frame.shape[:2],frame=bright,filters=filters)
            self.assertEqual(lights[0]['color'],'green')
            self.assertEqual(states['LEFT'],'green')
            lights,states=current_observation(dark,0,0,.22,1.,original_shape=frame.shape[:2],frame=dark_frame,filters=filters)
            self.assertEqual(lights[0]['color'],'UNKNOWN')
            self.assertEqual(states['LEFT'],'UNKNOWN')
            for n in range(3,10):expired=analyzer(frame,args,n,0,0,n/10)
            self.assertEqual(expired['lights'],[])

    def test_interrupted_crop_is_rebuilt_before_reuse(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "lamp.png"
            path.write_bytes(b"incomplete PNG")
            save_prepared_image(Image.new("RGB", (128, 128), "red"), path)
            with Image.open(path) as image:
                self.assertEqual(image.size, (128, 128))
                self.assertEqual(image.getpixel((64, 64)), (255, 0, 0))
            self.assertFalse(path.with_suffix(".png.part").exists())

    @unittest.skipUnless((Path(__file__).resolve().parents[1] / "data/hyps/hyp.scratch-low.yaml").is_file(), "Requires the local YOLOv5 source/configuration")
    def test_mixed_dataset_keeps_whole_recording_days_and_off_labels(self):
        import yaml

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            local = root / "direction_artifacts/review_v2"
            local.mkdir(parents=True)
            (local / "candidates.json").write_text("[]")
            (local / "annotations.csv").write_text("id,direction,split,group,reviewer\n")
            public = root / "public"
            atlas = public / "atlas_sparse"
            atlas.mkdir(parents=True)
            selection, labels = [], {}
            for index, (day, original_split) in enumerate((("2024-08-06", "train"), ("2024-08-06", "val"),
                                                          ("2024-08-02", "val"), ("2024-08-26", "test"))):
                timestamp = int(datetime.fromisoformat(day).replace(tzinfo=timezone.utc).timestamp()) + index
                image, label = f"front_tele_{timestamp}.jpg", f"front_tele_{timestamp}.txt"
                Image.new("RGB", (40, 60), (index * 60, 30, 90)).save(atlas / image)
                selection.append({"image": image, "label": label, "split": original_split})
                labels[label] = "0 0.5 0.5 0.5 0.5"
            (atlas / "selection.json").write_text(json.dumps(selection))
            (atlas / "labels.json").write_text(json.dumps(labels))
            (atlas / "ATLAS_classes.yaml").write_text("names:\n  0: off\n")
            (public / "bosch_sparse").mkdir()
            (public / "bosch_sparse/selection.json").write_text("[]")
            (public / "bosch_additional").mkdir()
            (public / "bosch_additional/additional_train.yaml").write_text("[]")
            output = root / "prepared"
            with redirect_stdout(io.StringIO()):
                prepare_generalization(root, public, output)
            frames = json.loads((output / "frame_manifest.json").read_text())
            self.assertEqual([r["split"] for r in frames], ["train", "train", "val", "test"])
            self.assertEqual(frames[0]["group"], frames[1]["group"])
            self.assertTrue(all(r["boxes"][0]["color"] == "off" for r in frames))
            self.assertEqual(yaml.safe_load((output / "detection/data.yaml").read_text())["names"],
                             ["red", "yellow", "green", "off"])

    def test_public_compound_arrows_and_red_yellow_do_not_grant_go(self):
        self.assertEqual(public_semantics("arrow_straight_right_green"), ("green", "other"))
        self.assertEqual(public_semantics("GreenStraightLeft"), ("green", "other"))
        self.assertEqual(public_semantics("arrow_left_red_yellow"), ("red", "left"))
        self.assertEqual(public_semantics("off"), ("off", "other"))
        with self.assertRaises(ValueError):
            public_semantics("unannotated")

    def test_missing_signal_revokes_green(self):
        state = StableState(3)
        for _ in range(3):
            state.update("green")
        self.assertEqual(state.stable, "green")
        self.assertEqual(state.update(None), "UNKNOWN")
        self.assertEqual(state.update("green"), "UNKNOWN")

    def test_red_interrupts_green_without_delay(self):
        state = StableState(3)
        for _ in range(3):
            state.update("green")
        self.assertEqual(state.update("red"), "red")
        self.assertEqual(state.update("green"), "UNKNOWN")
        self.assertEqual(state.update("green"), "UNKNOWN")
        self.assertEqual(state.update("green"), "green")

    def test_round_lamp_requires_explicit_mapping(self):
        lamps = [{"direction": "CIRCLE", "color": "green"}]
        self.assertIsNone(direction_states(lamps)["STRAIGHT"])
        self.assertEqual(direction_states(lamps, "STRAIGHT")["STRAIGHT"], "green")

    def test_conflicting_lamps_cannot_grant_permission(self):
        lamps = [{"direction": "LEFT", "color": "green"}, {"direction": "LEFT", "color": "red"}]
        self.assertIsNone(direction_states(lamps)["LEFT"])

    def test_strong_green_cannot_override_another_color(self):
        for conflicting in ("red", "yellow", "off"):
            with self.subTest(conflicting=conflicting):
                lamps = [{"direction": "STRAIGHT", "color": "green", "color_conf": .99},
                         {"direction": "STRAIGHT", "color": conflicting, "color_conf": .6}]
                state = StableState()
                for _ in range(6):
                    signal = state.update(direction_states(lamps)["STRAIGHT"])
                    self.assertEqual(make_decision("STRAIGHT", {"STRAIGHT": signal}, True), "WAIT_UNKNOWN")

    def test_other_signal_group_is_excluded(self):
        lamps = [{"direction": "LEFT", "color": "green", "in_signal_group": False}]
        self.assertIsNone(direction_states(lamps)["LEFT"])

    def test_calibration_and_moving_state(self):
        self.assertEqual(make_decision("LEFT", {"LEFT": "green"}), "WAIT_CALIBRATE_ROI")
        self.assertEqual(make_decision("LEFT", {"LEFT": "red"}, True, True), "ALARM_RED")
        self.assertEqual(make_decision("RIGHT", {"LEFT": "green"}, True), "WAIT_UNKNOWN")

    def test_crop_keeps_arrow_at_top_of_tall_housing(self):
        image = Image.new("RGB", (20, 100), "black")
        image.putpixel((10, 0), (255, 0, 0))
        padded = square_crop(image, (0, 0, 20, 100), size=100)
        self.assertEqual(padded.getpixel((50, 0)), (255, 0, 0))

    def test_invalid_roi_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_roi([0.8, 0, 0.2, 1])

    def test_color_fusion_rejects_weak_misclassification(self):
        lamps = [{"direction": "LEFT", "color": "red", "color_conf": .9},
                 {"direction": "LEFT", "color": "red", "color_conf": .75},
                 {"direction": "LEFT", "color": "green", "color_conf": .4}]
        self.assertEqual(direction_states(lamps)["LEFT"], "red")

    def test_off_lamp_is_not_a_green_observation(self):
        self.assertEqual(direction_states([{"direction": "LEFT", "color": "off"}])["LEFT"], "off")

    def test_group_is_filtered_before_classification(self):
        inside = [40, 40, 60, 60, .9, 0]
        outside = [0, 0, 10, 10, .99, 0]
        self.assertEqual(filter_signal_group([inside, outside], (100, 100), [.3, .3, .7, .7]), [inside])

    def test_original_calibrated_centers_assign_each_lamp(self):
        for center, expected in ((.36, "LEFT"), (.53, "STRAIGHT"), (.68, "RIGHT")):
            box = [center * 100 - 2, 10, center * 100 + 2, 30]
            self.assertEqual(assign_calibrated_direction(box, 100, [.22, .5, .78]), expected)

    def test_calibration_rejects_distant_lamp(self):
        self.assertEqual(assign_calibrated_direction([0, 10, 2, 30], 100, [.4, .5, .6]), "UNKNOWN")

    def test_group_selection_uses_preview_coordinates(self):
        frame = np.zeros((1920, 1080, 3), dtype=np.uint8)
        with patch("cv2.selectROI", return_value=(68, 240, 135, 120)) as select, patch("cv2.destroyWindow"), patch("cv2.namedWindow"):
            roi = select_signal_group(frame)
        self.assertEqual(select.call_args.args[1].shape[:2], (600, 338))
        np.testing.assert_allclose(roi, [68 / 338, .4, 203 / 338, .6])

    def test_cancelled_group_selection_does_not_select_full_frame(self):
        with patch("cv2.selectROI", return_value=(0, 0, 0, 0)), patch("cv2.destroyWindow"), patch("cv2.namedWindow"):
            self.assertIsNone(select_signal_group(np.zeros((720, 540, 3), dtype=np.uint8)))

    def test_video_picker_returns_selected_path(self):
        source = str(Path(__file__).resolve().parents[1] / "test_media" / "4.mp4")
        with patch("sys.argv", ["traffic", "--choose-video", "--pick-signal-group", "--show"]), \
             patch("tkinter.Tk") as window, patch("tkinter.filedialog.askopenfilename", return_value=source):
            args = parse_opt()
        self.assertEqual(args.source, source)
        self.assertTrue(args.pick_signal_group)
        self.assertIsNone(args.signal_roi)
        self.assertIsNone(args.direction_centers)
        window.return_value.destroy.assert_called_once()

    def test_cancelled_video_picker_exits_without_processing_default_video(self):
        with patch("sys.argv", ["traffic", "--choose-video"]), patch("tkinter.Tk"), \
             patch("tkinter.filedialog.askopenfilename", return_value=""):
            with self.assertRaises(SystemExit) as result:
                parse_opt()
        self.assertEqual(result.exception.code, 0)


class DirectionHistoryTests(unittest.TestCase):
    def test_consecutive_alignment_reuses_features_without_losing_housing_identity(self):
        gray=cv2.GaussianBlur(np.random.default_rng(16).integers(0,256,(240,320),np.uint8),(3,3),0)
        history=DirectionHistory(2)
        box=[80,80,120,140,.95,0]
        for _ in range(2):history.update([box],[('LEFT',.97)])
        history.advance(gray.shape,gray,[(0,gray)])
        frames=[(0,gray)]+[(i,cv2.warpAffine(gray,np.float32([[1,0,2*i],[0,1,0]]),(320,240))) for i in range(1,8)]
        with patch.object(cv2,'goodFeaturesToTrack',wraps=cv2.goodFeaturesToTrack) as features:
            history.advance(gray.shape,frames[-1][1],frames)
            self.assertEqual(features.call_count,1)
        self.assertEqual(history.tracks[0]['stable'],'LEFT')
        np.testing.assert_allclose(history.tracks[0]['box'],[94,80,134,140],atol=1.2)

    def setUp(self):
        self.box = [10, 10, 30, 70, .9, 0]
        self.history = DirectionHistory()

    def observe(self, direction):
        return self.history.update([self.box], [(direction, .95)])[0][0]

    def test_single_wrong_direction_does_not_assign_identity(self):
        self.assertEqual(self.observe("RIGHT"), "UNKNOWN")
        self.observe("LEFT")
        self.observe("LEFT")
        self.assertEqual(self.observe("LEFT"), "LEFT")

    def test_switch_requires_three_consistent_observations(self):
        for _ in range(3):
            self.observe("LEFT")
        self.assertEqual(self.observe("RIGHT"), "LEFT")
        self.assertEqual(self.observe("RIGHT"), "LEFT")
        self.assertEqual(self.observe("RIGHT"), "RIGHT")

    def test_missing_detections_never_emit_old_lamps(self):
        for _ in range(3):
            self.observe("LEFT")
        self.assertEqual(self.history.update([], []), [])
        for _ in range(8):
            self.history.update([], [])
        self.assertEqual(self.observe("LEFT"), "UNKNOWN")

    def test_uncertain_identity_is_eventually_revoked(self):
        for _ in range(3):
            self.observe("LEFT")
        self.observe("UNKNOWN")
        self.observe("UNKNOWN")
        self.assertEqual(self.observe("UNKNOWN"), "UNKNOWN")

    def test_reordered_boxes_keep_separate_identities(self):
        right = [80, 10, 100, 70, .9, 0]
        for _ in range(3):
            self.history.update([self.box, right], [("LEFT", .95), ("RIGHT", .95)])
        result = self.history.update([right, self.box], [("RIGHT", .95), ("LEFT", .95)])
        self.assertEqual([r[0] for r in result], ["RIGHT", "LEFT"])

    def test_camera_translation_does_not_match_one_housing_to_another_old_position(self):
        image = cv2.GaussianBlur(np.random.default_rng(32).integers(0, 256, (240, 320, 3), dtype=np.uint8), (3, 3), 0)
        boxes = [[60, 80, 72, 140, .95, 0], [76, 80, 88, 140, .95, 0]]
        history = DirectionHistory(2)
        for _ in range(2):
            history.update(boxes, [("LEFT", .96), ("RIGHT", .97)], image)
        moved = cv2.warpAffine(image, np.float32([[1, 0, 16], [0, 1, 8]]), (320, 240))
        shifted = [[b[0]+16, b[1]+8, b[2]+16, b[3]+8, *b[4:]] for b in boxes]
        result = history.update(shifted, [("UNKNOWN", .3), ("UNKNOWN", .3)], moved)
        self.assertEqual([item[0] for item in result], ["LEFT", "RIGHT"])

    def test_camera_cut_discards_direction_identity_instead_of_matching_unrelated_pixels(self):
        history = DirectionHistory(2)
        image = np.random.default_rng(2).integers(0, 256, (240, 320, 3), dtype=np.uint8)
        for _ in range(2):
            history.update([self.box], [("LEFT", .96)], image)
        other_scene = np.random.default_rng(3).integers(0, 256, image.shape, dtype=np.uint8)
        result = history.update([self.box], [("UNKNOWN", .3)], other_scene)
        self.assertEqual(result[0][0], "UNKNOWN")

    def test_direction_identity_follows_intermediate_frames_during_a_fast_pan(self):
        image = cv2.GaussianBlur(np.random.default_rng(32).integers(0, 256, (240, 320, 3), dtype=np.uint8), (3, 3), 0)
        history = DirectionHistory(2)
        boxes = [[60, 80, 72, 140, .95, 0], [76, 80, 88, 140, .95, 0]]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        frames = [(0, gray), (1, gray)]
        history.update(boxes, [("LEFT", .96), ("RIGHT", .97)], image, frames[:1])
        history.update(boxes, [("LEFT", .96), ("RIGHT", .97)], image, frames)
        for index in range(2, 12):
            moved = cv2.warpAffine(image, np.float32([[1, 0, (index-1)*4], [0, 1, 0]]), (320, 240))
            frames.append((index, cv2.cvtColor(moved, cv2.COLOR_BGR2GRAY)))
        shifted = [[b[0]+40, b[1], b[2]+40, b[3], *b[4:]] for b in boxes]
        motions = {(index-1,index):np.array([[1,0,4],[0,1,0],[0,0,1]], dtype=float) for index in range(2,12)}
        with patch("traffic_light_3way_v2_modified.align_light_boxes") as align:
            result = history.update(shifted, [("UNKNOWN", .3), ("UNKNOWN", .3)], moved, frames, motions)
            align.assert_not_called()
        self.assertEqual([item[0] for item in result], ["LEFT", "RIGHT"])


class VehicleSimulationTests(unittest.TestCase):
    def test_red_transition_key_press_uses_current_stop_state_and_alarms_in_same_update(self):
        car = VehicleSimulation()
        car.update({"STRAIGHT": "green"}, True)
        self.assertTrue(car.moving)
        self.assertEqual(car.update({"STRAIGHT": "red"}, True, "TOGGLE"), "ALARM_RED")
        self.assertTrue(car.manual_moving)
        self.assertTrue(car.alarm)
        self.assertFalse(car.guidance_visible)
        frame = np.zeros((200, 600, 3), np.uint8)
        draw_overlay(frame, [], {"STRAIGHT": "red"}, car, (0, 0, 1, 1))
        np.testing.assert_array_equal(frame[3, 300], [0, 0, 255])
        self.assertEqual(car.update({"STRAIGHT": "red"}, True, "TOGGLE"), "STOP_RED")
        self.assertFalse(car.alarm)
        frame[:] = 0
        draw_overlay(frame, [], {"STRAIGHT": "red"}, car, (0, 0, 1, 1))
        np.testing.assert_array_equal(frame[3, 300], [0, 0, 0])

    def test_green_transition_key_press_stops_current_movement(self):
        car = VehicleSimulation()
        car.update({"STRAIGHT": "red"}, True)
        self.assertEqual(car.update({"STRAIGHT": "green"}, True, "TOGGLE"), "STOP_MANUAL")
        self.assertFalse(car.manual_moving)
        self.assertFalse(car.guidance_visible)

    def test_unknown_after_green_does_not_turn_a_move_press_into_stop_or_false_red_alarm(self):
        car = VehicleSimulation()
        car.update({"STRAIGHT": "green"}, True)
        self.assertEqual(car.update({"STRAIGHT": "UNKNOWN"}, True, "TOGGLE"), "WAIT_UNKNOWN")
        self.assertTrue(car.manual_moving)
        self.assertFalse(car.alarm)
        self.assertEqual(car.update({"STRAIGHT": "red"}, True), "ALARM_RED")
        self.assertEqual(car.update({"STRAIGHT": "red"}, True, "AUTO"), "STOP_RED")
        self.assertFalse(car.alarm)

    def test_motion_buttons_show_manual_state_without_an_extra_status_panel(self):
        car = VehicleSimulation()
        for action, selected in [("TOGGLE", "MOVE"), ("TOGGLE", "STOP"), ("AUTO", "AUTO")]:
            car.update({"STRAIGHT": "red"}, True, action)
            frame = np.zeros((200, 600, 3), np.uint8)
            draw_overlay(frame, [], {"STRAIGHT": "red"}, car, (0, 0, 1, 1))
            for name, (x1, y1, _, _) in control_rectangles(frame.shape):
                if name in ("MOVE", "STOP", "AUTO"):
                    expected = [110, 78, 34] if name == selected else [64, 60, 55]
                    np.testing.assert_array_equal(frame[y1 + 10, x1 + 10], expected)

    def test_one_mixed_signal_group_is_reused_for_every_navigation_direction(self):
        lamps = [{"direction": "LEFT", "color": "red"}, {"direction": "STRAIGHT", "color": "green"},
                 {"direction": "RIGHT", "color": "yellow"}]
        observations = direction_states(lamps)
        filters = {direction: StableState() for direction in observations}
        for _ in range(3):
            states = {direction: filters[direction].update(color) for direction, color in observations.items()}
        car = VehicleSimulation()
        for direction, decision, allowed in (("LEFT", "STOP_RED", False), ("STRAIGHT", "GO_STRAIGHT", True),
                                             ("RIGHT", "WAIT_YELLOW", False)):
            car.command(direction)
            self.assertEqual(car.update(states, True), decision)
            self.assertEqual(car.moving, allowed)
            self.assertEqual(car.guidance_visible, allowed)
        self.assertEqual(states, {"LEFT": "red", "STRAIGHT": "green", "RIGHT": "yellow"})

    def test_guidance_arrow_is_hidden_without_permission_to_move(self):
        cases = [("red", None, True), ("yellow", None, True), ("UNKNOWN", None, True),
                 ("green", "STOP", True), ("red", "MOVE", True), ("yellow", "MOVE", True),
                 ("UNKNOWN", "MOVE", True), ("green", "MOVE", False)]
        for signal, action, calibrated in cases:
            with self.subTest(signal=signal, action=action, calibrated=calibrated):
                car = VehicleSimulation()
                if action:
                    car.command(action)
                car.update({"STRAIGHT": signal}, calibrated)
                self.assertFalse(car.snapshot()["guidance_visible"])
                frame = np.full((720, 540, 3), 100, dtype=np.uint8)
                before = frame.copy()
                draw_driving_arrow(frame, car)
                np.testing.assert_array_equal(frame, before)

    def test_green_guidance_projects_three_distinct_solid_arrows(self):
        from traffic_light_3way_v2_modified import CONTROL_PANEL_HEIGHT

        frames = []
        for direction in ("LEFT", "STRAIGHT", "RIGHT"):
            car = VehicleSimulation(direction)
            car.update({direction: "green"}, True)
            self.assertTrue(car.snapshot()["guidance_visible"])
            frame = np.full((720, 540, 3), 100, dtype=np.uint8)
            draw_driving_arrow(frame, car)
            self.assertGreater(np.count_nonzero(np.any(frame != 100, axis=2)), 1000)
            self.assertTrue(np.any(np.all(frame == (70, 240, 130), axis=2)))
            np.testing.assert_array_equal(frame[:124], 100)
            np.testing.assert_array_equal(frame[-CONTROL_PANEL_HEIGHT:], 100)
            frames.append(frame)
        for a, b in ((0, 1), (1, 2), (0, 2)):
            self.assertFalse(np.array_equal(frames[a], frames[b]))

    def test_auto_red_stop_green_start(self):
        car = VehicleSimulation()
        self.assertEqual(car.update({"STRAIGHT": "red"}, True), "STOP_RED")
        self.assertFalse(car.moving)
        self.assertEqual(car.update({"STRAIGHT": "green"}, True), "GO_STRAIGHT")
        self.assertTrue(car.moving)
        self.assertEqual(car.update({"STRAIGHT": "yellow"}, True), "WAIT_YELLOW")
        self.assertFalse(car.moving)

    def test_force_red_motion_then_stop_clears_alarm(self):
        car = VehicleSimulation("LEFT")
        car.command("MOVE")
        self.assertEqual(car.update({"LEFT": "red"}, True), "ALARM_RED")
        self.assertEqual(car.snapshot()["simulated_leds"], [1, 0, 0, 1])
        car.command("STOP")
        self.assertEqual(car.update({"LEFT": "red"}, True), "STOP_RED")
        self.assertEqual(car.snapshot()["simulated_leds"], [0, 0, 0, 0])

    def test_manual_stop_is_held_on_green_until_auto(self):
        car = VehicleSimulation()
        car.command("STOP")
        self.assertEqual(car.update({"STRAIGHT": "green"}, True), "STOP_MANUAL")
        self.assertFalse(car.moving)
        car.command("AUTO")
        self.assertEqual(car.update({"STRAIGHT": "green"}, True), "GO_STRAIGHT")
        self.assertEqual(car.mode, "AUTO")

    def test_lane_change_uses_new_lamp_and_one_direction_led(self):
        car = VehicleSimulation("RIGHT")
        lamps = {"LEFT": "red", "STRAIGHT": "green", "RIGHT": "green"}
        car.update(lamps, True)
        self.assertEqual(car.snapshot()["simulated_leds"], [0, 0, 1, 0])
        car.command("LEFT")
        self.assertEqual(car.update(lamps, True), "STOP_RED")
        self.assertFalse(car.moving)

    def test_auto_cannot_move_on_missing_or_uncalibrated_signal(self):
        car = VehicleSimulation()
        car.update({"STRAIGHT": "green"}, True)
        self.assertEqual(car.update({}, True), "WAIT_UNKNOWN")
        self.assertFalse(car.moving)
        self.assertEqual(car.update({"STRAIGHT": "green"}, False), "WAIT_CALIBRATE_ROI")
        self.assertFalse(car.moving)

    def test_space_toggles_actual_motion(self):
        car = VehicleSimulation()
        car.update({"STRAIGHT": "green"}, True)
        car.command("TOGGLE")
        car.update({"STRAIGHT": "green"}, True)
        self.assertFalse(car.moving)
        car.command("TOGGLE")
        car.update({"STRAIGHT": "red"}, True)
        self.assertTrue(car.alarm)

    def test_mouse_buttons_match_scaled_preview(self):
        car = VehicleSimulation()
        source, preview = (1920, 1080), (900, 506)
        for action, (x1, y1, x2, y2) in control_rectangles(source):
            x = int((x1 + x2) / 2 * preview[1] / source[1])
            y = int((y1 + y2) / 2 * preview[0] / source[0])
            handle_mouse(cv2.EVENT_LBUTTONDOWN, x, y, 0, (car, source, preview))
            if action in ("LEFT", "STRAIGHT", "RIGHT"):
                self.assertEqual(car.nav, action)
            elif action == "AUTO":
                self.assertIsNone(car.manual_moving)
            else:
                self.assertEqual(car.manual_moving, action == "MOVE")

    def test_mouse_outside_buttons_does_not_change_controls(self):
        car = VehicleSimulation()
        handle_mouse(cv2.EVENT_LBUTTONDOWN, 10, 10, 0, (car, (1920, 1080), (900, 506)))
        self.assertEqual(car.nav, "STRAIGHT")
        self.assertEqual(car.mode, "AUTO")


class PlaybackAndFPGAInteractionTests(unittest.TestCase):
    def test_exact_frame_cache_preserves_every_decoded_pixel_and_frame(self):
        from traffic_light_3way_v2_modified import playback_preview

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "original.avi"
            frames = np.random.default_rng(71).integers(0, 256, (4, 48, 64, 3), dtype=np.uint8)
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"FFV1"), 30, (64, 48))
            self.assertTrue(writer.isOpened())
            for frame in frames:
                writer.write(frame)
            writer.release()
            original = cv2.VideoCapture(str(source))
            cached = playback_preview(original, source, root / "cache", 30, (48, 64), 64, 48, lossless=True)
            try:
                for expected in frames:
                    ok, actual = cached.read()
                    self.assertTrue(ok)
                    np.testing.assert_array_equal(actual, expected)
                self.assertFalse(cached.read()[0])
                self.assertTrue(cached.set(cv2.CAP_PROP_POS_FRAMES, 2))
                np.testing.assert_array_equal(cached.read()[1], frames[2])
            finally:
                original.release()
                cached.release()

            original = cv2.VideoCapture(str(source))
            cached = playback_preview(original, source, root / "cache", 30, (48, 64), 64, 48,
                                      lossless=True, bounds=(16, 12, 48, 40), stride=2)
            try:
                self.assertEqual(int(cached.get(cv2.CAP_PROP_FRAME_COUNT)), 2)
                for index in (0, 2):
                    np.testing.assert_array_equal(cached.read()[1], frames[index, 12:40, 16:48])
                self.assertTrue(cached.set(cv2.CAP_PROP_POS_FRAMES, 1))
                np.testing.assert_array_equal(cached.read()[1], frames[2, 12:40, 16:48])
            finally:
                original.release()
                cached.release()

    def test_playback_copy_keeps_every_frame_and_reuses_cache_without_modifying_source(self):
        import hashlib
        from traffic_light_3way_v2_modified import playback_preview
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.avi"
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 10, (480, 800))
            self.assertTrue(writer.isOpened())
            for value in range(8):
                writer.write(np.full((800, 480, 3), value * 25, dtype=np.uint8))
            writer.release()
            before = hashlib.sha256(source.read_bytes()).digest()
            original = cv2.VideoCapture(str(source))
            preview = playback_preview(original, source, root / "cache", 10, (800, 480), 640, 600)
            try:
                self.assertEqual(int(preview.get(cv2.CAP_PROP_FRAME_COUNT)), 8)
                seen = []
                while True:
                    ok, frame = preview.read()
                    if not ok:
                        break
                    self.assertEqual(frame.shape[:2], (600, 360))
                    seen.append(float(frame.mean()))
                self.assertEqual(len(seen), 8)
                np.testing.assert_allclose(seen, np.arange(8) * 25, atol=6)
            finally:
                original.release()
                preview.release()
            original = MagicMock()
            preview = playback_preview(original, source, root / "cache", 10, (800, 480), 640, 600)
            preview.release()
            original.read.assert_not_called()
            self.assertEqual(hashlib.sha256(source.read_bytes()).digest(), before)

    def test_worker_seeks_original_frame_and_preserves_full_resolution_for_detection(self):
        from types import SimpleNamespace
        from traffic_light_3way_v2_modified import FrameAnalyzer
        cap = MagicMock()
        cap.read.return_value = (True, np.zeros((1920, 1080, 3), dtype=np.uint8))
        classifier = MagicMock()
        classifier.predict.return_value = []
        args = SimpleNamespace(confirm_frames=1, top_ratio=1, signal_roi=None, tiles=1, show=True,
                               iou_thres=.45, direction_centers=None, circle_direction=None)
        with patch("traffic_light_3way_v2_modified.detect_roi", return_value=[]) as detect:
            result = FrameAnalyzer(MagicMock(), classifier, None, cap)(None, args, 71, 0, 0, 1)
        cap.set.assert_called_once_with(cv2.CAP_PROP_POS_FRAMES, 71)
        self.assertEqual(detect.call_args.args[1].shape[:2], (1920, 1080))
        self.assertEqual(result["frame"], 71)
        self.assertIs(result["image"], cap.read.return_value[1])

    def test_detected_red_light_is_drawn_while_direction_is_pending(self):
        from traffic_light_3way_v2_modified import draw_overlay
        frame = np.zeros((600, 640, 3), dtype=np.uint8)
        light = {"box": [100, 180, 140, 250], "color": "red", "color_conf": .95,
                 "direction": "UNKNOWN", "direction_conf": 0, "in_signal_group": True}
        car = VehicleSimulation()
        car.update({}, True)
        draw_overlay(frame, [light], {}, car, None)
        np.testing.assert_array_equal(frame[200, 100], [20, 20, 255])
        self.assertFalse(car.guidance_visible)
        self.assertFalse(car.moving)

    def test_sparse_group_cache_uses_exact_frame_pixels_and_new_groups_use_original(self):
        from types import SimpleNamespace
        from traffic_light_3way_v2_modified import FrameAnalyzer

        image = np.random.default_rng(9).integers(0, 256, (240, 320, 3), dtype=np.uint8)
        cached, original, classifier = MagicMock(), MagicMock(), MagicMock()
        cached.get.return_value = 0
        cached.read.return_value = (True, image[32:224, 48:288].copy())
        original.read.return_value = (True, image)
        classifier.predict.return_value = []
        args = SimpleNamespace(confirm_frames=1, top_ratio=1, signal_roi=[.35, .4, .65, .6],
                               tiles=1, show=True, iou_thres=.45, direction_centers=None, circle_direction=None)
        analyzer = FrameAnalyzer(MagicMock(), classifier, None, cached, (48, 32, 288, 224), (240, 320), 3, original)
        with patch("traffic_light_3way_v2_modified.detect_roi", return_value=[]) as detect:
            result = analyzer(None, args, 6, 0, 0, 2)
            cached.set.assert_not_called()
            self.assertEqual(cached.grab.call_count, 2)
            np.testing.assert_array_equal(detect.call_args.args[1], image[32:208, 48:272])
            self.assertEqual((result["frame"], result["captured_at"]), (6, 2))
            args.signal_roi = [.2, .25, .5, .75]
            result = analyzer(None, args, 9, 0, 1, 3)
            original.set.assert_called_once_with(cv2.CAP_PROP_POS_FRAMES, 9)
            np.testing.assert_array_equal(detect.call_args.args[1], image[:, :224])
            self.assertIs(result["image"], image)

    def test_cache_forward_advance_and_loop_seek_preserve_source_pixels(self):
        from types import SimpleNamespace
        from traffic_light_3way_v2_modified import FrameAnalyzer, playback_preview

        frames = np.random.default_rng(12).integers(0, 256, (18, 240, 320, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.avi"
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"HFYU"), 30, (320, 240))
            self.assertTrue(writer.isOpened())
            for frame in frames:
                writer.write(frame)
            writer.release()
            original = cv2.VideoCapture(str(source))
            cached = playback_preview(original, source, Path(temporary), 30, (240, 320), 320, 240,
                                      lossless=True, bounds=(48, 32, 288, 224), stride=3)
            classifier = MagicMock()
            classifier.predict.return_value = []
            analyzer = FrameAnalyzer(MagicMock(), classifier, None, cached, (48, 32, 288, 224), (240, 320), 3, original)
            args = SimpleNamespace(confirm_frames=2, top_ratio=1, signal_roi=[.35, .4, .65, .6],
                                   tiles=1, show=True, iou_thres=.45, direction_centers=None, circle_direction=None)
            try:
                with patch("traffic_light_3way_v2_modified.detect_roi", return_value=[]) as detect:
                    for index in (6, 15, 0, 3):
                        result = analyzer(None, args, index, int(index == 0), 0, 2)
                        np.testing.assert_array_equal(detect.call_args.args[1], frames[index, 32:208, 48:272])
                        self.assertEqual(result["frame"], index)
            finally:
                cached.release()
                original.release()

    def test_clean_overlay_preserves_video_above_bottom_controls(self):
        from traffic_light_3way_v2_modified import CONTROL_PANEL_HEIGHT, draw_overlay

        frame = np.random.default_rng(14).integers(0, 256, (600, 640, 3), dtype=np.uint8)
        original = frame.copy()
        with patch("traffic_light_3way_v2_modified.cv2.putText"):
            draw_overlay(frame, [], {}, VehicleSimulation(), None)
        np.testing.assert_array_equal(frame[:-CONTROL_PANEL_HEIGHT], original[:-CONTROL_PANEL_HEIGHT])

    def test_stale_model_picture_shows_actual_boxes_without_granting_live_permission(self):
        from traffic_light_3way_v2_modified import current_observation, draw_model_observation
        image = np.zeros((600, 800, 3), dtype=np.uint8)
        image[:, :, 0] = 60
        result = {"frame": 20, "loop": 0, "group_revision": 0, "captured_at": 10,
                  "signal_roi": [.25, .2, .75, .8], "image": image, "processing_ms": 18000,
                  "lights": [{"box": [250, 170, 310, 250], "color": "red", "color_conf": .95,
                              "direction": "UNKNOWN", "direction_conf": 0}],
                  "states": {"STRAIGHT": "green"}}
        panel = draw_model_observation(result, 384, 600, 28, 1.5)
        self.assertEqual(panel.shape, (600, 384, 3))
        self.assertGreater(np.count_nonzero(np.all(panel == [20, 20, 255], axis=2)), 30)
        lights, states = current_observation(result, 0, 0, 28, 1.5)
        self.assertEqual(lights, [])
        car = VehicleSimulation()
        car.update(states, True)
        self.assertFalse(car.guidance_visible)
        self.assertFalse(car.moving)

    def test_portrait_canvas_keeps_controls_readable_and_transforms_the_signal_region(self):
        from traffic_light_3way_v2_modified import display_canvas
        frame = np.zeros((1920, 1080, 3), dtype=np.uint8)
        lights = [{"box": [108, 192, 540, 960], "color": "green"}]
        canvas, scaled, roi = display_canvas(frame, lights, [.1, .1, .5, .5])
        self.assertEqual(canvas.shape[:2], (600, 640))
        self.assertEqual(scaled[0]["box"], [185, 60, 320, 300])
        np.testing.assert_allclose(roi, [(.1 * 338 + 151) / 640, .1, .5, .5])

    def test_preview_fits_both_landscape_and_portrait_without_upscaling(self):
        from traffic_light_3way_v2_modified import preview_dimensions
        self.assertEqual(preview_dimensions((1920, 1080)), (338, 600))
        self.assertEqual(preview_dimensions((720, 1280)), (1024, 576))
        self.assertEqual(preview_dimensions((360, 640)), (640, 360))

    def test_inference_replaces_pending_frames_instead_of_building_a_backlog(self):
        from threading import Event
        from traffic_light_3way_v2_modified import LatestFrameInference
        started, release, completed = Event(), Event(), Event()
        seen = []
        def analyze(value):
            seen.append(value)
            if value == 1:
                started.set()
                self.assertTrue(release.wait(2))
            if value == 3:
                completed.set()
            return value
        worker = LatestFrameInference(analyze)
        try:
            worker.submit((1,))
            self.assertTrue(started.wait(2))
            worker.submit((2,))
            worker.submit((3,))
            release.set()
            self.assertTrue(completed.wait(2))
        finally:
            release.set()
            worker.close()
        self.assertEqual(seen, [1, 3])
        self.assertEqual(worker.take(), 3)

    def test_worker_error_is_reported_to_the_main_thread(self):
        from traffic_light_3way_v2_modified import LatestFrameInference
        def broken():
            raise ValueError("bad inference output")
        worker = LatestFrameInference(broken)
        worker.submit(())
        worker.thread.join(2)
        self.assertFalse(worker.thread.is_alive())
        try:
            with self.assertRaisesRegex(RuntimeError, "inference failed") as caught:
                worker.take()
            self.assertIsInstance(caught.exception.__cause__, ValueError)
        finally:
            worker.close()

    def test_old_or_previous_video_group_green_never_grants_permission(self):
        from traffic_light_3way_v2_modified import current_observation
        result = {"loop": 0, "group_revision": 0, "captured_at": 10,
                  "lights": [{"color": "green"}], "states": {"STRAIGHT": "green"}}
        for loop, revision, now in ((0, 0, 12), (1, 0, 10.1), (0, 1, 10.1)):
            lights, states = current_observation(result, loop, revision, now, 1.5)
            car = VehicleSimulation()
            car.update(states, True)
            self.assertFalse(car.guidance_visible)
            self.assertFalse(car.moving)
            self.assertEqual(lights, [])
        self.assertEqual(current_observation(result, 0, 0, 10.2, 1.5)[1], result["states"])

    def test_light_cell_change_preserves_stationary_housing_boxes(self):
        from traffic_light_3way_v2_modified import align_light_boxes

        source = np.full((240, 360), 40, np.uint8)
        lights = []
        for direction, x in zip(("LEFT", "STRAIGHT", "RIGHT"), (90, 155, 220)):
            lights.append({"box": [x, 90, x + 32, 165], "direction": direction, "color": "red"})
            cv2.rectangle(source, (x, 90), (x + 32, 165), 15, -1)
            for px in range(x - 20, x + 53, 9):
                for py in (65, 75, 175, 185):
                    cv2.rectangle(source, (px, py), (px + 3, py + 3), 80, -1)
            for px in (x + 6, x + 13, x + 20):
                for py in (98, 105, 112):
                    cv2.rectangle(source, (px, py), (px + 3, py + 3), 255, -1)
        current = source.copy()
        for light in lights:
            x, y, _, _ = light["box"]
            current[y + 1:y + 74, x + 1:x + 31] = 15
            for px in (x + 6, x + 13, x + 20):
                for py in (132, 139, 146):
                    cv2.rectangle(current, (px, py), (px + 3, py + 3), 255, -1)
        aligned = align_light_boxes(source, current, lights, source.shape)
        self.assertEqual(len(aligned), 3)
        for before, after in zip(lights, aligned):
            np.testing.assert_allclose(after["box"], before["box"], atol=1)

    def test_delayed_boxes_follow_camera_translation_in_current_video_pixels(self):
        from traffic_light_3way_v2_modified import align_light_boxes

        source = cv2.GaussianBlur(np.random.default_rng(16).integers(0, 256, (240, 320), dtype=np.uint8), (3, 3), 0)
        current = cv2.warpAffine(source, np.float32([[1, 0, 16], [0, 1, 9]]), (320, 240))
        light = {"box": [180, 160, 240, 260], "direction": "STRAIGHT", "color": "green", "color_conf": .95}
        aligned = align_light_boxes(source, current, [light], (480, 640))
        self.assertEqual(len(aligned), 1)
        np.testing.assert_allclose(aligned[0]["box"], [212, 178, 272, 278], atol=1.2)
        self.assertEqual(light["box"], [180, 160, 240, 260])
        self.assertEqual(aligned[0]["color_conf"], .95)

    def test_delayed_boxes_follow_camera_zoom(self):
        from traffic_light_3way_v2_modified import align_light_boxes

        source = cv2.GaussianBlur(np.random.default_rng(18).integers(0, 256, (240, 320), dtype=np.uint8), (3, 3), 0)
        transform = cv2.getRotationMatrix2D((160, 120), 0, 1.08)
        current = cv2.warpAffine(source, transform, (320, 240))
        light = {"box": [120, 80, 155, 135], "direction": "LEFT", "color": "red", "color_conf": .95}
        aligned = align_light_boxes(source, current, [light], (240, 320))
        self.assertEqual(len(aligned), 1)
        expected = np.array([[120, 80, 1], [155, 135, 1]]) @ transform.T
        np.testing.assert_allclose(aligned[0]["box"], expected.ravel(), atol=1.0)

    def test_alignment_reuses_source_features_but_tracks_each_current_frame(self):
        from traffic_light_3way_v2_modified import align_light_boxes

        source = cv2.GaussianBlur(np.random.default_rng(16).integers(0, 256, (240, 320), dtype=np.uint8), (3, 3), 0)
        light = {"box": [180, 160, 240, 260], "direction": "STRAIGHT", "color": "green"}
        cache = {}
        with patch.object(cv2, "goodFeaturesToTrack", wraps=cv2.goodFeaturesToTrack) as features:
            for shift in (16, 18):
                current = cv2.warpAffine(source, np.float32([[1, 0, shift], [0, 1, 9]]), (320, 240))
                aligned = align_light_boxes(source, current, [light], (480, 640), cache)
                self.assertEqual(len(aligned), 1)
                np.testing.assert_allclose(aligned[0]["box"], [180 + shift * 2, 178, 240 + shift * 2, 278], atol=1.2)
            self.assertEqual(features.call_count, 1)
            align_light_boxes(source.copy(), current, [light], (480, 640), cache)
            self.assertEqual(features.call_count, 2)
        self.assertEqual(len(cache), 3)
        from traffic_light_3way_v2_modified import warp_light_boxes
        np.testing.assert_allclose(warp_light_boxes([light], cache["motion"], (480,640))[0]["box"], aligned[0]["box"])

    def test_incremental_alignment_reuses_valid_tracked_features(self):
        from traffic_light_3way_v2_modified import align_light_boxes

        previous = cv2.GaussianBlur(np.random.default_rng(16).integers(0,256,(240,320),dtype=np.uint8),(3,3),0)
        image = previous.copy()
        lights = [{"box":[80,80,120,140],"direction":"STRAIGHT","color":"red"}]
        cache = {}
        with patch.object(cv2,"goodFeaturesToTrack",wraps=cv2.goodFeaturesToTrack) as features:
            for shift in range(2,12,2):
                current = cv2.warpAffine(image,np.float32([[1,0,shift],[0,1,0]]),(320,240))
                lights = align_light_boxes(previous,current,lights,(240,320),cache)
                self.assertEqual(len(lights),1)
                np.testing.assert_allclose(lights[0]["box"],[80+shift,80,120+shift,140],atol=1.2)
                previous = current
            self.assertEqual(features.call_count,1)

    def test_delayed_live_observation_aligns_boxes_without_refreshing_model_age(self):
        from traffic_light_3way_v2_modified import current_observation

        source = cv2.GaussianBlur(np.random.default_rng(21).integers(0, 256, (240, 320), dtype=np.uint8), (3, 3), 0)
        current = cv2.warpAffine(source, np.float32([[1, 0, 12], [0, 1, -7]]), (320, 240))
        light = {"box": [120, 80, 155, 135], "direction": "STRAIGHT", "color": "green", "color_conf": .95}
        result = {"frame": 0, "loop": 0, "group_revision": 0, "captured_at": 10,
                  "lights": [light], "states": {"STRAIGHT": "green"}}
        frames = [(0, 0, 0, source), (20, 0, 0, current)]
        lights, states = current_observation(result, 0, 0, 10.8, 1.5, frames, (240, 320))
        np.testing.assert_allclose(lights[0]["box"], [132, 73, 167, 128], atol=1.2)
        self.assertEqual(states["STRAIGHT"], "green")
        lights, states = current_observation(result, 0, 0, 11.6, 1.5, frames, (240, 320))
        self.assertEqual(lights, [])
        self.assertEqual(states["STRAIGHT"], "UNKNOWN")
        self.assertEqual(result["captured_at"], 10)

    def test_live_boxes_advance_from_previous_display_frame_and_still_expire(self):
        from traffic_light_3way_v2_modified import current_observation

        gray = np.zeros((120, 160), np.uint8)
        light = {"box": [20, 30, 40, 80], "direction": "STRAIGHT", "color": "red", "color_conf": .95}
        result = {"frame": 0, "loop": 0, "group_revision": 0, "captured_at": 10,
                  "lights": [light], "states": {"STRAIGHT": "red"}}
        gray1, gray2 = gray.copy(), gray.copy()
        with patch("traffic_light_3way_v2_modified.align_light_boxes", return_value=[light]) as align:
            current_observation(result, 0, 0, 10.7, 1.5, [(0, 0, 0, gray), (18, 0, 0, gray1)], (120, 160))
            current_observation(result, 0, 0, 10.8, 1.5, [(0, 0, 0, gray), (18, 0, 0, gray1), (19, 0, 0, gray2)], (120, 160))
            self.assertIs(align.call_args.args[0], gray1)
            self.assertIs(align.call_args.args[1], gray2)
            lights, states = current_observation(result, 0, 0, 11.6, 1.5,
                                                [(18, 0, 0, gray1), (19, 0, 0, gray2)], (120, 160))
            self.assertEqual(lights, [])
            self.assertEqual(states["STRAIGHT"], "UNKNOWN")

    def test_new_model_boxes_use_cached_motion_between_capture_and_display(self):
        from traffic_light_3way_v2_modified import current_observation

        frames = [(index,0,0,np.zeros((120,160),np.uint8)) for index in range(11)]
        motions = {(index-1,index,0,0):np.array([[1,0,4],[0,1,0],[0,0,1]],dtype=float) for index in range(1,11)}
        light = {"box":[20,30,40,80],"direction":"STRAIGHT","color":"red","color_conf":.95}
        result = {"frame":0,"loop":0,"group_revision":0,"captured_at":10,
                  "lights":[light],"states":{"STRAIGHT":"red"}}
        with patch("traffic_light_3way_v2_modified.align_light_boxes") as align:
            lights, states = current_observation(result,0,0,10.8,1.5,frames,(120,160),motions=motions)
            align.assert_not_called()
        np.testing.assert_allclose(lights[0]["box"],[60,30,80,80])
        self.assertEqual(states["STRAIGHT"],"red")

    def test_lost_motion_or_missing_source_frame_revokes_old_green(self):
        from traffic_light_3way_v2_modified import current_observation

        source = cv2.GaussianBlur(np.random.default_rng(23).integers(0, 256, (240, 320), dtype=np.uint8), (3, 3), 0)
        current = np.zeros_like(source)
        result = {"frame": 0, "loop": 0, "group_revision": 0, "captured_at": 10,
                  "lights": [{"box": [120, 80, 155, 135], "direction": "STRAIGHT", "color": "green", "color_conf": .95}],
                  "states": {"STRAIGHT": "green"}}
        for frames in ([(0, 0, 0, source), (20, 0, 0, current)], [(20, 0, 0, source)]):
            lights, states = current_observation(result, 0, 0, 10.8, 1.5, frames, (240, 320))
            self.assertEqual(lights, [])
            self.assertEqual(states["STRAIGHT"], "UNKNOWN")
            car = VehicleSimulation()
            car.update(states, True)
            self.assertFalse(car.guidance_visible)

    def test_aligned_circle_keeps_explicit_round_lamp_mapping(self):
        from traffic_light_3way_v2_modified import current_observation

        gray = np.zeros((240, 320), dtype=np.uint8)
        result = {"frame": 7, "loop": 0, "group_revision": 0, "captured_at": 10,
                  "lights": [{"box": [120, 80, 155, 135], "direction": "CIRCLE", "color": "green", "color_conf": .95}],
                  "states": {"STRAIGHT": "green"}}
        lights, states = current_observation(result, 0, 0, 10.1, 1.5, [(7, 0, 0, gray)], (240, 320), "STRAIGHT")
        self.assertEqual(states["STRAIGHT"], "green")
        self.assertEqual(len(lights), 1)

    def test_current_pixels_change_labels_and_decisions_without_waiting_for_a_new_model_result(self):
        from traffic_light_3way_v2_modified import current_observation

        light = {"box": [30, 20, 60, 100], "direction": "STRAIGHT", "color": "red", "color_conf": .95}
        result = {"frame": 0, "loop": 0, "group_revision": 0, "captured_at": 10,
                  "lights": [light], "states": {"STRAIGHT": "red"}}
        filters = {direction: StableState(2) for direction in ("LEFT", "STRAIGHT", "RIGHT")}
        car = VehicleSimulation()
        for color, pixels, expected in (("green", (120, 240, 5), "WAIT_UNKNOWN"),
                                        ("green", (120, 240, 5), "GO_STRAIGHT"),
                                        ("red", (5, 5, 240), "STOP_RED"),
                                        ("yellow", (5, 230, 240), "WAIT_YELLOW"),
                                        ("UNKNOWN", (30, 30, 30), "WAIT_UNKNOWN")):
            frame = np.full((120, 100, 3), 30, np.uint8)
            cv2.arrowedLine(frame, (40, 65), (52, 45), pixels, 5, tipLength=.6)
            lights, states = current_observation(result, 0, 0, 10.8, 1.5, original_shape=(120, 100),
                                                 frame=frame, filters=filters)
            self.assertEqual(lights[0]["color"], color)
            self.assertEqual(car.update(states, True), expected)
        self.assertEqual(result["lights"][0]["color"], "red")

    def test_live_colors_ignore_blue_dim_reflections_and_isolated_noise(self):
        from traffic_light_3way_v2_modified import observe_light_color

        for pixels in ((255, 20, 5), (140, 170, 20), (30, 30, 30)):
            frame = np.full((80, 40, 3), pixels, np.uint8)
            self.assertEqual(observe_light_color(frame, [0, 0, 40, 80])[0], "UNKNOWN")
        frame = np.full((80, 40, 3), 30, np.uint8)
        frame[5, 10] = frame[30, 15] = frame[60, 20] = (0, 255, 0)
        self.assertEqual(observe_light_color(frame, [0, 0, 40, 80])[0], "UNKNOWN")

    @unittest.skipUnless(all((Path(__file__).parent / "fixtures" / name).is_file() for name in ("off_housing_reflection.png", "overexposed_amber.png")), "Requires the original camera test fixtures")
    def test_real_off_housing_reflection_and_overexposed_amber_from_another_video(self):
        from traffic_light_3way_v2_modified import observe_light_color

        for name, expected in (("off_housing_reflection.png", "UNKNOWN"), ("overexposed_amber.png", "yellow")):
            frame = cv2.imread(str(Path(__file__).parent / "fixtures" / name))
            self.assertIsNotNone(frame)
            self.assertEqual(observe_light_color(frame, [0, 0, frame.shape[1], frame.shape[0]])[0], expected)

    @unittest.skipUnless((Path(__file__).parent / "fixtures/compressed_amber.png").is_file(), "Requires the original camera test fixtures")
    def test_board_compressed_amber_and_red_middle_cell_remain_distinct(self):
        from traffic_light_3way_v2_modified import observe_light_color

        frame = cv2.imread(str(Path(__file__).parent / "fixtures" / "compressed_amber.png"))
        self.assertIsNotNone(frame)
        self.assertEqual(observe_light_color(frame, [0, 0, frame.shape[1], frame.shape[0]])[0], "yellow")
        for pixels in ((0, 0, 240), (10, 90, 240)):
            frame = np.full((80, 30, 3), 30, np.uint8)
            cv2.circle(frame, (15, 40), 5, pixels, -1)
            self.assertEqual(observe_light_color(frame, [0, 0, 30, 80])[0], "red")

    @unittest.skipUnless(all((Path(__file__).parent / "fixtures" / f"board_compressed_red_{i}.png").is_file() for i in (293, 296, 297)), "Requires the original camera test fixtures")
    def test_board_compressed_red_arrows_keep_their_connected_dim_stem(self):
        from traffic_light_3way_v2_modified import observe_light_color

        for index in (293, 296, 297):
            with self.subTest(frame=index):
                frame = cv2.imread(str(Path(__file__).parent / "fixtures" / f"board_compressed_red_{index}.png"))
                self.assertIsNotNone(frame)
                self.assertEqual(observe_light_color(frame, [0, 0, frame.shape[1], frame.shape[0]])[0], "red")

    @unittest.skipUnless(all((Path(__file__).parent / "fixtures" / f"board_small_signal_{i}.png").is_file() for i in range(8)), "Requires the original camera test fixtures")
    def test_live_small_signal_pixels_remain_distinct_from_dim_reflection_and_single_seed(self):
        from traffic_light_3way_v2_modified import observe_light_color

        for index in range(8):
            with self.subTest(index=index):
                frame = cv2.imread(str(Path(__file__).parent / 'fixtures' / f'board_small_signal_{index}.png'))
                self.assertIsNotNone(frame)
                expected = 'red' if index < 3 else 'yellow'
                self.assertEqual(observe_light_color(frame, [0, 0, frame.shape[1], frame.shape[0]])[0], expected)
        for color in ((0, 0, 150), (20, 170, 140), (140, 170, 20)):
            frame = np.full((14, 6, 3), color, np.uint8)
            self.assertEqual(observe_light_color(frame, [0, 0, 6, 14])[0], 'UNKNOWN')
        frame = np.full((14, 6, 3), 30, np.uint8)
        frame[3, 2], frame[3, 3] = (0, 255, 0), (0, 160, 0)
        self.assertEqual(observe_light_color(frame, [0, 0, 6, 14])[0], 'UNKNOWN')
        frame[3, 3] = (0, 255, 0)
        self.assertEqual(observe_light_color(frame, [0, 0, 6, 14])[0], 'UNKNOWN')

    def test_slow_housing_refresh_still_reads_current_red_and_expires_at_the_limit(self):
        from traffic_light_3way_v2_modified import current_observation

        frame = np.full((120, 100, 3), 30, np.uint8)
        cv2.circle(frame, (45, 65), 7, (0, 0, 240), -1)
        result = {"frame": 0, "loop": 0, "group_revision": 0, "captured_at": 10,
                  "lights": [{"box": [30, 20, 60, 100], "direction": "STRAIGHT", "color": "green", "color_conf": .95}]}
        lights, states = current_observation(result, 0, 0, 12, 2.5, original_shape=frame.shape[:2], frame=frame)
        self.assertEqual(lights[0]["color"], "red")
        self.assertEqual(states["STRAIGHT"], "red")
        lights, states = current_observation(result, 0, 0, 12.6, 2.5, original_shape=frame.shape[:2], frame=frame)
        self.assertFalse(lights)
        self.assertTrue(all(state == "UNKNOWN" for state in states.values()))

    def test_conflicting_current_colors_and_invalid_crops_never_reuse_old_green(self):
        from traffic_light_3way_v2_modified import observe_light_color

        frame = np.full((80, 40, 3), 30, np.uint8)
        cv2.rectangle(frame, (10, 10), (25, 25), (0, 0, 240), -1)
        cv2.rectangle(frame, (10, 50), (25, 65), (0, 240, 0), -1)
        for box in ([0, 0, 40, 80], [-50, -50, -5, -5], [50, 90, 60, 100], [0, 0, 2, 2]):
            self.assertEqual(observe_light_color(frame, box)[0], "UNKNOWN")

    def test_current_color_coordinates_scale_and_loop_or_group_change_clear_green_confirmation(self):
        from traffic_light_3way_v2_modified import current_observation

        frame = np.full((120, 100, 3), 30, np.uint8)
        cv2.circle(frame, (45, 65), 7, (0, 240, 0), -1)
        result = {"frame": 0, "loop": 0, "group_revision": 0, "captured_at": 10,
                  "lights": [{"box": [60, 40, 120, 200], "direction": "STRAIGHT", "color": "red", "color_conf": .95}],
                  "states": {"STRAIGHT": "red"}}
        filters = {direction: StableState(2) for direction in ("LEFT", "STRAIGHT", "RIGHT")}
        for _ in range(2):
            lights, states = current_observation(result, 0, 0, 10.8, 1.5, original_shape=(240, 200), frame=frame, filters=filters)
        self.assertEqual(lights[0]["color"], "green")
        self.assertEqual(states["STRAIGHT"], "green")
        lights, states = current_observation(result, 1, 0, 10.9, 1.5, original_shape=(240, 200), frame=frame, filters=filters)
        self.assertEqual(lights, [])
        self.assertEqual(states["STRAIGHT"], "UNKNOWN")
        self.assertEqual(filters["STRAIGHT"].stable, "UNKNOWN")
        result["loop"] = 1
        _, states = current_observation(result, 1, 0, 11, 1.5, original_shape=(240, 200), frame=frame, filters=filters)
        self.assertEqual(states["STRAIGHT"], "UNKNOWN")
        current_observation(result, 1, 1, 11.1, 1.5, original_shape=(240, 200), frame=frame, filters=filters)
        self.assertEqual(filters["STRAIGHT"].stable, "UNKNOWN")

    def test_incompatible_fpga_signature_is_rejected_before_any_register_write(self):
        from traffic_fpga import FPGAInteraction
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            endpoint = root / "endpoint"
            endpoint.mkdir()
            (endpoint / "vendor").write_text("0x0755\n")
            (endpoint / "device").write_text("0x0755\n")
            (endpoint / "resource0").write_bytes(bytes(0x2000))
            original = bytes([91]) * 0x1000
            (endpoint / "resource1").write_bytes(original)
            with patch("traffic_fpga.Path") as path_factory:
                path_factory.return_value.__truediv__.return_value = endpoint
                with self.assertRaisesRegex(RuntimeError, "signature missing"):
                    FPGAInteraction("0002:21:00.0")
            self.assertEqual((endpoint / "resource1").read_bytes(), original)

    def test_fpga_controls_and_output_bits_match_the_rtl_and_clear_on_exit(self):
        import struct
        import mmap
        from types import SimpleNamespace
        from traffic_fpga import FPGAInteraction, MAGIC, OFFSET, SEQUENCE_OFFSET
        linux_mmap = SimpleNamespace(ACCESS_READ=mmap.ACCESS_READ, MAP_SHARED=1, PROT_READ=1, PROT_WRITE=2,
            mmap=lambda fd, length, **options: mmap.mmap(fd, length, access=options.get("access", mmap.ACCESS_WRITE)))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            endpoint = root / "endpoint"
            endpoint.mkdir()
            (endpoint / "vendor").write_text("0x0755\n")
            (endpoint / "device").write_text("0x0755\n")
            data = bytearray(0x2000)
            struct.pack_into("<4I", data, OFFSET, MAGIC, 1, 17, 1 << 8)
            (endpoint / "resource0").write_bytes(data)
            (endpoint / "resource1").write_bytes(bytes(0x1000))
            with patch("traffic_fpga.Path") as path_factory, patch("traffic_fpga.mmap", linux_mmap), \
                    patch("traffic_fpga.os.O_SYNC", 0, create=True):
                path_factory.return_value.__truediv__.return_value = endpoint
                bridge = FPGAInteraction("0002:21:00.0")
            try:
                self.assertEqual(bridge.poll(), {"navigation": "STRAIGHT", "manual_moving": None})
                self.assertIsNone(bridge.poll())
                car = VehicleSimulation("LEFT", True)
                car.update({"LEFT": "red"}, True)
                bridge.publish(car, True)
                word = struct.unpack_from("<I", (endpoint / "resource1").read_bytes(), OFFSET)[0]
                self.assertEqual(word, 0xA55A01FC)
                sequence = struct.unpack_from("<I", (endpoint / "resource1").read_bytes(), SEQUENCE_OFFSET)[0]
                self.assertEqual(sequence, 17)
                bridge.publish(car, False)
                word = struct.unpack_from("<I", (endpoint / "resource1").read_bytes(), OFFSET)[0]
                self.assertFalse(word & (1 << 6))
            finally:
                bridge.close()
            word = struct.unpack_from("<I", (endpoint / "resource1").read_bytes(), OFFSET)[0]
            self.assertEqual(word, 0xA55A0100)

    def test_fpga_bridge_requires_a_control_snapshot_before_publishing(self):
        from traffic_fpga import FPGAInteraction
        bridge = FPGAInteraction.__new__(FPGAInteraction)
        bridge.last_control = None
        with self.assertRaisesRegex(RuntimeError, "Read the FPGA controls"):
            bridge.publish(VehicleSimulation(), True)

    def test_transition_key_alarm_border_and_fpga_alarm_bit_use_the_same_decision(self):
        from types import SimpleNamespace
        from traffic_fpga import FPGAInteraction

        bridge = FPGAInteraction.__new__(FPGAInteraction)
        bridge.last_control, bridge.hardware_control, bridge.source, bridge.sequence = (1, 0), 1, 0, 0
        bridge.read_map = bytearray(0x2000)
        bridge.sequence_word = SimpleNamespace(value=0)
        bridge.command_word = SimpleNamespace(value=0)
        car = VehicleSimulation()
        car.update({"STRAIGHT": "green"}, True)
        for expected_bits in (0x70, 0x40):
            car.update({"STRAIGHT": "red"}, True, "TOGGLE")
            bridge.publish(car, True)
            self.assertEqual(bridge.command_word.value & 0x70, expected_bits)
            frame = np.zeros((200, 600, 3), np.uint8)
            draw_overlay(frame, [], {"STRAIGHT": "red"}, car, (0, 0, 1, 1))
            self.assertEqual(bool(np.array_equal(frame[3, 300], [0, 0, 255])), bool(expected_bits & 0x20))

    def test_fpga_poll_retries_a_control_snapshot_torn_by_a_key_event(self):
        from traffic_fpga import FPGAInteraction, MAGIC
        bridge = FPGAInteraction.__new__(FPGAInteraction)
        bridge.read_map = None
        bridge.last_control = None
        reads = [(6,), (MAGIC, 1, 7, 0), (7,), (7,), (MAGIC, 14, 7, 0), (7,)]
        with patch('traffic_fpga.struct.unpack_from', side_effect=reads):
            self.assertEqual(bridge.poll(), {'navigation': 'RIGHT', 'manual_moving': True})
        self.assertEqual(bridge.sequence, 7)

    def test_fpga_poll_refuses_continuously_changing_control_snapshots(self):
        from traffic_fpga import FPGAInteraction, MAGIC
        bridge = FPGAInteraction.__new__(FPGAInteraction)
        bridge.read_map = None
        bridge.last_control = None
        with patch('traffic_fpga.struct.unpack_from', side_effect=[(1,), (MAGIC, 1, 2, 0), (2,)] * 4):
            with self.assertRaisesRegex(RuntimeError, 'changed repeatedly'):
                bridge.poll()

    def test_fpga_preflight_reads_status_without_opening_the_write_bar(self):
        import struct
        from check_fpga import inspect_fpga
        from traffic_fpga import MAGIC, OFFSET
        with tempfile.TemporaryDirectory() as temporary:
            endpoint = Path(temporary) / 'endpoint'
            endpoint.mkdir()
            (endpoint / 'vendor').write_text('0x0755\n')
            (endpoint / 'device').write_text('0x0755\n')
            data = bytearray(0x2000)
            struct.pack_into('<4I', data, OFFSET, MAGIC, 14, 3, 0)
            (endpoint / 'resource0').write_bytes(data)
            with patch('check_fpga.Path') as path_factory:
                path_factory.return_value.glob.return_value = [endpoint]
                report = inspect_fpga()[0]
            self.assertTrue(report['ready'])
            self.assertEqual(report['navigation'], 'RIGHT')
            self.assertTrue(report['manual_moving'])
            self.assertFalse((endpoint / 'resource1').exists())

    def test_fpga_preflight_reports_old_firmware_and_inaccessible_devices(self):
        import struct
        from check_fpga import inspect_fpga
        from traffic_fpga import OFFSET
        with tempfile.TemporaryDirectory() as temporary:
            endpoint = Path(temporary)
            (endpoint / 'vendor').write_text('0x0755\n')
            (endpoint / 'device').write_text('0x0755\n')
            data = bytearray(0x2000)
            struct.pack_into('<I', data, OFFSET, 0x54524631)
            (endpoint / 'resource0').write_bytes(data)
            with patch('check_fpga.Path') as path_factory:
                path_factory.return_value.glob.return_value = [endpoint]
                self.assertFalse(inspect_fpga()[0]['ready'])
                (endpoint / 'resource0').unlink()
                self.assertFalse(inspect_fpga()[0]['ready'])

    def test_lane_selects_its_own_light_and_manual_motion_controls_alarm(self):
        states = {"LEFT": "red", "STRAIGHT": "green", "RIGHT": "yellow"}
        car = VehicleSimulation("STRAIGHT", None)
        car.update(states, True)
        self.assertTrue(car.guidance_visible)
        car.command("LEFT")
        car.update(states, True)
        self.assertFalse(car.moving)
        self.assertFalse(car.alarm)
        car.manual_moving = True
        car.update(states, True)
        self.assertTrue(car.moving)
        self.assertTrue(car.alarm)
        self.assertFalse(car.guidance_visible)
        car.command("STRAIGHT")
        car.manual_moving = False
        car.update(states, True)
        self.assertFalse(car.alarm)
        self.assertFalse(car.guidance_visible)
        car.manual_moving = None
        car.update({"STRAIGHT": "UNKNOWN"}, True)
        self.assertFalse(car.moving)
        self.assertFalse(car.guidance_visible)


class CPUMotionKeyTests(unittest.TestCase):
    def setUp(self):
        from traffic_fpga import CPUMotionKey
        self.key = CPUMotionKey.__new__(CPUMotionKey)
        self.key.view = bytearray(4096)
        self.key.observed = self.key.stable = 1
        self.key.changed_at = 0
        self.key.pressed_at = None

    def sample(self, level, now):
        import struct
        struct.pack_into("<I", self.key.view, 0x70, level << 8)
        return self.key.poll(now)

    def test_short_press_toggles_only_once_after_release(self):
        for level, now in [(0, .01), (0, .04), (0, .4), (1, .41)]:
            self.assertIsNone(self.sample(level, now))
        self.assertEqual(self.sample(1, .44), "TOGGLE")
        self.assertIsNone(self.sample(1, 1))

    def test_hold_returns_auto_on_release(self):
        for level, now in [(0, .01), (0, .04), (0, 3.2), (1, 3.3)]:
            self.assertIsNone(self.sample(level, now))
        self.assertEqual(self.sample(1, 3.34), "AUTO")

    def test_bounce_and_key_held_at_start_do_not_toggle(self):
        for level, now in [(0, .01), (1, .02), (1, .05)]:
            self.assertIsNone(self.sample(level, now))
        self.key.observed = self.key.stable = 0
        self.assertIsNone(self.sample(1, 1))
        self.assertIsNone(self.sample(1, 1.1))


if __name__ == "__main__":
    unittest.main()
