# 姓名：丁黄炎
"""Native camera tests run on RK3568; PC regression reports an explicit skip."""
import ctypes as ct
import ast
import io
import os
from contextlib import redirect_stderr
from pathlib import Path
import unittest
from unittest.mock import Mock,patch
from types import SimpleNamespace
import cv2
import numpy as np

LIBRARY=Path(os.environ.get('TRAFFIC_CAMERA_TEST_LIBRARY',
    str(Path(__file__).resolve().parents[1]/'native_camera/libcamera_threads.so')))
CAPTURE_LIBRARY=Path(os.environ.get('TRAFFIC_CAPTURE_TEST_LIBRARY',
    str(Path(__file__).resolve().parents[1]/'native_capture/libtraffic_capture.so')))


@unittest.skipUnless(os.name=='posix' and LIBRARY.is_file(),'RK3568 camera native library required')
class CameraMotionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.library=ct.CDLL(str(LIBRARY))
        cls.library.traffic_camera_configure_threads.argtypes=[]
        cls.library.traffic_camera_configure_threads.restype=ct.c_int
        assert cls.library.traffic_camera_configure_threads()==1
        cls.move=cls.library.traffic_camera_move_people
        cls.move.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_int,
            ct.c_void_p,ct.c_int,ct.c_int,ct.c_double,ct.c_int,ct.c_int,ct.c_void_p]
        cls.move.restype=ct.c_int
        cls.match=cls.library.traffic_camera_match_people
        cls.match.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_int,
            ct.c_void_p,ct.c_int,ct.c_int,ct.c_double,ct.c_int,ct.c_int,ct.c_void_p]
        cls.match.restype=ct.c_int

    def move_box(self,before,after,dx,dy,fraction=1):
        flow=np.zeros((*before.shape,2),np.float32);flow[:]=[dx*.3,dy*.3]
        people=np.array([[80,40,120,120,.93,0]],np.float64)
        output=np.empty_like(people)
        count=self.move(before.ctypes.data,after.ctypes.data,flow.ctypes.data,160,90,
            people.ctypes.data,1,6,fraction,320,180,output.ctypes.data)
        self.assertGreaterEqual(count,0)
        return output[:count]

    def setUp(self):
        self.texture=np.random.default_rng(17).integers(80,220,(40,20),dtype=np.uint8)
        self.before=np.full((90,160),30,np.uint8)
        self.before[20:60,40:60]=self.texture

    def test_current_image_corrects_underestimated_motion_without_a_later_detection(self):
        for dx in (-10,-6,-2,0,2,6,10):
            for dy in (-4,0,4):
                with self.subTest(dx=dx,dy=dy):
                    after=np.full_like(self.before,30)
                    after[20+dy:60+dy,40+dx:60+dx]=self.texture
                    result=self.move_box(self.before,after,dx,dy)
                    self.assertEqual(len(result),1)
                    np.testing.assert_allclose(result[0,:4],np.array([80,40,120,120])+[2*dx,2*dy]*2,atol=1)
                    np.testing.assert_allclose(result[0,4:],[.93,0])

    def test_disappeared_person_is_cleared_and_partial_time_motion_is_not_overcorrected(self):
        self.assertEqual(len(self.move_box(self.before,np.full_like(self.before,30),0,0)),0)
        result=self.move_box(self.before,self.before,0,0,.5)
        np.testing.assert_allclose(result[0,:4],[80,40,120,120])

    def match_box(self, after, fraction=1):
        people=np.array([[80,40,120,120,.93,0]],np.float64)
        output=np.empty_like(people)
        count=self.match(self.before.ctypes.data,after.ctypes.data,160,90,
            people.ctypes.data,1,6,fraction,320,180,output.ctypes.data)
        return count,output

    def test_fast_match_tracks_translation_and_preserves_metadata(self):
        for dx in (-10,-2,0,2,10):
            after=np.full_like(self.before,30)
            after[20:60,40+dx:60+dx]=self.texture
            count,output=self.match_box(after)
            self.assertEqual(count,1)
            np.testing.assert_allclose(output[0,:4],np.array([80,40,120,120])+[2*dx,0]*2)
            np.testing.assert_allclose(output[0,4:],[.93,0])

    def test_missing_ambiguous_and_partial_time_matches_require_dense_flow(self):
        self.assertEqual(self.match_box(np.full_like(self.before,30))[0],-2)
        ambiguous=np.full_like(self.before,30)
        ambiguous[20:60,28:48]=self.texture
        ambiguous[20:60,52:72]=self.texture
        self.assertEqual(self.match_box(ambiguous)[0],-2)
        self.assertEqual(self.match_box(self.before,.5)[0],-2)

    def test_signal_fast_match_requires_near_exact_pixels_and_falls_back_for_changed_scene(self):
        from traffic_rknn import NativeSignals
        library=ct.CDLL(str(CAPTURE_LIBRARY))
        signals=NativeSignals(library,camera=True)
        before=np.random.default_rng(71).integers(0,256,(180,320),dtype=np.uint8)
        after=cv2.warpAffine(before,np.float32([[1,0,3],[0,1,-2]]),(320,180))
        lights=[{'box':[80,50,100,95],'direction':'LEFT','color':'red'},
                {'box':[180,55,200,100],'direction':'STRAIGHT','color':'green'}]
        cache={}
        actual=signals.align(before,after,lights,(180,320),cache)
        self.assertEqual(len(actual),2)
        for original,moved in zip(lights,actual):
            np.testing.assert_allclose(moved['box'],np.array(original['box'])+[3,-2,3,-2],atol=.01)
            self.assertEqual(moved['direction'],original['direction'])
        np.testing.assert_allclose(cache['motion'][:2,2],[3,-2],atol=.01)
        other=np.random.default_rng(72).integers(0,256,before.shape,dtype=np.uint8)
        self.assertEqual(signals.align(before,other,lights,(180,320),{}),[])


class CameraTrackingStateTests(unittest.TestCase):
    def tracker(self, native=True):
        from traffic_camera import CameraPersonTracker
        from traffic_pedestrian import PersonTracker
        tracker=CameraPersonTracker.__new__(CameraPersonTracker)
        PersonTracker.__init__(tracker)
        if native:
            def gray(previous,pixels,width,height,stride,output,flow):
                image=np.ctypeslib.as_array((ct.c_uint8*(stride*height)).from_address(pixels))
                image=image.reshape(height,stride)[:,:width*3].reshape(height,width,3)
                converted=cv2.resize(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),(width//2,height//2))
                ct.memmove(output,converted.ctypes.data,converted.nbytes)
                return 0
            tracker.native=SimpleNamespace(traffic_gray_flow=gray)
            def dense(previous,current,width,height,output):
                ct.memset(output,0,width*height*2*ct.sizeof(ct.c_float))
                return 1
            tracker.camera=SimpleNamespace(traffic_camera_dense_flow=Mock(side_effect=dense))
        tracker.flow=Mock()
        tracker.flow.calc.return_value=np.zeros((90,160,2),np.float32)
        tracker.match_native=Mock()
        tracker.move_native=Mock()
        return tracker

    def detection(self, frame=0, people=None):
        return dict(frame=frame,captured_at=0,people=people if people is not None else
                    [[80,40,120,120,.93,0]],shape=(180,320))

    def test_delayed_detection_replays_every_frame_without_dense_flow_for_verified_matches(self):
        tracker=self.tracker()
        image=np.full((180,320,3),30,np.uint8)
        for number in range(3):tracker.update(image,None,number)
        moved=[[84,40,124,120,.93,0]]
        tracker.match_native.return_value=moved
        result=tracker.update(image,self.detection(),3)
        self.assertEqual(result['people'],moved)
        self.assertEqual(tracker.match_native.call_count,3)
        self.assertTrue(all(call.args[3]==1 for call in tracker.match_native.call_args_list))
        tracker.flow.calc.assert_not_called()
        tracker.move_native.assert_not_called()

    def test_ambiguous_match_falls_back_and_an_outer_scene_reset_rejects_old_detections(self):
        tracker=self.tracker()
        image=np.full((180,320,3),30,np.uint8)
        tracker.update(image,None,0)
        tracker.match_native.return_value=None
        tracker.move_native.return_value=[]
        self.assertTrue(tracker.update(image,self.detection(),1)['tracking_lost'])
        tracker.camera.traffic_camera_dense_flow.assert_called_once()
        tracker.flow.calc.assert_not_called()
        tracker.move_native.assert_called_once()
        tracker=self.tracker()
        tracker.cut_frame=2
        stale=dict(self.detection(),frame=20,source_frame=0)
        self.assertIsNone(tracker.update(np.full_like(image,220),stale,2))
        self.assertEqual(tracker.cut_frame,2)
        fresh=dict(self.detection(frame=3),source_frame=2)
        self.assertEqual(tracker.update(np.full_like(image,220),fresh,3)['source_frame'],2)

    def test_empty_detection_needs_no_matching_or_flow_and_native_is_optional(self):
        tracker=self.tracker(native=False)
        image=np.full((180,320,3),30,np.uint8)
        tracker.update(image,None,0)
        self.assertEqual(tracker.update(image,self.detection(people=[]),1)['people'],[])
        tracker.match_native.assert_not_called()
        tracker.flow.calc.assert_not_called()
        tracker.match_native.return_value=None
        tracker.move_native.return_value=self.detection()['people']
        tracker.update(image,self.detection(frame=1),2)
        tracker.flow.calc.assert_called_once()

    def test_identical_grayscale_keeps_boxes_without_native_matching(self):
        tracker=self.tracker()
        del tracker.match_native
        tracker.camera=SimpleNamespace(traffic_camera_match_people=Mock())
        previous=np.random.default_rng(8).integers(0,256,(90,160),dtype=np.uint8)
        people=self.detection()['people']
        self.assertEqual(tracker.match_native(previous,previous.copy(),people,.5,(180,320)),people)
        tracker.camera.traffic_camera_match_people.assert_not_called()

    def test_joint_hdmi_fallback_uses_original_native_mover_without_camera_refinement(self):
        tracker=self.tracker()
        tracker.camera_refinement=False
        tracker.camera.traffic_camera_move_people=Mock()
        def move(previous,current,flow,width,height,boxes,count,columns,fraction,frame_width,frame_height,output):
            ct.memmove(output,boxes,count*columns*ct.sizeof(ct.c_double))
            return count
        tracker.native.traffic_move_people=Mock(side_effect=move)
        del tracker.move_native
        gray=np.zeros((90,160),np.uint8)
        people=self.detection()['people']
        self.assertEqual(tracker.move_native(gray,gray,np.zeros((90,160,2),np.float32),people,1,(180,320)),people)
        tracker.native.traffic_move_people.assert_called_once()
        tracker.camera.traffic_camera_move_people.assert_not_called()

    def test_tracker_factory_changes_joint_native_hdmi_only(self):
        import traffic_light_3way_v2_modified as app
        tree=ast.parse(Path(app.__file__).read_text(encoding='utf-8'))
        run=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='run')
        start=next(index for index,node in enumerate(run.body) if isinstance(node,ast.Assign) and
                   any(isinstance(target,ast.Name) and target.id=='camera_person_tracker' for target in node.targets))
        end=next(index for index,node in enumerate(run.body[start:],start) if isinstance(node,ast.Assign) and
                 any(isinstance(target,ast.Name) and target.id=='pair_person_trackers' for target in node.targets))
        setup=compile(ast.Module(body=run.body[start:end+1],type_ignores=[]),app.__file__,'exec')
        for joint,native in ((False,True),(True,False),(True,True)):
            with self.subTest(joint=joint,native=native),patch('traffic_camera.CameraPersonTracker') as camera:
                original=Mock()
                existing=object()
                namespace=dict(PersonTracker=original,person_enabled=True,both_cameras=joint,
                               person_tracker=existing,other_channel=1,args=SimpleNamespace(
                                   native_motion=native,capture_library='original.so',front_channel=0))
                exec(setup,namespace)
                namespace['hdmi_person_tracker']('original.so')
                if joint and native:
                    self.assertEqual(camera.call_args.kwargs,dict(camera_refinement=False))
                else:
                    camera.assert_not_called()
                    original.assert_called()
                    self.assertIs(namespace['person_tracker'],existing)

    def test_worker_projection_and_serial_delayed_replay_produce_the_same_boxes(self):
        serial,worker=self.tracker(),self.tracker()
        def moved(previous,current,people,fraction,shape):
            return [[person[0]+2,person[1],person[2]+2,person[3],*person[4:]] for person in people]
        for tracker in (serial,worker):tracker.match_native.side_effect=moved
        images=[]
        for number in range(6):
            image=np.full((180,320,3),30,np.uint8)
            image[40:120,80+2*number:120+2*number]=180
            images.append(image)
        for number,image in enumerate(images[:5]):
            serial.update(image,None,number)
            worker.update(image,None,number)
        detection=self.detection()
        projected=worker.align_detection(detection)
        self.assertEqual(projected['frame'],4)
        self.assertEqual(projected['source_frame'],0)
        expected=serial.update(images[5],detection,5)
        actual=worker.update(images[5],projected,5)
        self.assertEqual(actual['people'],expected['people'])
        self.assertEqual(actual['tracking_lost'],expected['tracking_lost'])
        self.assertEqual(serial.match_native.call_count,worker.match_native.call_count)

    def test_worker_projection_preserves_evidence_age_and_does_not_mutate_live_state_or_cache(self):
        tracker=self.tracker()
        image=np.full((180,320,3),30,np.uint8)
        tracker.update(image,None,0)
        tracker.update(image,None,1)
        gray,number,entries=tracker.gray,tracker.number,list(tracker.frames)
        tracker.match_native.return_value=None
        tracker.move_native.return_value=[]
        detection=self.detection()
        detection['captured_at']=12.345
        projected=tracker.align_detection(detection)
        self.assertEqual(projected['captured_at'],12.345)
        self.assertEqual(projected['shape'],detection['shape'])
        self.assertEqual(detection['frame'],0)
        self.assertEqual(len(detection['people']),1)
        self.assertIs(tracker.gray,gray)
        self.assertEqual(tracker.number,number)
        self.assertIsNone(tracker.observation)
        self.assertTrue(all(left is right for left,right in zip(entries,tracker.frames)))
        self.assertTrue(all(item[4] is None for item in tracker.frames))
        self.assertTrue(projected['tracking_lost'])
        self.assertTrue(tracker.update(image,projected,2)['tracking_lost'])

    def test_projected_old_scene_result_is_rejected_using_real_model_frame(self):
        tracker=self.tracker()
        image=np.full((180,320,3),30,np.uint8)
        tracker.update(image,None,0)
        tracker=self.tracker()
        tracker.cut_frame=2
        tracker.update(np.full_like(image,220),None,2)
        stale=dict(self.detection(),frame=20,source_frame=0,camera_channel=0)
        self.assertIsNone(tracker.update(np.full_like(image,220),stale,3))
        self.assertEqual(tracker.cut_frame,2)
        import traffic_light_3way_v2_modified as app
        tree=ast.parse(Path(app.__file__).read_text(encoding='utf-8'))
        run=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='run')
        prepare=next(node for node in run.body if isinstance(node,ast.FunctionDef) and node.name=='prepare_camera_pair')
        channel=next(node for node in prepare.body if isinstance(node,ast.FunctionDef) and node.name=='prepare_channel')
        result=next(node for node in channel.body if isinstance(node,ast.Assign) and
                    any(isinstance(target,ast.Name) and target.id=='result' for target in node.targets))
        gate=compile(ast.Expression(body=result.value),app.__file__,'eval')
        self.assertIsNone(eval(gate,dict(person_result=stale,channel=0,scene_start_frame={0:2})))
        fresh=dict(stale,source_frame=2)
        self.assertIs(eval(gate,dict(person_result=fresh,channel=0,scene_start_frame={0:2})),fresh)
        self.assertEqual(tracker.update(np.full_like(image,220),fresh,4)['source_frame'],2)


@unittest.skipUnless(os.name=='posix' and LIBRARY.is_file() and CAPTURE_LIBRARY.is_file(),
                     'RK3568 camera and original capture native libraries required')
class CameraFlowParityTests(unittest.TestCase):
    def test_dense_flow_and_moved_boxes_match_original_native_for_translation_and_disappearance(self):
        camera=ct.CDLL(str(LIBRARY))
        capture=ct.CDLL(str(CAPTURE_LIBRARY))
        camera.traffic_camera_configure_threads.argtypes=[]
        camera.traffic_camera_configure_threads.restype=ct.c_int
        self.assertEqual(camera.traffic_camera_configure_threads(),1)
        dense=camera.traffic_camera_dense_flow
        dense.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_int,ct.c_void_p]
        dense.restype=ct.c_int
        reference=capture.traffic_gray_flow
        reference.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_int,ct.c_int,ct.c_void_p,ct.c_void_p]
        reference.restype=ct.c_int
        move=camera.traffic_camera_move_people
        move.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_int,
                       ct.c_void_p,ct.c_int,ct.c_int,ct.c_double,ct.c_int,ct.c_int,ct.c_void_p]
        move.restype=ct.c_int
        texture=np.random.default_rng(17).integers(80,220,(80,40,3),dtype=np.uint8)
        before=np.full((180,320,3),30,np.uint8)
        before[40:120,80:120]=texture
        previous=np.empty((90,160),np.uint8)
        self.assertEqual(reference(None,before.ctypes.data,320,180,before.strides[0],previous.ctypes.data,None),0)
        people=np.array([[80,40,120,120,.93,0]],np.float64)
        for dx in (-10,-6,-2,0,2,6,10,None):
            with self.subTest(gray_translation=dx):
                after=np.full_like(before,30)
                if dx is not None:after[40:120,80+dx*2:120+dx*2]=texture
                gray=np.empty_like(previous)
                original_flow=np.empty((90,160,2),np.float32)
                candidate_flow=np.empty_like(original_flow)
                self.assertEqual(reference(previous.ctypes.data,after.ctypes.data,320,180,after.strides[0],
                                           gray.ctypes.data,original_flow.ctypes.data),1)
                self.assertEqual(dense(previous.ctypes.data,gray.ctypes.data,160,90,candidate_flow.ctypes.data),1)
                self.assertTrue(np.isfinite(original_flow).all())
                np.testing.assert_array_equal(candidate_flow,original_flow)
                outputs=[]
                for flow in (original_flow,candidate_flow):
                    output=np.empty_like(people)
                    count=move(previous.ctypes.data,gray.ctypes.data,flow.ctypes.data,160,90,
                               people.ctypes.data,1,6,1.,320,180,output.ctypes.data)
                    self.assertGreaterEqual(count,0)
                    outputs.append(output[:count])
                np.testing.assert_array_equal(outputs[0],outputs[1])
                if dx is None:self.assertEqual(len(outputs[1]),0)


class NativePreviewEventTests(unittest.TestCase):
    def display(self, events=()):
        from traffic_camera import NativeDisplay
        library=SimpleNamespace(**{name:Mock(return_value=1) for name in (
            'traffic_preview_open','traffic_preview_show','traffic_preview_move','traffic_preview_close',
            'traffic_preview_poll','traffic_camera_error')})
        library.traffic_preview_open.return_value=123
        library.traffic_camera_error.return_value=b'preview failed'
        pending=list(events);visible=[1]
        def poll(handle,fields):
            if pending:
                fields[:]=pending.pop(0);visible[0]=fields[5]
                return 1
            fields[5]=visible[0]
            return 0
        library.traffic_preview_poll.side_effect=poll
        with patch('traffic_camera.ct.CDLL',return_value=library):display=NativeDisplay()
        display.namedWindow('Traffic light direction')
        return display,library

    def test_key_mouse_events_use_preview_coordinates_without_gui_wait_or_readback(self):
        events=[(1,13,17,1,-1,1),(0,19,23,1,-1,1),(4,19,23,0,ord('b'),1),(-1,0,0,0,27,1)]
        display,library=self.display(events)
        received=[]
        display.setMouseCallback('Traffic light direction',lambda *args:received.append(args),'controls')
        image=np.zeros((95,132,3),np.uint8)[:,:127]
        display.imshow('Traffic light direction',image)
        self.assertEqual(library.traffic_preview_show.call_args.args[2:],(127,95,381,None))
        with patch('traffic_camera.time.sleep') as sleep:
            self.assertEqual(display.waitKey(1),27)
            self.assertEqual(display.waitKey(1),-1)
            sleep.assert_not_called()
        self.assertEqual(received,[(1,13,17,1,'controls'),(0,19,23,1,'controls'),(4,19,23,0,'controls')])
        display.destroyAllWindows()
        library.traffic_preview_close.assert_called_once_with(123)

    def test_close_updates_cached_visibility_and_cleanup_is_idempotent(self):
        display,library=self.display([(-1,0,0,0,-1,0)])
        display.moveWindow('Traffic light direction',7,11)
        display.imshow('Traffic light direction',np.zeros((64,96,3),np.uint8))
        library.traffic_preview_move.assert_called_once_with(123,7,11)
        self.assertEqual(display.waitKey(1),-1)
        self.assertEqual(display.getWindowProperty('Traffic light direction',cv2.WND_PROP_VISIBLE),0)
        display.destroyAllWindows();display.destroyAllWindows()
        library.traffic_preview_close.assert_called_once()

    def test_native_display_is_opt_in_and_rejects_standalone_hdmi(self):
        from traffic_light_3way_v2_modified import parse_opt
        with patch('sys.argv',['traffic']):self.assertFalse(parse_opt().native_display)
        options=['traffic','--native-display','--native-motion','--show','--pipeline-capture',
                 '--pcie-capture','--fpga-pci','0002:21:00.0','--front-channel','2']
        with patch('sys.argv',options),redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):parse_opt()


@unittest.skipUnless(os.name=='posix' and LIBRARY.is_file() and os.environ.get('TRAFFIC_PREVIEW_TEST')=='1',
                     'Opt-in native X11 window test; set TRAFFIC_PREVIEW_TEST=1')
class NativePreviewPixelTests(unittest.TestCase):
    def test_native_preview_preserves_bgr_pixels_orientation_and_layout(self):
        from traffic_camera import NativeDisplay
        display=NativeDisplay(LIBRARY)
        display.namedWindow('Traffic native preview pixel test')
        image=np.random.default_rng(73).integers(0,256,(95,127,3),dtype=np.uint8)
        image[:30,:40]=(0,0,255);image[:30,80:]=(255,0,0)
        image[60:,:40]=(0,255,0);image[60:,80:]=(15,80,190)
        output=np.empty_like(image)
        try:
            display.imshow('Traffic native preview pixel test',image,readback=output)
            np.testing.assert_array_equal(output,image)
            display.waitKey(1)
        finally:
            display.destroyAllWindows()

    def test_async_frames_poll_resize_and_pending_cleanup_preserve_pixels(self):
        from traffic_camera import NativeDisplay
        display=NativeDisplay(LIBRARY)
        display.namedWindow('Traffic native preview pixel test')
        random=np.random.default_rng(74)
        try:
            for height,width in ((95,127),(95,127),(96,128),(95,127)):
                for _ in range(4):
                    image=random.integers(0,256,(height,width,3),dtype=np.uint8)
                    display.imshow('Traffic native preview pixel test',image)
                    display.waitKey(1)
                output=np.empty_like(image)
                display.imshow('Traffic native preview pixel test',image,readback=output)
                np.testing.assert_array_equal(output,image)
            display.imshow('Traffic native preview pixel test',image)
        finally:
            display.destroyAllWindows()


if __name__=='__main__':unittest.main()
