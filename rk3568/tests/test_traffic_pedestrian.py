import unittest
import io
import tempfile
import time
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import cv2
import numpy as np

from traffic_lane import FusionSimulation
from traffic_pedestrian import PersonModel, PersonTracker, PedestrianGuard, in_vehicle_path
from traffic_rknn import RKNNModel, decode_heads


def observation(number, timestamp, people=()):
    return {'frame': number, 'captured_at': timestamp, 'people': list(people), 'shape': (360, 640)}


class PedestrianTests(unittest.TestCase):
    person = [290, 170, 350, 310, .9, 0]

    def test_person_adapter_uses_metadata_size_and_removes_other_classes(self):
        with patch('traffic_pedestrian.RKNNModel') as factory:
            factory.return_value.config={'classes':['person','car'],'input_size':320}
            factory.return_value.detect.return_value=[self.person,[100,100,200,200,.9,1]]
            model=PersonModel('person.rknn','existing-runtime')
            result=model.infer(np.zeros((360,640,3),np.uint8),4,1.)
            self.assertEqual(result['people'],[self.person])
            self.assertEqual(factory.return_value.detect.call_args.args[1].imgsz,320)

    def test_rectangular_detector_preserves_full_front_resolution_and_uses_less_padding(self):
        model=RKNNModel.__new__(RKNNModel)
        model.config={'input_size':640,'input_shape':[384,640],'input_scale':255,
                      'classes':['person'],'anchors':[[[10,12]]*3],'strides':[8]}
        model.infer=MagicMock(return_value=[np.full((1,18,1,1),-40.,np.float32)])
        rows=model.detect(np.zeros((360,640,3),np.uint8),SimpleNamespace(imgsz=640,conf_thres=.35,iou_thres=.45))
        self.assertEqual(rows,[])
        tensor=model.infer.call_args.args[0]
        self.assertEqual(tensor.shape,(1,3,384,640))
        self.assertTrue(np.all(tensor[:,:,12:372,:]==0))
        np.testing.assert_allclose(tensor[:,:,:12,:],114/255)

    def test_decode_eighty_classes_preserves_person_label(self):
        head = np.full((1, 255, 1, 1), -40., np.float32)
        head[0, :5, 0, 0] = [0, 0, 0, 0, 4]
        head[0, 5, 0, 0] = 4
        rows = decode_heads([head], {'classes': ['person'] + [str(i) for i in range(79)],
                                     'anchors': [[[10, 12]] * 3], 'strides': [8]}, .5)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][5], 0)
        self.assertGreater(rows[0][4], .95)

    def test_sidewalk_and_distant_people_do_not_trigger_front_conflict(self):
        self.assertTrue(in_vehicle_path(self.person, (360, 640)))
        for row in ([10, 180, 55, 310], [580, 180, 630, 310], [310, 75, 325, 95], [310, 80, 330, 150]):
            self.assertFalse(in_vehicle_path(row, (360, 640)), row)

    def test_hazard_overrides_green_and_forced_move_in_same_update(self):
        guard = PedestrianGuard()
        state = guard.update(observation(1, 1., [self.person]), 1.)
        simulation = FusionSimulation('STRAIGHT')
        simulation.update_fused({}, {'STRAIGHT': 'green'}, True, 'MOVE')
        self.assertTrue(simulation.guidance_visible)
        guard.apply(simulation, state)
        self.assertFalse(simulation.moving)
        self.assertFalse(simulation.guidance_visible)
        self.assertTrue(simulation.alarm)
        self.assertIs(simulation.manual_moving, True)
        self.assertEqual(simulation.decision, 'BRAKE_PEDESTRIAN')

    def test_single_miss_or_reused_result_cannot_release_braking(self):
        guard = PedestrianGuard()
        guard.update(observation(1, 0., [self.person]), 0.)
        empty = observation(2, 1.1)
        self.assertEqual(guard.update(empty, 1.1)['status'], 'BRAKE')
        self.assertEqual(guard.update(empty, 1.2)['status'], 'BRAKE')
        self.assertEqual(guard.update(observation(3, 1.3), 1.3)['status'], 'CLEAR')

    def test_missing_model_evidence_cannot_grant_movement(self):
        simulation = FusionSimulation('STRAIGHT')
        simulation.update_fused({}, {'STRAIGHT': 'green'}, True)
        guard = PedestrianGuard()
        state = guard.update(None, 1.)
        guard.apply(simulation, state)
        self.assertFalse(simulation.moving)
        self.assertFalse(simulation.guidance_visible)
        self.assertEqual(simulation.decision, 'WAIT_PEDESTRIAN')

    def test_stale_clear_result_cannot_release_a_hazard(self):
        guard = PedestrianGuard()
        guard.update(observation(1, 0., [self.person]), 0.)
        self.assertEqual(guard.update(observation(2, .1), 5.)['status'], 'BRAKE')

    def test_clear_result_tolerates_normal_queue_delay_but_still_expires(self):
        guard = PedestrianGuard()
        guard.update(observation(1, 0.), 0.)
        clear = observation(2, .2)
        self.assertEqual(guard.update(clear, .2)['status'], 'CLEAR')
        self.assertEqual(guard.update(clear, 1.05)['status'], 'CLEAR')
        self.assertEqual(guard.update(clear, 1.25)['status'], 'UNKNOWN')

    def test_tracked_person_can_enter_conflict_before_next_inference(self):
        guard = PedestrianGuard()
        guard.update(observation(1, 0., [[10,170,50,310,.9,0]]), 0.)
        self.assertEqual(guard.update(observation(1, 0., [self.person]), .1)['status'], 'BRAKE')

    def test_lost_box_is_removed_without_counting_tracking_loss_as_clearance(self):
        guard = PedestrianGuard()
        guard.update(observation(1, 0., [self.person]), 0.)
        for number in (2,3):
            state=guard.update(dict(observation(number,1.2),tracking_lost=True),1.2)
            self.assertEqual(state['people'],[])
            self.assertEqual(state['status'],'BRAKE')
        self.assertEqual(guard.update(observation(4,1.3),1.3)['status'],'BRAKE')
        self.assertEqual(guard.update(observation(5,1.4),1.4)['status'],'CLEAR')

    def test_person_box_follows_body_instead_of_static_background(self):
        rng=np.random.default_rng(5)
        background=rng.integers(20,45,(200,320),dtype=np.uint8)
        body=rng.integers(80,230,(100,40),dtype=np.uint8)
        def frame(x):
            image=background.copy();image[70:170,x:x+40]=body
            return cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
        first=frame(90);tracker=PersonTracker()
        tracker.update(first,None,0)
        detected=dict(observation(0,0.,[[90,70,130,170,.9,0]]),shape=(200,320),
                      gray=cv2.cvtColor(first,cv2.COLOR_BGR2GRAY))
        state=tracker.update(frame(102),detected,1)
        np.testing.assert_allclose(state['people'][0][:4],[102,70,142,170],atol=1.)
        state=tracker.update(frame(114),None,2)
        np.testing.assert_allclose(state['people'][0][:4],[114,70,154,170],atol=1.)

    def test_delayed_detection_keeps_all_motion_at_high_frame_rates(self):
        tracker=PersonTracker();frame=np.zeros((200,320,3),np.uint8)
        flow=np.zeros((100,160,2),np.float32);flow[:,:,0]=1
        tracker.flow=MagicMock();tracker.flow.calc.return_value=flow
        for number in range(20):tracker.update(frame,None,number)
        detected=dict(observation(0,0.,[[20,20,60,160,.9,0]]),shape=(200,320))
        state=tracker.update(frame,detected,20)
        np.testing.assert_allclose(state['people'][0][:4],[60,20,100,160],atol=1e-6)

    def test_lazy_motion_preserves_a_late_detection_without_processing_empty_frames(self):
        tracker=PersonTracker();tracker.lazy_flow=True
        frame=np.zeros((200,320,3),np.uint8)
        flow=np.zeros((100,160,2),np.float32);flow[:,:,0]=1
        tracker.flow=MagicMock();tracker.flow.calc.return_value=flow
        for number in range(20):tracker.update(frame,None,number)
        tracker.flow.calc.assert_not_called()
        result=dict(observation(0,0.,[[20,20,60,160,.9,0]]),shape=(200,320))
        state=tracker.update(frame,result,20)
        np.testing.assert_allclose(state['people'][0][:4],[60,20,100,160],atol=1e-6)
        self.assertEqual(tracker.flow.calc.call_count,20)
        empty=dict(observation(21,1.),shape=(200,320))
        tracker.update(frame,empty,21);count=tracker.flow.calc.call_count
        for number in range(22,30):tracker.update(frame,None,number)
        self.assertEqual(tracker.flow.calc.call_count,count)

    def test_disappearance_and_video_cut_remove_boxes_before_detection_expires(self):
        rng=np.random.default_rng(5)
        background=rng.integers(20,45,(200,320),dtype=np.uint8)
        first=background.copy();first[70:170,90:130]=rng.integers(80,230,(100,40),dtype=np.uint8)
        first=cv2.cvtColor(first,cv2.COLOR_GRAY2BGR)
        detection=dict(observation(0,0.,[[90,70,130,170,.9,0]]),shape=(200,320),
                       gray=cv2.cvtColor(first,cv2.COLOR_BGR2GRAY))
        for current in (background,rng.integers(0,255,(200,320),dtype=np.uint8)):
            tracker=PersonTracker();tracker.update(first,detection,0)
            state=tracker.update(cv2.cvtColor(current,cv2.COLOR_GRAY2BGR),None,1)
            self.assertTrue(state is None or state['people']==[])
            state=tracker.update(first,None,2)
            self.assertTrue(state is None or state['people']==[])
            refreshed={**detection,'frame':3}
            self.assertEqual(len(tracker.update(first,refreshed,3)['people']),1)

    def test_weak_texture_keeps_small_confirmed_people_and_fresh_negative_clears_them(self):
        tracker=PersonTracker();frame=np.full((200,320,3),80,np.uint8)
        people=[[90,70,95,95,.8,0],[150,70,180,170,.8,0]]
        result=dict(observation(0,0.,people),shape=(200,320))
        self.assertEqual(len(tracker.update(frame,result,0)['people']),2)
        self.assertEqual(len(tracker.update(frame,None,1)['people']),2)
        self.assertEqual(tracker.update(frame,dict(observation(2,.2),shape=(200,320)),2)['people'],[])

    def test_late_detection_from_before_a_scene_cut_cannot_restore_a_ghost(self):
        tracker=PersonTracker();first=np.full((200,320,3),20,np.uint8)
        second=np.full((200,320,3),230,np.uint8)
        result=dict(observation(0,0.,[[90,70,130,170,.8,0]]),shape=(200,320))
        tracker.update(first,result,0)
        self.assertIsNone(tracker.update(second,None,1))
        self.assertIsNone(tracker.update(second,result,2))
        refreshed={**result,'frame':3,'people':[[200,70,240,170,.8,0]]}
        self.assertEqual(len(tracker.update(second,refreshed,3)['people']),1)

    def test_manual_stop_clears_the_alarm_and_keeps_the_vehicle_stopped(self):
        simulation = FusionSimulation('STRAIGHT')
        simulation.update_fused({}, {'STRAIGHT': 'green'}, True, 'STOP')
        PedestrianGuard.apply(simulation, {'status': 'BRAKE'})
        self.assertFalse(simulation.moving)
        self.assertFalse(simulation.alarm)
        self.assertEqual(simulation.decision, 'STOP_MANUAL')

    def test_motion_button_can_cancel_a_forced_move_while_guard_holds_braking(self):
        simulation = FusionSimulation('STRAIGHT')
        simulation.update_fused({}, {'STRAIGHT': 'green'}, True, 'MOVE')
        PedestrianGuard.apply(simulation, {'status': 'BRAKE'})
        simulation.update_fused({}, {'STRAIGHT': 'green'}, True, 'TOGGLE')
        PedestrianGuard.apply(simulation, {'status': 'BRAKE'})
        self.assertIs(simulation.manual_moving, False)
        self.assertFalse(simulation.moving)
        self.assertFalse(simulation.alarm)

    def test_live_pipeline_publishes_braking_before_drawing_and_uses_clean_input(self):
        simulation,valid,person_worker,signal_worker=self._run_live_pipeline(15)
        self.assertTrue(valid)
        self.assertEqual(simulation.decision,'BRAKE_PEDESTRIAN')
        self.assertFalse(simulation.moving)
        self.assertTrue(simulation.alarm)
        self.assertFalse(simulation.guidance_visible)
        self.assertTrue(np.all(person_worker.submit.call_args.args[0][0]==0))
        self.assertTrue(np.all(signal_worker.submit.call_args.args[0][0]==0))

    def test_invalid_front_frame_does_not_consume_or_submit_person_observations(self):
        simulation,valid,person_worker,_=self._run_live_pipeline(3)
        self.assertFalse(valid)
        self.assertFalse(simulation.moving)
        self.assertFalse(simulation.alarm)
        self.assertEqual(simulation.decision,'WAIT_PEDESTRIAN')
        person_worker.take.assert_not_called()
        person_worker.submit.assert_not_called()

    def test_overlapped_capture_keeps_model_input_clean_and_publishes_braking(self):
        simulation,valid,person_worker,_=self._run_live_pipeline(15,True)
        self.assertTrue(valid)
        self.assertEqual(simulation.decision,'BRAKE_PEDESTRIAN')
        self.assertTrue(simulation.alarm)
        self.assertTrue(np.all(person_worker.submit.call_args.args[0][0]==0))

    def _run_live_pipeline(self,valid_mask,pipeline=False):
        import traffic_light_3way_v2_modified as traffic
        from traffic_rknn import PCIeCapture
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            root=Path(folder);weights=root/'color.rknn';weights.touch()
            options=['traffic','--assist-mode','fusion','--pcie-capture','--show','--no-record',
                     '--weights',str(weights),'--direction-weights',str(root/'direction.rknn'),
                     '--pedestrian-weights',str(root/'person.rknn'),'--fpga-pci','0002:21:00.0','--front-channel','2',
                     '--output',str(root/'out.mp4')]
            if pipeline:options+=['--pipeline-capture']
            stack.enter_context(patch('sys.argv',options));args=traffic.parse_opt()
            capture=MagicMock();capture.dma_ms=1.;capture.capture_mode=0
            capture.read.side_effect=lambda **kwargs:(True,np.zeros((720,1280,3),np.uint8))
            capture.get.side_effect=lambda key:{cv2.CAP_PROP_FRAME_WIDTH:1280,cv2.CAP_PROP_FRAME_HEIGHT:720,
                                               cv2.CAP_PROP_FPS:30}.get(key,0)
            capture.video_status={'full_hdmi_capture':False,'valid_mask':valid_mask,
                                  'source_channels':(0,1,2,2),'good_frames':(1,1,1,1)}
            factory=stack.enter_context(patch('traffic_rknn.PCIeCapture',return_value=capture))
            factory.channel_bounds.side_effect=PCIeCapture.channel_bounds
            stack.enter_context(patch('traffic_rknn.RKNNModel'))
            stack.enter_context(patch.object(traffic,'DirectionClassifier'))
            stack.enter_context(patch('traffic_lane.LaneModel'))
            stack.enter_context(patch('traffic_pedestrian.PersonModel'))
            tracker=stack.enter_context(patch('traffic_lane.LaneEvidenceTracker')).return_value
            tracker.update.return_value={'LEFT':{'kind':'DASHED'}} if valid_mask==15 else {}
            lane_worker,person_worker,signal_worker=MagicMock(),MagicMock(),MagicMock()
            lane_worker.take.return_value=signal_worker.take.return_value=None
            person_worker.take.side_effect=lambda:dict(observation(0,time.perf_counter(),[self.person]),processing_ms=10.,
                                                       gray=np.zeros((360,640),np.uint8))
            stack.enter_context(patch.object(traffic,'LatestFrameInference',side_effect=[lane_worker,person_worker,signal_worker]))
            fpga=stack.enter_context(patch('traffic_fpga.FPGAInteraction')).return_value
            fpga.poll.return_value={'navigation':'STRAIGHT','manual_moving':True}
            fpga.publish.return_value=0xA55A006D
            stack.enter_context(patch('traffic_fpga.CPUMotionKey')).return_value.poll.return_value=None
            ui={name:stack.enter_context(patch('cv2.'+name)) for name in
                ('namedWindow','moveWindow','setMouseCallback','imshow','destroyAllWindows')}
            stack.enter_context(patch('cv2.waitKey',return_value=ord('q')))
            stack.enter_context(patch('traffic_lane.draw_lanes',side_effect=lambda image,boundaries:image.fill(231)))
            stack.enter_context(patch.object(traffic,'draw_overlay'))
            stack.enter_context(redirect_stdout(io.StringIO()))
            traffic.run(args)
            actual_shape=ui['imshow'].call_args.args[1].shape[:2]
            mouse_context=ui['setMouseCallback'].call_args.args[2]
            self.assertEqual(mouse_context[1],actual_shape)
            self.assertEqual(mouse_context[2],actual_shape)
            simulation,valid=fpga.publish.call_args.args
            return simulation,valid,person_worker,signal_worker
