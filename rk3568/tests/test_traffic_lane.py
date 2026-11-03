"""Safety decisions and geometry regressions for actual lane-change instructions."""
import unittest
import cv2
import numpy as np
from traffic_lane import (LaneSimulation, LaneEvidenceTracker, FusionSimulation, lane_canvas,
                          ego_boundaries, decode_instances, paint_evidence)
from traffic_rknn import letterbox_rgb


def paint(bottom, top, kind, cls=0):
    mask=np.zeros((360,640),np.uint8)
    cv2.line(mask,(top,170),(bottom,320),1,8)
    return {'mask':mask.astype(bool),'kind':kind,'model_class':cls,'score':.9}


class LaneTests(unittest.TestCase):
    def test_current_paint_uses_nearer_ego_boundary_and_preserves_a_nearer_model_boundary(self):
        from pathlib import Path
        from unittest.mock import MagicMock,patch
        from traffic_lane import LaneModel
        model=LaneModel.__new__(LaneModel)
        model.weights=Path('unused.rknn');model.shape=(288,480);model.threshold=.35
        model.model=MagicMock()
        frame=np.full((360,640,3),40,np.uint8)
        for side,bottom,top,near,far in (('LEFT',80,300,.35,.05),('RIGHT',560,340,.65,.95)):
            for base,expected in ((near,'DASHED'),(far,'SOLID')):
                current={'kind':'DASHED','base_x':base,'source':'current_paint_geometry'}
                with patch('traffic_lane.decode_instances',return_value=[paint(bottom,top,'SOLID')]), \
                     patch('traffic_lane.paint_boundaries',return_value={side:current}):
                    result=model.infer(frame,17,1.)
                self.assertEqual(result['boundaries'][side]['kind'],expected)
                self.assertEqual(result['frame'],17)

    def test_native_evidence_rejects_mismatched_buffers_before_calling_cpp(self):
        from unittest.mock import MagicMock
        native=MagicMock();frame=np.zeros((200,320,3),np.uint8)
        for image,channels in ((frame.astype(np.float32),None),
                               (frame,(np.zeros((100,160),np.uint8),None)),
                               (frame,(np.zeros((200,320),np.uint8),np.zeros((100,160,3),np.uint8)))):
            with self.assertRaisesRegex(ValueError,'matching uint8'):
                paint_evidence(image,0.,.5,channels,native=native)
        native.traffic_lane_evidence.assert_not_called()

    def test_two_models_do_not_overlap_runtime_transactions(self):
        import ctypes as ct
        import time
        from concurrent.futures import ThreadPoolExecutor
        from traffic_rknn import RKNNModel
        class Runtime:
            active=None
            buffer=(ct.c_float*1)(42.)
            def rknn_inputs_set(self, context, count, inputs):
                if self.active is not None:raise RuntimeError('Runtime transaction overlapped')
                self.active=context.value
                time.sleep(.002)
                return 0
            def rknn_run(self, context, options):return 0
            def rknn_outputs_get(self, context, count, outputs, options):
                outputs[0].size=4
                outputs[0].buf=ct.cast(self.buffer,ct.c_void_p)
                return 0
            def rknn_outputs_release(self, context, count, outputs):
                self.active=None
                return 0
        runtime=Runtime()
        models=[]
        for index in (1,2):
            model=RKNNModel.__new__(RKNNModel)
            model.context=ct.c_uint64(index);model.lib=runtime
            model.config={'input_size':1};model.outputs=[(1,)]
            models.append(model)
        with ThreadPoolExecutor(max_workers=2) as executor:
            results=list(executor.map(lambda i:models[i%2].infer(np.zeros((1,3,1,1),np.float32)),range(12)))
        self.assertTrue(all(float(result[0][0])==42. for result in results))

    def test_sampled_paint_color_matches_full_frame_conversion(self):
        f,_=self.tracked_road()
        fast=paint_evidence(f,-.8,.25)
        full=paint_evidence(f,-.8,.25,(cv2.cvtColor(f,cv2.COLOR_BGR2GRAY),cv2.cvtColor(f,cv2.COLOR_BGR2HSV)))
        np.testing.assert_array_equal(fast[0],full[0])
        self.assertEqual(fast[1:],full[1:])

    def test_red_signal_cannot_be_overridden_by_a_dashed_boundary(self):
        s=FusionSimulation('LEFT')
        self.assertEqual(s.update_fused({'LEFT':{'kind':'DASHED'}},{'LEFT':'red'},True),'STOP_RED')
        self.assertFalse(s.guidance_visible)
        self.assertEqual(s.update_fused({'LEFT':{'kind':'DASHED'}},{'LEFT':'red'},True,'MOVE'),'ALARM_RED')
        self.assertTrue(s.alarm)

    def test_green_does_not_allow_solid_line_lane_change(self):
        s=FusionSimulation('RIGHT');s.command('CHANGE')
        self.assertEqual(s.update_fused({'RIGHT':{'kind':'SOLID'}},{'RIGHT':'green'},True),'ALARM_SOLID')
        self.assertTrue(s.alarm)
        self.assertFalse(s.guidance_visible)

    def test_junction_turn_is_not_mistaken_for_crossing_an_adjacent_line(self):
        s=FusionSimulation('RIGHT')
        self.assertEqual(s.update_fused({'RIGHT':{'kind':'SOLID'}},{'RIGHT':'green'},True),'GO_RIGHT')
        self.assertFalse(s.alarm)

    def test_green_change_requires_known_boundary(self):
        s=FusionSimulation('LEFT');s.command('CHANGE')
        self.assertEqual(s.update_fused({}, {'LEFT':'green'},True),'WAIT_UNKNOWN')
        self.assertFalse(s.guidance_visible)

    def test_missing_active_signal_does_not_fall_back_to_dashed_permission(self):
        s=FusionSimulation('LEFT')
        self.assertEqual(s.update_fused({'LEFT':{'kind':'DASHED'}},{},True),'WAIT_UNKNOWN')
        self.assertFalse(s.guidance_visible)

    def test_road_without_signal_group_uses_lane_rules(self):
        s=FusionSimulation('LEFT')
        self.assertEqual(s.update_fused({'LEFT':{'kind':'DASHED'}},{},False),'GO_LEFT')
        self.assertEqual(s.update_fused({'LEFT':{'kind':'SOLID'}},{},False),'ALARM_SOLID')
        self.assertEqual(s.update_fused({'LEFT':{'kind':'SOLID'}},{},False,'STOP'),'STOP_MANUAL')
        self.assertFalse(s.alarm)

    def test_fusion_intent_control_is_clickable(self):
        from traffic_light_3way_v2_modified import handle_mouse,control_rectangles
        s=FusionSimulation('LEFT')
        action,(x1,y1,x2,y2)=control_rectangles((600,942),s.controls)[-1]
        self.assertEqual(action,'CHANGE')
        handle_mouse(cv2.EVENT_LBUTTONDOWN,(x1+x2)//2,(y1+y2)//2,None,(s,(600,942),(600,942)))
        self.assertTrue(s.lane_change)
        self.assertEqual(s.controls[-1],'TURN')

    def test_four_view_retains_cameras_and_copies_the_annotated_hdmi(self):
        mosaic=np.zeros((720,1280,3),np.uint8)
        mosaic[:360,:640]=20;mosaic[:360,640:]=70
        front=np.full((360,640,3),45,np.uint8)
        b={'LEFT':paint(160,290,'DASHED')}
        s=LaneSimulation('LEFT');s.update_lane(b)
        canvas=lane_canvas(mosaic,front,b,s,2,{'source_channels':(0,1,2,2),'valid_mask':15},1024,600)
        self.assertEqual(canvas.shape,(600,942,3))
        self.assertTrue((canvas[10:200,20:400]==20).all())
        self.assertTrue((canvas[10:200,500:900]==70).all())
        np.testing.assert_array_equal(canvas[265:530,:471],canvas[265:530,471:942])

    def tracked_road(self, kind='DASHED', side='LEFT'):
        frame=np.full((360,640,3),45,np.uint8)
        slope,base=(-.8,.25) if side=='LEFT' else (.8,.75)
        mask=np.zeros((360,640),np.uint8)
        for y in range(162,332):
            if kind=='DASHED' and (y-162)%60>=35:continue
            x=round((base+slope*(y/360-.86))*640)
            cv2.line(frame,(x-2,y),(x+2,y),(235,235,235),1)
            mask[y,max(0,x-2):x+3]=1
        item={'kind':kind,'base_x':base,'slope':slope,'mask':mask.astype(bool),'score':.9}
        return frame,{side:item}

    def test_dashed_confirmation_requires_distinct_inferences(self):
        f,b=self.tracked_road();t=LaneEvidenceTracker()
        for n in range(4):
            self.assertEqual(t.update(f,b,0,n,n/30)['LEFT']['kind'],'UNKNOWN')
        self.assertEqual(t.update(f,b,4,4,4/30)['LEFT']['kind'],'DASHED')

    def test_dark_dashed_paint_misclassified_as_high_confidence_solid_stays_unknown(self):
        frame,boundaries=self.tracked_road('DASHED')
        frame[~boundaries['LEFT']['mask']]=30
        boundaries['LEFT'].update(kind='SOLID',score=.99)
        tracker=LaneEvidenceTracker();simulation=FusionSimulation('LEFT');simulation.command('CHANGE')
        for number in range(5):
            result=tracker.update(frame,boundaries,number,number,number*.3)
            self.assertEqual(result['LEFT']['kind'],'UNKNOWN')
            self.assertEqual(simulation.update_fused(result,{},False),'WAIT_UNKNOWN')
            self.assertFalse(simulation.guidance_visible)
            self.assertFalse(simulation.alarm)

    def test_dark_current_solid_paint_still_prohibits_lane_change(self):
        frame,boundaries=self.tracked_road('SOLID')
        frame[~boundaries['LEFT']['mask']]=30
        tracker=LaneEvidenceTracker()
        result=tracker.update(frame,boundaries,0,0,0.)
        self.assertEqual(result['LEFT']['kind'],'SOLID')
        simulation=FusionSimulation('LEFT');simulation.command('CHANGE')
        self.assertEqual(simulation.update_fused(result,{},False),'ALARM_SOLID')
        self.assertTrue(simulation.alarm);self.assertFalse(simulation.guidance_visible)

    def test_dark_untrusted_solid_refresh_cannot_extend_verified_occlusion_hold(self):
        frame,boundaries=self.tracked_road('SOLID')
        frame[~boundaries['LEFT']['mask']]=30
        tracker=LaneEvidenceTracker()
        self.assertEqual(tracker.update(frame,boundaries,0,0,0.)['LEFT']['kind'],'SOLID')
        hidden=np.full_like(frame,30)
        for number,(now,expected) in enumerate(((.2,'SOLID'),(.55,'SOLID'),(.61,'UNKNOWN'),(1.,'UNKNOWN')),1):
            self.assertEqual(tracker.update(hidden,boundaries,number,number,now)['LEFT']['kind'],expected)
        self.assertEqual(tracker.history['LEFT']['last_solid'],0.)
        self.assertEqual(tracker.history['LEFT']['last_model'],1.)

    def test_bright_high_confidence_solid_retains_original_model_confirmation(self):
        frame,boundaries=self.tracked_road('DASHED')
        frame[~boundaries['LEFT']['mask']]=120
        boundaries['LEFT'].update(kind='SOLID',score=.99)
        tracker=LaneEvidenceTracker()
        for number,now in enumerate((0.,.9,1.8)):
            self.assertEqual(tracker.update(frame,boundaries,number,number,now)['LEFT']['kind'],'SOLID')

    def test_short_bright_solid_model_without_continuous_paint_cannot_confirm_prohibition(self):
        frame,boundaries=self.tracked_road('DASHED')
        frame[~boundaries['LEFT']['mask']]=120
        boundaries['LEFT']['mask'][:280]=False
        boundaries['LEFT'].update(kind='SOLID',score=.99)
        tracker=LaneEvidenceTracker();simulation=FusionSimulation('LEFT');simulation.command('CHANGE')
        for number in range(4):
            result=tracker.update(frame,boundaries,number,number,number*.3)
            self.assertEqual(result['LEFT']['kind'],'UNKNOWN')
            self.assertEqual(simulation.update_fused(result,{},False),'WAIT_UNKNOWN')
            self.assertFalse(simulation.guidance_visible);self.assertFalse(simulation.alarm)

    def test_current_continuous_solid_paint_overrides_short_model_extent(self):
        frame,boundaries=self.tracked_road('SOLID')
        frame[~boundaries['LEFT']['mask']]=120
        boundaries['LEFT']['mask'][:280]=False
        result=LaneEvidenceTracker().update(frame,boundaries,0,0,0.)
        self.assertEqual(result['LEFT']['kind'],'SOLID')
        simulation=FusionSimulation('LEFT');simulation.command('CHANGE')
        self.assertEqual(simulation.update_fused(result,{},False),'ALARM_SOLID')
        self.assertTrue(simulation.alarm);self.assertFalse(simulation.guidance_visible)

    def test_short_bright_solid_labels_do_not_extend_verified_solid_hold(self):
        frame,boundaries=self.tracked_road('SOLID')
        original=boundaries['LEFT']['mask'].copy();frame[~original]=120
        tracker=LaneEvidenceTracker()
        self.assertEqual(tracker.update(frame,boundaries,0,0,0.)['LEFT']['kind'],'SOLID')
        boundaries['LEFT']['mask'][:280]=False
        frame[original & ~boundaries['LEFT']['mask']]=120
        for number,(now,expected) in enumerate(((.2,'SOLID'),(.55,'SOLID'),(.61,'UNKNOWN'),(1.,'UNKNOWN')),1):
            self.assertEqual(tracker.update(frame,boundaries,number,number,now)['LEFT']['kind'],expected)
        self.assertEqual(tracker.history['LEFT']['last_solid'],0.)

    def test_short_dashed_model_keeps_original_distinct_inference_requirement(self):
        frame,boundaries=self.tracked_road('DASHED')
        frame[~boundaries['LEFT']['mask']]=120
        boundaries['LEFT']['mask'][:280]=False
        tracker=LaneEvidenceTracker()
        for number in range(3):
            self.assertEqual(tracker.update(frame,boundaries,0,number,number/30)['LEFT']['kind'],'UNKNOWN')
        self.assertEqual(tracker.update(frame,boundaries,3,3,.1)['LEFT']['kind'],'DASHED')

    def test_shared_gray_preserves_evidence_without_converting_the_frame_twice(self):
        from unittest.mock import patch
        frame,boundaries=self.tracked_road()
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        reference,reused=LaneEvidenceTracker(),LaneEvidenceTracker()
        convert=cv2.cvtColor
        def sampled_color(image,*args,**kwargs):
            self.assertNotEqual(image.shape,frame.shape)
            return convert(image,*args,**kwargs)
        for number in range(4):
            expected=reference.update(frame,boundaries,number,number,number/30)
            with patch('cv2.cvtColor',side_effect=sampled_color):
                actual=reused.update(frame,boundaries,number,number,number/30,grayscale=gray)
            for side in ('LEFT','RIGHT'):
                self.assertEqual(actual[side]['kind'],expected[side]['kind'])
                np.testing.assert_array_equal(actual[side]['mask'],expected[side]['mask'])

    def test_single_false_dashed_after_solid_cannot_grant_right(self):
        f,b=self.tracked_road('SOLID','RIGHT');t=LaneEvidenceTracker()
        self.assertEqual(t.update(f,b,0,0,0)['RIGHT']['kind'],'SOLID')
        b['RIGHT']['kind']='DASHED'
        for n in range(1,6):
            self.assertEqual(t.update(f,b,n,n,n/30)['RIGHT']['kind'],'SOLID')

    def test_one_occlusion_does_not_turn_solid_into_dashed_permission(self):
        f,b=self.tracked_road('SOLID','RIGHT');t=LaneEvidenceTracker()
        t.update(f,b,0,0,0)
        f[230:260]=45;b['RIGHT']['kind']='DASHED'
        for n in range(1,7):
            self.assertEqual(t.update(f,b,n,n,n*.1)['RIGHT']['kind'],'SOLID')

    def test_weak_dashed_cannot_reverse_solid_even_with_visible_gaps(self):
        f,b=self.tracked_road('SOLID','RIGHT');t=LaneEvidenceTracker()
        t.update(f,b,0,0,0)
        f,b=self.tracked_road('DASHED','RIGHT');b['RIGHT']['score']=.4
        for n in range(1,7):
            self.assertEqual(t.update(f,b,n,n,n*.1)['RIGHT']['kind'],'SOLID')

    def test_weak_dashed_does_not_extend_previous_permission_forever(self):
        f,b=self.tracked_road();t=LaneEvidenceTracker()
        t.update(f,b,0,0,0);t.update(f,b,1,1,.1)
        b['LEFT']['score']=.4
        self.assertEqual(t.update(f,b,2,2,.9)['LEFT']['kind'],'DASHED')
        self.assertEqual(t.update(f,b,3,3,2.2)['LEFT']['kind'],'UNKNOWN')

    def test_shadowed_paint_is_measured_by_contrast_with_asphalt(self):
        f,b=self.tracked_road('SOLID','RIGHT')
        f[:]=35;f[b['RIGHT']['mask']]=75
        _,visible,solid,gaps,_=paint_evidence(f,.8,.75)
        self.assertTrue(visible)
        self.assertTrue(solid)
        self.assertFalse(gaps)

    def test_far_reflections_are_not_drawn_as_lane_markings(self):
        frame=np.full((360,640,3),45,np.uint8)
        for y in (167,176,185,194,203):
            x=round((.25-.8*(y/360-.86))*640)
            cv2.rectangle(frame,(x-3,y),(x+3,y+1),(235,235,235),-1)
        mask,_,_,_,_=paint_evidence(frame,-.8,.25)
        self.assertFalse(mask.any())

    def test_filter_keeps_distant_paint_and_rejects_car_highlight(self):
        frame=np.full((360,640,3),45,np.uint8)
        for y in range(210,219):
            x=round((.25-.8*(y/360-.86))*640)
            cv2.line(frame,(x-1,y),(x+1,y),(235,235,235),1)
        x=round((.25-.8*(175/360-.86))*640)
        cv2.rectangle(frame,(x-3,175),(x+3,176),(235,235,235),-1)
        mask,visible,_,_,_=paint_evidence(frame,-.8,.25)
        self.assertFalse(mask[175:177].any())
        self.assertEqual(int(mask[210:219].sum()),27)
        self.assertTrue(visible)

    def test_model_extent_does_not_extend_lane_onto_distant_vehicle(self):
        frame,boundaries=self.tracked_road()
        observed=boundaries['LEFT']['mask']
        observed[:240]=False
        tracker=LaneEvidenceTracker()
        result=tracker.update(frame,boundaries,0,0,0)['LEFT']
        self.assertFalse(result['mask'][:240].any())
        self.assertTrue(result['mask'][240:].any())

    def test_vertical_reflector_is_not_paint_in_a_diagonal_lane(self):
        frame=np.full((360,640,3),45,np.uint8)
        x=round((.25-.8*(210/360-.86))*640)
        cv2.line(frame,(x,207),(x,213),(235,235,235),1)
        self.assertFalse(paint_evidence(frame,-.8,.25)[0].any())

    def test_filtered_reflectors_cannot_supply_lane_permission(self):
        frame=np.full((360,640,3),45,np.uint8)
        for y in (175,210,245):
            x=round((.25-.8*(y/360-.86))*640)
            cv2.line(frame,(x,y-4),(x,y+4),(235,235,235),1)
        mask,visible,solid,gaps,coverage=paint_evidence(frame,-.8,.25)
        self.assertFalse(mask.any())
        self.assertFalse(visible)
        self.assertFalse(solid)
        self.assertFalse(gaps)
        self.assertEqual(coverage,0.)

    def test_short_visible_distant_paint_remains_evidence(self):
        frame=np.full((360,640,3),45,np.uint8)
        for y in range(210,216):
            x=round((.25-.8*(y/360-.86))*640)
            cv2.line(frame,(x-1,y),(x+1,y),(235,235,235),1)
        mask,visible,_,_,_=paint_evidence(frame,-.8,.25)
        self.assertEqual(int(mask.sum()),18)
        self.assertTrue(visible)

    def test_missed_dashed_requires_paint_in_current_picture(self):
        f,b=self.tracked_road();t=LaneEvidenceTracker()
        t.update(f,b,0,0,0);t.update(f,b,1,1,.1)
        self.assertEqual(t.update(f,{},2,2,.2)['LEFT']['kind'],'DASHED')
        dark=np.full_like(f,45)
        self.assertEqual(t.update(dark,{},3,3,.3)['LEFT']['kind'],'UNKNOWN')

    def test_recovery_expires_without_new_model_evidence(self):
        f,b=self.tracked_road();t=LaneEvidenceTracker()
        t.update(f,b,0,0,0);t.update(f,b,1,1,.1)
        self.assertEqual(t.update(f,{},2,2,2.2)['LEFT']['kind'],'UNKNOWN')

    def test_actual_solid_to_dashed_transition_can_be_confirmed(self):
        f,b=self.tracked_road('SOLID');t=LaneEvidenceTracker()
        t.update(f,b,0,0,0)
        f,b=self.tracked_road('DASHED')
        for n in (1,2):self.assertEqual(t.update(f,b,n,n,n/30)['LEFT']['kind'],'SOLID')
        self.assertEqual(t.update(f,b,3,3,.1)['LEFT']['kind'],'DASHED')

    def test_new_solid_blocks_previously_permitted_change_immediately(self):
        f,b=self.tracked_road();t=LaneEvidenceTracker()
        t.update(f,b,0,0,0);t.update(f,b,1,1,.1)
        f,b=self.tracked_road('SOLID')
        self.assertEqual(t.update(f,b,2,2,.2)['LEFT']['kind'],'SOLID')

    def test_close_dashed_segment_is_not_reclassified_by_brightness_alone(self):
        f,b=self.tracked_road();t=LaneEvidenceTracker()
        t.update(f,b,0,0,0);t.update(f,b,1,1,.1)
        f,b=self.tracked_road('SOLID');b['LEFT']['kind']='DASHED'
        self.assertEqual(t.update(f,b,2,2,.2)['LEFT']['kind'],'DASHED')

    def test_weak_solid_conflict_pauses_permission_without_latching(self):
        f,b=self.tracked_road();t=LaneEvidenceTracker()
        t.update(f,b,0,0,0);t.update(f,b,1,1,.1)
        f,b=self.tracked_road('SOLID')
        b['LEFT'].update(kind='SOLID',score=.39)
        self.assertEqual(t.update(f,b,2,2,.2)['LEFT']['kind'],'UNKNOWN')
        b['LEFT'].update(kind='DASHED',score=.9)
        self.assertEqual(t.update(f,b,3,3,.3)['LEFT']['kind'],'DASHED')

    def test_overlapping_weak_opposite_class_is_not_a_second_vote(self):
        item=paint(150,290,'DASHED');item['score']=.88
        duplicate={**item,'kind':'SOLID','score':.44}
        self.assertEqual(ego_boundaries([item,duplicate],(360,640))['LEFT']['kind'],'DASHED')

    def test_equally_plausible_conflicting_masks_remain_unknown(self):
        item=paint(150,290,'DASHED');item['score']=.7
        duplicate={**item,'kind':'SOLID','score':.65}
        self.assertEqual(ego_boundaries([item,duplicate],(360,640))['LEFT']['kind'],'UNKNOWN')

    def test_scene_cut_discards_pre_cut_permission(self):
        f,b=self.tracked_road();t=LaneEvidenceTracker()
        t.update(f,b,0,0,0);t.update(f,b,1,1,.1)
        changed=np.full_like(f,160)
        self.assertEqual(t.update(changed,b,1,2,.2)['LEFT']['kind'],'UNKNOWN')

    def test_rectangular_input_preserves_resolution_and_padding(self):
        rgb,gain,padding=letterbox_rgb(np.zeros((360,640,3),np.uint8),(288,480))
        self.assertEqual(rgb.shape,(288,480,3))
        self.assertEqual(gain,.75)
        self.assertEqual(padding,(0.,9.))

    def test_wrong_side_class_does_not_swap_boundaries(self):
        b=ego_boundaries([paint(150,290,'DASHED',2),paint(520,350,'SOLID',1)],(360,640))
        self.assertEqual([b[x]['kind'] for x in ('LEFT','RIGHT')],['DASHED','SOLID'])

    def test_nearest_boundaries_exclude_outer_lane(self):
        b=ego_boundaries([paint(30,220,'SOLID'),paint(190,290,'DASHED'),
                          paint(490,350,'DASHED'),paint(620,390,'SOLID')],(360,640))
        self.assertEqual([b[x]['kind'] for x in ('LEFT','RIGHT')],['DASHED','DASHED'])

    def test_horizontal_reflection_is_not_a_boundary(self):
        p=paint(200,290,'DASHED')
        p['mask'][:]=False
        p['mask'][260:270,30:620]=True
        self.assertTrue(all(x['kind']=='UNKNOWN' for x in ego_boundaries([p],(360,640)).values()))

    def test_empty_input_does_not_invent_lanes(self):
        self.assertTrue(all(x['kind']=='UNKNOWN' for x in ego_boundaries([],(360,640)).values()))

    def test_solid_immediately_alarms_and_hides_arrow(self):
        s=LaneSimulation('LEFT')
        self.assertEqual(s.update_lane({'LEFT':{'kind':'SOLID'}}),'ALARM_SOLID')
        self.assertTrue(s.alarm)
        self.assertFalse(s.guidance_visible)

    def test_dashed_allows_corresponding_change(self):
        for side in ('LEFT','RIGHT'):
            s=LaneSimulation(side)
            self.assertEqual(s.update_lane({side:{'kind':'DASHED'}}),'GO_'+side)
            self.assertTrue(s.guidance_visible)
            self.assertFalse(s.alarm)

    def test_solid_still_allows_straight(self):
        s=LaneSimulation('STRAIGHT')
        self.assertEqual(s.update_lane({'LEFT':{'kind':'SOLID'}}),'GO_STRAIGHT')

    def test_stop_clears_alarm_in_same_control_update(self):
        s=LaneSimulation('RIGHT')
        b={'RIGHT':{'kind':'SOLID'}}
        s.update_lane(b)
        s.update_lane(b,'STOP')
        self.assertFalse(s.alarm)
        self.assertFalse(s.moving)

    def test_direction_change_does_not_reuse_opposite_solid(self):
        s=LaneSimulation('LEFT')
        b={'LEFT':{'kind':'SOLID'},'RIGHT':{'kind':'DASHED'}}
        s.update_lane(b)
        s.update_lane(b,'RIGHT')
        self.assertFalse(s.alarm)
        self.assertTrue(s.guidance_visible)

    def test_unknown_never_grants_a_lane_change(self):
        s=LaneSimulation('LEFT')
        s.update_lane({'LEFT':{'kind':'DASHED'}})
        self.assertEqual(s.update_lane({}),'WAIT_UNKNOWN')
        self.assertFalse(s.guidance_visible)
        self.assertFalse(s.moving)

    def test_forced_unknown_boundary_warns_without_claiming_a_solid_line(self):
        for side in ('LEFT','RIGHT'):
            simulation=FusionSimulation(side)
            simulation.command('CHANGE')
            self.assertEqual(simulation.update_fused({}, {},False,'MOVE'),'ALARM_UNKNOWN_LANE')
            self.assertTrue(simulation.alarm)
            self.assertFalse(simulation.moving or simulation.guidance_visible)
            self.assertEqual(simulation.update_fused({}, {},False,'STOP'),'STOP_MANUAL')
            self.assertFalse(simulation.alarm)

    def test_missing_model_geometry_recovers_current_solid_paint_but_rejects_blank_and_horizontal_edges(self):
        from traffic_lane import paint_boundaries
        for side in ('LEFT','RIGHT'):
            frame,_=self.tracked_road('SOLID',side)
            evidence=paint_boundaries(frame)
            self.assertEqual(evidence[side]['kind'],'SOLID')
            self.assertEqual(evidence[side]['source'],'current_paint_geometry')
        blank=np.full((360,640,3),40,np.uint8)
        self.assertEqual(paint_boundaries(blank),{})
        cv2.rectangle(blank,(180,190),(460,245),(235,235,235),3)
        self.assertEqual(paint_boundaries(blank),{})

    def test_mask_decode_removes_letterbox_padding(self):
        prediction=np.zeros((1,40,1),np.float32)
        prediction[0,:4,0]=[240,240,100,250]
        prediction[0,4,0]=.9
        prediction[0,8,0]=1
        proto=np.ones((1,32,120,120),np.float32)
        lanes=decode_instances([prediction,proto],(360,640,3),480,.75,(0,105))
        self.assertEqual(len(lanes),1)
        self.assertEqual(lanes[0]['mask'].shape,(360,640))
        self.assertFalse(lanes[0]['mask'][:,0].any())
        self.assertTrue(lanes[0]['mask'][:,320].any())


if __name__=='__main__':unittest.main()
