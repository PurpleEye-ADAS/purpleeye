# 姓名：丁黄炎
"""Exercise concurrent source dispatch and HDMI mirror rendering without board hardware."""
from contextlib import ExitStack,redirect_stdout
from pathlib import Path
from unittest.mock import patch,MagicMock
import io,time,tempfile,unittest
import ast,copy
from types import SimpleNamespace
import cv2,numpy as np


class FourInputTests(unittest.TestCase):
    def test_independent_routes_do_not_overwrite_driver_intent_and_hardware_alarm_does_not_mutate_views(self):
        from traffic_lane import FusionSimulation
        from traffic_light_3way_v2_modified import select_fpga_control
        from traffic_pedestrian import PedestrianGuard
        views=[FusionSimulation() for _ in range(3)]
        views[0].navigation_peers=tuple(views[1:])
        for view,direction in zip(views,('LEFT','RIGHT','STRAIGHT')):
            view.navigation_enabled=True;view.navigation_direction=direction
            view.update_fused({}, {},False)
        self.assertEqual([view.nav for view in views],['LEFT','RIGHT','STRAIGHT'])
        self.assertEqual([view.requested_nav for view in views],['STRAIGHT']*3)
        views[0].command('RIGHT')
        for view in views:
            view.requested_nav=views[0].requested_nav
            view.navigation_manual=views[0].navigation_manual
            view.update_fused({}, {},False)
        self.assertEqual([view.nav for view in views],['RIGHT']*3)
        views[0].command('AUTO')
        for view in views:
            view.navigation_manual=False
            view.update_fused({}, {},False)
        self.assertEqual([view.nav for view in views],['LEFT','RIGHT','STRAIGHT'])
        views[0].navigation_enabled=False;views[0].update_fused({}, {},False)
        self.assertEqual(views[0].nav,'RIGHT')
        PedestrianGuard.apply(views[1],{'status':'BRAKE'})
        before=[view.snapshot() for view in views]
        self.assertIs(select_fpga_control(*views),views[1])
        self.assertEqual([view.snapshot() for view in views],before)
        self.assertFalse(views[0].alarm or views[2].alarm)
        self.assertTrue(views[2].guidance_visible)

    def test_single_view_alarm_border_stays_in_its_own_tile(self):
        from traffic_light_3way_v2_modified import VehicleSimulation,draw_overlay
        simulation=VehicleSimulation('STRAIGHT',True)
        simulation.update({'STRAIGHT':'red'},True)
        frame=np.zeros((430,640,3),np.uint8)
        draw_overlay(frame,[],{},simulation,None,guidance_bounds=(0,0,.5,180/430),video_shape=frame.shape)
        self.assertTrue(np.any(frame[:180,:320,2]>180))
        self.assertFalse(np.any(frame[:360,320:,2]))
        self.assertFalse(np.any(frame[180:360,:320,2]))

    def test_empty_signal_dispatch_preserves_observations_filters_and_parallel_nonempty_cases(self):
        import traffic_light_3way_v2_modified as app
        tree=ast.parse(Path(app.__file__).read_text(encoding='utf-8'))
        run=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='run')
        block=next(node for node in ast.walk(run) if isinstance(node,ast.If) and
            isinstance(node.test,ast.Name) and node.test.id=='both_cameras' and node.body and
            isinstance(node.body[0],ast.Assign) and any(isinstance(target,ast.Name) and target.id=='tracking_frames'
                                                       for target in node.body[0].targets))
        dispatch=compile(ast.Module(body=block.body,type_ignores=[]),app.__file__,'exec')
        light=dict(box=[12,8,38,55],direction='STRAIGHT',color='green',color_conf=1.)
        empty=dict(frame=1,loop=0,group_revision=0,captured_at=1.,lights=[],states={})
        frame=np.full((80,160,3),30,np.uint8);cv2.circle(frame,(25,25),6,(0,255,0),-1)
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        frames=[(1,0,0,gray),(2,0,0,gray)]
        def filters():return {direction:app.StableState(1) for direction in app.DIRECTIONS}
        cases=(('none',None,False),('empty',empty,False),
               ('expired_nonempty',dict(empty,captured_at=-10.,lights=[light]),True),
               ('retained_alignment',dict(empty,_live_alignment=(1,[light])),True))
        for label,result,parallel in cases:
            with self.subTest(label=label):
                results=[copy.deepcopy(result),copy.deepcopy(empty),None]
                originals=copy.deepcopy(results)
                expected_filters=[filters() for _ in range(3)]
                actual_filters=[filters() for _ in range(3)]
                expected=[app.current_observation(value,0,0,1.1,2.5,frames,(80,160),None,frame,
                                                  expected_filters[index],{},camera_color=index<2)
                          for index,value in enumerate(originals)]
                pool=MagicMock();pool.map.side_effect=lambda function,jobs:map(function,jobs)
                observe=MagicMock(side_effect=app.current_observation)
                namespace=dict(live_frames=frames,time=SimpleNamespace(perf_counter=lambda:1.1),
                    latest_result=results[0],other_latest=results[1],hdmi_latest=results[2],
                    loop_index=0,group_revision=0,height=80,width=160,frame=frame,
                    args=SimpleNamespace(max_result_age=2.5,circle_direction=None),
                    live_filters=actual_filters[0],other_filters=actual_filters[1],hdmi_filters=actual_filters[2],
                    motion_history={},other_motion_history={},hdmi_motion_history={},tracking_pool=pool,
                    current_observation=observe)
                exec(dispatch,namespace)
                actual=[(namespace['lights'],namespace['view_states']),namespace['other_observation'],
                        (namespace['hdmi_lights'],namespace['hdmi_states'])]
                self.assertEqual(actual,expected)
                self.assertEqual(observe.call_count,3)
                self.assertTrue(all(call.args[3]==1.1 for call in observe.call_args_list))
                self.assertEqual(pool.map.call_count,int(parallel))
                self.assertEqual([[value.__dict__ for value in item.values()] for item in actual_filters],
                                 [[value.__dict__ for value in item.values()] for item in expected_filters])

    def test_weak_player_icon_cannot_inherit_a_camera_signal_identity(self):
        from types import SimpleNamespace
        from traffic_light_3way_v2_modified import FrameAnalyzer
        image=np.full((360,640,3),123,np.uint8);image[250:270,105:115]=(0,255,0)
        args=SimpleNamespace(pcie_capture=True,front_channel=0,signal_roi=None,direction_centers=None,
            circle_direction=None,confirm_frames=2,top_ratio=1,tiles=1,iou_thres=.45,conf_thres=.65,
            show=True,camera_generation=0)
        classifier=MagicMock();classifier.predict.side_effect=[[('LEFT',.99)],[('LEFT',.99),('UNKNOWN',.5)]]
        analyzer=FrameAnalyzer(SimpleNamespace(names={0:'red',1:'green'}),classifier,None)
        head=[100,40,120,100,.98,0];icon=[100,240,120,300,.4,1]
        with patch('traffic_light_3way_v2_modified.detect_roi',side_effect=[[head],[head,icon]]):
            first=analyzer(image,args,1,0,0,1.)
            moved=dict(first['lights'][0],box=icon[:4])
            first['_live_alignment']=(2,[moved]);args.live_signal_alignment=(first['frame'],first['_live_alignment'])
            result=analyzer(image,args,3,0,0,1.2)
        self.assertEqual([light['box'] for light in result['lights']],[head[:4]])

    def test_camera_missing_proposal_can_recover_a_currently_lit_known_housing(self):
        from types import SimpleNamespace
        from traffic_light_3way_v2_modified import FrameAnalyzer
        image=np.full((360,640,3),30,np.uint8);image[50:60,105:115]=(0,0,255)
        args=SimpleNamespace(pcie_capture=True,front_channel=0,signal_roi=None,direction_centers=None,
            circle_direction=None,confirm_frames=2,top_ratio=1,tiles=1,iou_thres=.45,conf_thres=.65,
            show=True,camera_generation=0)
        classifier=MagicMock();classifier.predict.side_effect=[[('LEFT',.99)],[]]
        analyzer=FrameAnalyzer(SimpleNamespace(names={0:'red'}),classifier,None)
        with patch('traffic_light_3way_v2_modified.detect_roi',side_effect=[[[100,40,120,100,.98,0]],[]]):
            first=analyzer(image,args,1,0,0,1.)
            first['_live_alignment']=(2,first['lights']);args.live_signal_alignment=(first['frame'],first['_live_alignment'])
            result=analyzer(image,args,3,0,0,1.2)
        self.assertEqual(len(result['lights']),1)
        self.assertEqual((result['lights'][0]['direction'],result['lights'][0]['color']),('LEFT','red'))

    def test_multisource_restarts_identity_when_live_geometry_is_missing(self):
        from types import SimpleNamespace
        from traffic_light_3way_v2_modified import FrameAnalyzer
        image=np.full((180,320,3),123,np.uint8)
        args=SimpleNamespace(pcie_capture=True,front_channel=2,signal_roi=None,direction_centers=None,
            circle_direction=None,confirm_frames=2,top_ratio=1,tiles=1,iou_thres=.45,show=True,camera_generation=0)
        classifier=MagicMock();classifier.predict.side_effect=[[('STRAIGHT',.99)],[('UNKNOWN',0.)]]
        analyzer=FrameAnalyzer(SimpleNamespace(names={0:'red'}),classifier,None)
        with patch('traffic_light_3way_v2_modified.detect_roi',return_value=[[100,40,120,80,.98,0]]):
            analyzer(image,args,1,0,0,1.)
            with patch.object(analyzer.direction_history,'advance',side_effect=AssertionError('obsolete history')):
                result=analyzer(image,args,10,0,0,2.)
        self.assertEqual(result['lights'][0]['direction'],'UNKNOWN')
        self.assertFalse(result['geometry_reused'])

    def test_multisource_hdmi_merges_nested_housings_but_keeps_adjacent_lights(self):
        from types import SimpleNamespace
        from traffic_light_3way_v2_modified import FrameAnalyzer
        image=np.full((180,320,3),123,np.uint8)
        boxes=[[100,40,120,100,.99,0],[102,65,118,85,.98,0],[140,40,160,100,.97,0]]
        def analyze(concurrent):
            args=SimpleNamespace(pcie_capture=True,front_channel=2,signal_roi=None,
                direction_centers=None,circle_direction=None,confirm_frames=1,top_ratio=1,tiles=1,
                iou_thres=.45,show=True)
            if concurrent:args.camera_generation=0
            classifier=MagicMock();classifier.predict.side_effect=lambda image,rows:[('STRAIGHT',.99)]*len(rows)
            analyzer=FrameAnalyzer(SimpleNamespace(names={0:'red'}),classifier,None)
            with patch('traffic_light_3way_v2_modified.detect_roi',return_value=boxes):
                return analyzer(image,args,1,0,0,1.)['lights']
        self.assertEqual([light['box'] for light in analyze(True)],[boxes[0][:4],boxes[2][:4]])
        self.assertEqual(len(analyze(False)),3)

    def test_model_reuses_only_recent_live_geometry_in_camera_and_hdmi_coordinates(self):
        from types import SimpleNamespace
        from traffic_light_3way_v2_modified import FrameAnalyzer
        for channel,shape,bounds,scale in ((0,(720,1280),(0.,0.,.5,.5),.5),
                                          (1,(720,1280),(.5,0.,1.,.5),.5),
                                          (2,(360,640),(0.,.5,.5,1.),1.)):
            with self.subTest(channel=channel):
                image=np.full((*shape,3),123,np.uint8)
                args=SimpleNamespace(pcie_capture=True,front_channel=channel,signal_roi=None,
                    direction_centers=None,circle_direction=None,confirm_frames=2,top_ratio=1,tiles=1,
                    iou_thres=.45,conf_thres=.65,show=True,camera_generation=0)
                classifier=MagicMock();classifier.predict.side_effect=[[('STRAIGHT',.99)],[('UNKNOWN',0.)]]
                analyzer=FrameAnalyzer(SimpleNamespace(names={0:'red'}),classifier,None)
                with patch('traffic_light_3way_v2_modified.detect_roi',side_effect=[
                        [[100,40,120,80,.98,0]],[[130,40,150,80,.98,0]]]):
                    first=analyzer(image,args,1,0,0,1.,native_bounds=bounds,display_shape=(720,1280))
                    moved=dict(first['lights'][0],box=(np.array(first['lights'][0]['box'])+[30*scale,0,30*scale,0]).tolist())
                    first['_live_alignment']=(2,[moved])
                    args.live_signal_alignment=(first['frame'],first['_live_alignment'])
                    first['_live_alignment']=(10,[dict(moved,box=[0,0,1,1])])
                    with patch.object(analyzer.direction_history,'advance',side_effect=AssertionError('duplicate history')):
                        second=analyzer(image,args,3,0,0,1.2,native_bounds=bounds,display_shape=(720,1280))
                self.assertEqual(second['lights'][0]['direction'],'STRAIGHT')
                self.assertTrue(second['geometry_reused'])
                np.testing.assert_allclose(second['lights'][0]['box'],moved['box'])
                second['_live_alignment']=(3,second['lights'])
                self.assertFalse(analyzer.reuse_live_tracks(shape,7,bounds,(720,1280)))
                self.assertFalse(analyzer.reuse_live_tracks(shape,2,bounds,(720,1280)))
                self.assertFalse(analyzer.reuse_live_tracks((200,300),3,bounds,(720,1280)))

    def test_scene_change_ignores_foreground_traffic_and_small_lamp_changes(self):
        from traffic_light_3way_v2_modified import source_scene_changed
        original=np.full((360,640),60,np.uint8)
        self.assertFalse(source_scene_changed(None,original))
        foreground=original.copy();foreground[180:]=240
        self.assertFalse(source_scene_changed(original,foreground))
        lamp=original.copy();lamp[20:40,100:140]=255
        self.assertFalse(source_scene_changed(original,lamp))
        daylight=np.full_like(original,160)
        self.assertTrue(source_scene_changed(original,daylight))

    @staticmethod
    def _scene_texture(seed=12):
        image=np.random.default_rng(seed).integers(0,256,(45,80),dtype=np.uint8)
        return cv2.resize(image,(640,360),interpolation=cv2.INTER_LINEAR)

    def test_supported_scene_translation_and_rotation_do_not_reset_either_roi(self):
        from traffic_rknn import source_scene_changed
        original=self._scene_texture()
        transforms=(np.array([[1,0,40],[0,1,-8]],np.float32),np.array([[1,0,180],[0,1,-8]],np.float32),
                    cv2.getRotationMatrix2D((320,180),4,1))
        for transform in transforms:
            current=cv2.warpAffine(original,transform,(640,360),borderMode=cv2.BORDER_REFLECT)
            for upper_only in (True,False):
                with self.subTest(transform=transform.tolist(),upper_only=upper_only):
                    top=180 if upper_only else 360
                    self.assertGreater(cv2.norm(original[:top],current[:top],cv2.NORM_L1)/current[:top].size,35)
                    self.assertFalse(source_scene_changed(original,current,upper_only=upper_only))

    def test_scene_owner_preserves_real_cuts_brightness_changes_and_insufficient_texture(self):
        from traffic_rknn import source_scene_changed
        original=self._scene_texture()
        brighter=cv2.add(original,70)
        edge=np.zeros_like(original);edge[:,320:]=255
        moved_edge=cv2.warpAffine(edge,np.array([[1,0,100],[0,1,0]],np.float32),(640,360))
        for previous,current in ((original,self._scene_texture(43)),(original,brighter),(edge,moved_edge)):
            for upper_only in (True,False):
                with self.subTest(upper_only=upper_only,low_texture=previous is edge):
                    self.assertTrue(source_scene_changed(previous,current,upper_only=upper_only))
        with patch('traffic_rknn.cv2.ORB_create',side_effect=AssertionError('unneeded motion estimate')):
            self.assertFalse(source_scene_changed(original,cv2.add(original,10)))

    def test_joint_tracker_does_not_own_scene_reset_or_refresh_result_age(self):
        import traffic_light_3way_v2_modified as app
        import traffic_camera
        from traffic_rknn import source_scene_changed
        from traffic_pedestrian import PersonTracker,PedestrianGuard
        self.assertIs(app.source_scene_changed,source_scene_changed)
        tracker=traffic_camera.CameraPersonTracker.__new__(traffic_camera.CameraPersonTracker)
        PersonTracker.__init__(tracker)
        tracker.camera_refinement=False
        gray=self._scene_texture();image=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
        result=dict(frame=1,source_frame=1,captured_at=1.,people=[],shape=image.shape[:2])
        original=tracker.update(image,result,1)
        moved=cv2.warpAffine(image,np.array([[1,0,40],[0,1,-8]],np.float32),(640,360),borderMode=cv2.BORDER_REFLECT)
        observation=tracker.update(moved,None,2)
        self.assertIs(observation,original)
        foreground=moved.copy();foreground[180:]=240
        self.assertGreater(cv2.norm(moved,foreground,cv2.NORM_L1)/foreground.size,35)
        self.assertIs(tracker.update(foreground,None,3),original)
        self.assertEqual(observation['captured_at'],1.)
        guard=PedestrianGuard(max_age=1.5,clear_frames=1)
        self.assertEqual(guard.update(observation,1.1)['status'],'CLEAR')
        self.assertEqual(guard.update(observation,2.6)['status'],'UNKNOWN')

    def test_outer_scene_and_revision_resets_rebuild_trackers_with_a_source_frame_boundary(self):
        import traffic_light_3way_v2_modified as app
        from traffic_camera import CameraPersonTracker
        from traffic_pedestrian import PersonTracker
        tree=ast.parse(Path(app.__file__).read_text(encoding='utf-8'))
        prepare=next(node for node in ast.walk(tree) if isinstance(node,ast.FunctionDef) and node.name=='prepare_camera_pair')
        revision=next(node for node in prepare.body if isinstance(node,ast.If))
        scenes=next(node for node in prepare.body if isinstance(node,ast.For))
        resets=compile(ast.Module(body=[revision,scenes],type_ignores=[]),app.__file__,'exec')
        def create_tracker(*args):
            tracker=CameraPersonTracker.__new__(CameraPersonTracker)
            PersonTracker.__init__(tracker);tracker.camera_refinement=False
            return tracker
        factory=MagicMock(side_effect=create_tracker)
        gray=np.full((360,640),30,np.uint8)
        namespace=dict(prepared_revision=(0,0),prepared_camera_mask=7,mask=7,
            scene_start_frame={channel:0 for channel in (0,1,2)},
            pair_lane_latest={channel:None for channel in (0,1,2)},pair_lane_trackers={},
            pair_person_trackers={channel:create_tracker() for channel in (0,1,2)},
            pair_scene_grays={channel:gray for channel in (0,1,2)},
            person_enabled=True,hdmi_person_tracker=factory,camera_person_tracker=factory,
            LaneEvidenceTracker=lambda *args:object(),args=SimpleNamespace(capture_library=None,native_motion=False),
            changed_camera_channels=app.changed_camera_channels,source_scene_changed=app.source_scene_changed)
        for number,channel,current in ((10,0,gray),(20,2,np.full_like(gray,200))):
            with self.subTest(channel=channel):
                before=dict(namespace['pair_person_trackers'])
                namespace.update(packet={'number':number},revision=(1,0),
                    grays={index:current if index==channel else namespace['pair_scene_grays'][index] for index in (0,1,2)})
                exec(resets,namespace)
                factory.assert_called_once();factory.reset_mock()
                tracker=namespace['pair_person_trackers'][channel]
                self.assertIsNot(tracker,before[channel])
                self.assertEqual((tracker.cut_frame,namespace['scene_start_frame'][channel]),(number,number))
                self.assertTrue(all(namespace['pair_person_trackers'][other] is before[other] for other in (0,1,2) if other!=channel))
                image=cv2.cvtColor(current,cv2.COLOR_GRAY2BGR)
                old=dict(frame=number+5,source_frame=number-1,captured_at=1.,people=[],shape=image.shape[:2])
                self.assertIsNone(tracker.update(image,old,number+5))
                fresh=dict(old,source_frame=number,captured_at=2.)
                observation=tracker.update(image,fresh,number+6)
                self.assertEqual((observation['source_frame'],observation['captured_at']),(number,2.))

    def test_shared_scene_owner_does_not_replace_the_standalone_person_tracker_cut(self):
        from traffic_pedestrian import PersonTracker
        tracker=PersonTracker()
        image=np.full((360,640,3),30,np.uint8)
        tracker.update(image,dict(frame=1,captured_at=1.,people=[],shape=image.shape[:2]),1)
        self.assertIsNone(tracker.update(np.full_like(image,160),None,2))

    def test_round_robin_clear_evidence_is_bounded_and_never_releases_a_stale_hazard(self):
        from traffic_pedestrian import PedestrianGuard
        def observation(number,captured,people=()):
            return dict(frame=number,captured_at=captured,people=list(people),shape=(360,640))
        guard=PedestrianGuard(max_age=1.5)
        guard.update(observation(1,0.),0.)
        clear=observation(2,.4)
        self.assertEqual(guard.update(clear,.5)['status'],'CLEAR')
        self.assertEqual(guard.update(clear,1.6)['status'],'CLEAR')
        self.assertEqual(guard.update(clear,2.)['status'],'UNKNOWN')
        guard.update(observation(3,2.,[[290,170,350,310,.9,0]]),2.)
        self.assertEqual(guard.update(observation(4,2.1),4.)['status'],'BRAKE')
        self.assertEqual(PedestrianGuard().max_age,1.)


    def test_all_sources_are_processed_with_independent_boxes_and_both_hdmi_tiles_render(self):
        import traffic_light_3way_v2_modified as app
        from traffic_rknn import PCIeCapture
        calls=[];rendered=[];decisions={}
        class Capture:
            def __init__(self):
                self.source_channel=self.capture_mode=0
                self.dma_ms=self.convert_ms=0.
                self.screen_executor=None
                self.video_status=dict(full_hdmi_capture=True,full_camera_capture=True,native_ready=True,
                    native_source=0,valid_mask=15,source_channels=(0,1,2,2),good_frames=(1,1,1,1))
            def select_native_source(self,channel):
                self.source_channel=channel;self.video_status=dict(self.video_status,native_source=channel)
            def read(self,full_hdmi=False):
                time.sleep(.002);self.capture_mode=int(full_hdmi)
                return True,np.zeros((720,1280,3),np.uint8)
            def get(self,key):
                return {cv2.CAP_PROP_FRAME_WIDTH:1280,cv2.CAP_PROP_FRAME_HEIGHT:720,cv2.CAP_PROP_FPS:30}.get(key,0)
            def isOpened(self):return True
            def release(self):pass
        class Analyzer:
            def __init__(self,*args):pass
            def __call__(self,image,args,number,loop,revision,captured,*rest):
                channel=args.front_channel;calls.append(channel)
                if channel==2:
                    assert args.signal_roi is None
                    assert rest[-2:]==((0.,.5,.5,1.),(720,1280))
                x=(channel%2)*640;y=(channel//2)*360
                color=('green','red','yellow')[channel]
                return dict(frame=number,loop=loop,group_revision=revision,captured_at=captured,
                    processing_ms=1.,signal_roi=None,image=np.zeros((720,1280,3),np.uint8),
                    lights=[dict(box=[x+20,y+20,x+40,y+60],direction='STRAIGHT',color=color,color_conf=1.)],
                    states={'STRAIGHT':color},raw_states={'STRAIGHT':color})
        def current(result,*args,**kwargs):
            return (result['lights'],result['states']) if result else ([],{})
        def render(image,lights,states,simulation,*args,**kwargs):
            rendered.append((image.shape,lights))
        def arrow(image,simulation,*args,**kwargs):
            decisions[simulation.decision]=simulation.snapshot()
        def key(delay):
            time.sleep(.005)
            return ord('q') if rendered and len(rendered[-1][1])==4 else -1
        with tempfile.TemporaryDirectory() as folder,ExitStack() as stack:
            root=Path(folder);weights=root/'color.rknn';weights.touch()
            options=['traffic','--pcie-capture','--pipeline-capture','--assist-mode','fusion','--show','--no-record',
                '--fpga-pci','0002:21:00.0',
                '--front-channel','0','--weights',str(weights),'--direction-weights',str(root/'direction.rknn'),
                '--lane-weights',str(root/'lane.rknn'),'--signal-fps','1000','--auxiliary-fps','1000',
                '--moving','--max-display-frames','100','--output',str(root/'result.mp4')]
            stack.enter_context(patch('sys.argv',options))
            stack.enter_context(patch('traffic_rknn.RKNNModel'))
            stack.enter_context(patch.object(app,'DirectionClassifier'))
            stack.enter_context(patch.object(app,'FrameAnalyzer',Analyzer))
            factory=stack.enter_context(patch('traffic_rknn.PCIeCapture',return_value=Capture()))
            factory.channel_bounds.side_effect=PCIeCapture.channel_bounds
            stack.enter_context(patch('traffic_lane.LaneModel')).return_value.infer.side_effect=lambda image,n,t:dict(
                frame=n,captured_at=t,boundaries={},processing_ms=1.)
            stack.enter_context(patch('traffic_lane.LaneEvidenceTracker')).return_value.update.return_value={}
            stack.enter_context(patch.object(app,'current_observation',side_effect=current))
            bridge=stack.enter_context(patch('traffic_fpga.FPGAInteraction')).return_value
            bridge.poll.return_value=None;bridge.publish.return_value=0xa55a0051
            stack.enter_context(patch('traffic_fpga.CPUMotionKey')).return_value.poll.return_value=None
            stack.enter_context(patch.object(app,'draw_overlay',side_effect=render))
            stack.enter_context(patch.object(app,'draw_driving_arrow',side_effect=arrow))
            for name in ('namedWindow','moveWindow','setMouseCallback','imshow','destroyAllWindows'):
                stack.enter_context(patch('cv2.'+name))
            stack.enter_context(patch('cv2.getWindowProperty',return_value=1))
            stack.enter_context(patch('cv2.waitKey',side_effect=key))
            stack.enter_context(redirect_stdout(io.StringIO()))
            app.run(app.parse_opt())
        self.assertEqual(set(calls),{0,1,2})
        self.assertIn('ALARM_RED',decisions)
        self.assertIn('WAIT_YELLOW',decisions)
        self.assertTrue(any(call.args[0].alarm for call in bridge.publish.call_args_list))
        shape,lights=rendered[-1]
        boxes=[light['box'] for light in lights]
        self.assertEqual(len(boxes),4)
        for expected,actual in zip(([20,20,40,60],[660,20,680,60],[20,380,40,420],[660,380,680,420]),boxes):
            np.testing.assert_allclose(actual,np.array(expected)*shape[1]/1280,atol=1.1)

    def test_real_hdmi_analyzer_maps_coarse_pixels_to_the_lower_tile_of_the_full_canvas(self):
        from types import SimpleNamespace
        from traffic_light_3way_v2_modified import FrameAnalyzer
        image=np.full((360,640,3),123,np.uint8)
        args=SimpleNamespace(pcie_capture=True,front_channel=2,signal_roi=(0,.5,.5,1),
            direction_centers=None,circle_direction=None,confirm_frames=1,top_ratio=1,tiles=1,
            iou_thres=.45,show=True)
        classifier=MagicMock();classifier.predict.return_value=[('STRAIGHT',.99)]
        analyzer=FrameAnalyzer(SimpleNamespace(names={0:'red'}),classifier,None)
        with patch('traffic_light_3way_v2_modified.detect_roi',return_value=[[100,40,120,80,.98,0]]):
            result=analyzer(image,args,10,0,0,1.,native_bounds=(0,.5,.5,1),display_shape=(720,1280))
        self.assertEqual(result['lights'][0]['box'],[100,400,120,440])
        self.assertEqual(result['image'].shape,(720,1280,3))
        self.assertTrue(np.all(result['image'][360:,:640]==123))
        self.assertTrue(np.all(result['image'][:360]==0))

    def test_hdmi_new_scene_discards_direction_identity_from_the_previous_video(self):
        from types import SimpleNamespace
        from traffic_light_3way_v2_modified import FrameAnalyzer
        image=np.full((360,640,3),123,np.uint8)
        args=SimpleNamespace(pcie_capture=True,front_channel=2,signal_roi=(0,.5,.5,1),
            direction_centers=None,circle_direction=None,confirm_frames=1,top_ratio=1,tiles=1,
            iou_thres=.45,show=True,camera_generation=0)
        classifier=MagicMock();classifier.predict.side_effect=[[('STRAIGHT',.99)],[]]
        analyzer=FrameAnalyzer(SimpleNamespace(names={0:'red'}),classifier,None)
        with patch('traffic_light_3way_v2_modified.detect_roi',side_effect=[[[100,40,120,80,.98,0]],[]]):
            analyzer(image,args,1,0,0,1.,native_bounds=(0,.5,.5,1),display_shape=(720,1280))
            self.assertTrue(analyzer.direction_history.tracks)
            args.camera_generation=1
            result=analyzer(image,args,2,0,0,1.1,native_bounds=(0,.5,.5,1),display_shape=(720,1280))
        self.assertEqual(result['lights'],[])
        self.assertEqual(analyzer.direction_history.tracks,[])

    def test_multisource_hdmi_accepts_a_confident_identity_without_changing_single_source_confirmation(self):
        from types import SimpleNamespace
        from traffic_light_3way_v2_modified import FrameAnalyzer
        image=np.full((360,640,3),123,np.uint8)
        def analyze(concurrent):
            args=SimpleNamespace(pcie_capture=True,front_channel=2,signal_roi=None,
                direction_centers=None,circle_direction=None,confirm_frames=2,top_ratio=1,tiles=1,
                iou_thres=.45,show=True)
            if concurrent:args.camera_generation=0
            classifier=MagicMock();classifier.predict.return_value=[('STRAIGHT',.99)]
            analyzer=FrameAnalyzer(SimpleNamespace(names={0:'red'}),classifier,None)
            with patch('traffic_light_3way_v2_modified.detect_roi',return_value=[[100,40,120,80,.98,0]]):
                return analyzer(image,args,1,0,0,1.)['lights'][0]['direction']
        self.assertEqual(analyze(True),'STRAIGHT')
        self.assertEqual(analyze(False),'UNKNOWN')


if __name__=='__main__':unittest.main()
