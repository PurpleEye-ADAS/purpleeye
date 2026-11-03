import unittest
import cv2
import numpy as np
from traffic_navigation import NavigationReader,arrow_mask
from traffic_lane import FusionSimulation
from traffic_pedestrian import PedestrianGuard


def card(direction, shape=(180,200)):
    result=np.full((*shape,3),(130,70,10),np.uint8)
    size=round(min(shape)*.7)
    icon=cv2.resize(arrow_mask(direction),(size,size))
    y,x=shape[0]//12,(shape[1]-size)//2
    result[y:y+icon.shape[0],x:x+icon.shape[1]][icon>127]=245
    return result


class NavigationTests(unittest.TestCase):
    def test_navigation_worker_caches_fixed_and_auto_regions_separately(self):
        from unittest.mock import patch
        import traffic_navigation as navigation
        frame=np.full((360,640,3),100,np.uint8)
        frame[:180,:160]=card('LEFT',(180,160))
        frame[20:110,400:500]=card('RIGHT',(90,100))
        roi=(0.,0.,.25,.5)
        with patch.dict(navigation._camera_readers,{},clear=True):
            self.assertEqual(navigation._read_camera_navigation(frame,roi)[0],'LEFT')
            cached=navigation._camera_readers[roi]
            self.assertEqual(navigation._read_camera_navigation(frame,list(roi))[0],'LEFT')
            self.assertIs(navigation._camera_readers[roi],cached)
            self.assertEqual(navigation._read_camera_navigation(frame)[0],'UNKNOWN')
            self.assertIsNot(navigation._camera_readers[None],cached)

    def test_fixed_region_async_navigation_keeps_confirmation_and_expiry(self):
        from concurrent.futures import Future
        from unittest.mock import MagicMock,patch
        executor=MagicMock();first,second=Future(),Future()
        executor.submit.side_effect=[first,second,Future()]
        roi=(0.,0.,1.,1.);reader=NavigationReader(roi,executor=executor);image=card('LEFT')
        with patch.object(reader,'classify',side_effect=AssertionError('must stay off the main loop')):
            self.assertEqual(reader.update(image,1.,1.)['direction'],'UNKNOWN')
        self.assertFalse(first.done())
        function,pixels,region=executor.submit.call_args.args
        self.assertEqual(region,roi);self.assertIsNot(pixels,image)
        first.set_result(function(pixels,region))
        self.assertEqual(reader.update(image,1.21,1.21)['direction'],'UNKNOWN')
        function,pixels,region=executor.submit.call_args.args
        second.set_result(function(pixels,region))
        self.assertEqual(reader.update(image,1.42,1.42)['direction'],'LEFT')
        self.assertEqual(reader.update(image,1.42,1.9)['direction'],'UNKNOWN')

    def test_camera_async_route_does_not_wait_and_expired_work_cannot_authorize_motion(self):
        from concurrent.futures import Future
        from unittest.mock import MagicMock
        executor=MagicMock();first,second,third=Future(),Future(),Future()
        executor.submit.side_effect=[first,second,third,Future()]
        reader=NavigationReader(None,executor=executor);image=card('LEFT')
        self.assertEqual(reader.update(image,1.,1.)['direction'],'UNKNOWN')
        self.assertFalse(first.done())
        first.set_result(('LEFT',.9,2.))
        self.assertEqual(reader.update(image,1.21,1.21)['direction'],'UNKNOWN')
        second.set_result(('LEFT',.9,2.))
        self.assertEqual(reader.update(image,1.42,1.42)['direction'],'LEFT')
        third.set_result(('RIGHT',.9,2.))
        self.assertEqual(reader.update(image,1.9,1.9)['direction'],'UNKNOWN')


    def test_camera_auto_card_needs_no_fixed_location_and_rejects_conflicting_cards(self):
        reader=NavigationReader(None)
        for direction in ('LEFT','STRAIGHT','RIGHT'):
            for x,y in ((20,16),(380,90)):
                frame=np.full((360,640,3),100,np.uint8)
                frame[y:y+90,x:x+100]=card(direction,(90,100))
                self.assertEqual(reader.classify(frame)[0],direction)
        frame=np.full((360,640,3),100,np.uint8)
        frame[20:110,20:120]=card('LEFT',(90,100))
        frame[20:110,400:500]=card('RIGHT',(90,100))
        self.assertEqual(reader.classify(frame)[0],'UNKNOWN')
        self.assertEqual(reader.classify(np.full_like(frame,170))[0],'UNKNOWN')

    def test_camera_accepts_bright_saturated_navigation_background(self):
        reader=NavigationReader(None)
        for direction in ('LEFT','STRAIGHT','RIGHT'):
            image=card(direction,(90,100))
            image[np.all(image==(130,70,10),axis=2)]=(255,80,0)
            frame=np.full((360,640,3),100,np.uint8)
            frame[20:110,20:120]=image
            self.assertEqual(reader.classify(frame)[0],direction)

    def test_software_direction_echo_is_distinct_from_next_physical_key(self):
        import struct
        from traffic_fpga import FPGAInteraction,MAGIC,OFFSET
        bridge=FPGAInteraction.__new__(FPGAInteraction)
        bridge.read_map=bytearray(0x2000);bridge.last_control=(1,20)
        bridge.software_control=(2,21)
        struct.pack_into('<4I',bridge.read_map,OFFSET,MAGIC,2,21,0)
        self.assertIsNone(bridge.poll())
        self.assertTrue(bridge.software_echo)
        struct.pack_into('<4I',bridge.read_map,OFFSET,MAGIC,0,22,0)
        self.assertEqual(bridge.poll()['navigation'],'LEFT')
        self.assertFalse(bridge.software_echo)

    def fpga_main_controls(self,navigation=False,control=1,sequence=10):
        import ast,struct
        from pathlib import Path
        from types import SimpleNamespace
        from traffic_fpga import FPGAInteraction,MAGIC,OFFSET
        class Hardware:
            def __init__(self):
                self.control=control;self.sequence=sequence;self.echo=0
                self.memory=bytearray(0x2000);self.expected=SimpleNamespace(value=0);self.update()
            def update(self):
                struct.pack_into('<4I',self.memory,OFFSET,MAGIC,self.control,self.sequence,self.echo)
            @property
            def value(self):return self.echo
            @value.setter
            def value(self,command):
                matches=self.expected.value==self.sequence
                if command>>16!=0xA55A or (command&64 and not
                        (matches and command&3!=3 and (command&128 or command&15==self.control))):return
                self.echo=command
                if matches and command&128 and command&3!=3:
                    self.control=command&15;self.sequence=(self.sequence+1)&0xffffffff
                self.update()
            def key(self,value):
                self.control=value;self.sequence=(self.sequence+1)&0xffffffff;self.update()
        source=(Path(__file__).resolve().parents[1]/'traffic_light_3way_v2_modified.py').read_text(encoding='utf-8')
        blocks=[node for node in ast.walk(ast.parse(source)) if isinstance(node,ast.If) and
                any(isinstance(child,ast.Assign) and any(isinstance(target,ast.Name) and
                    target.id=='previous_control' for target in child.targets) for child in node.body)]
        self.assertEqual(len(blocks),1)
        block=compile(ast.Module(body=blocks,type_ignores=[]),'actual_main_poll_block','exec')
        hardware=Hardware();bridge=FPGAInteraction.__new__(FPGAInteraction)
        bridge.read_map=hardware.memory;bridge.sequence_word=hardware.expected
        bridge.command_word=hardware;bridge.last_control=None
        simulation=FusionSimulation();simulation.navigation_enabled=navigation
        environment=dict(fpga=bridge,simulation=simulation,fusion=True)
        exec(block,environment)
        return hardware,bridge,simulation,lambda:exec(block,environment)

    def test_software_echo_cannot_overwrite_later_gui_direction_or_stop_in_main(self):
        for navigation in (False,True):
            for action in ('RIGHT','STOP'):
                with self.subTest(navigation=navigation,action=action):
                    hardware,bridge,simulation,poll=self.fpga_main_controls(navigation)
                    simulation.command('LEFT');simulation.command('MOVE');bridge.publish(simulation,False)
                    simulation.command(action);expected=(simulation.nav,simulation.manual_moving)
                    poll()
                    self.assertTrue(bridge.software_echo)
                    self.assertEqual((simulation.nav,simulation.manual_moving),expected)

    def test_software_echo_keeps_sequence_bookkeeping_for_following_publish_and_wrap(self):
        for sequence in (10,0xffffffff):
            with self.subTest(sequence=sequence):
                hardware,bridge,simulation,poll=self.fpga_main_controls(sequence=sequence)
                simulation.command('LEFT');bridge.publish(simulation,False)
                self.assertIsNone(bridge.poll())
                expected=(sequence+1)&0xffffffff
                self.assertEqual((bridge.sequence,bridge.hardware_control,bridge.last_control),
                                 (expected,0,(0,expected)))
                simulation.command('STOP');bridge.publish(simulation,False)
                self.assertEqual(hardware.expected.value,expected)
                self.assertEqual(hardware.control,8)
                self.assertEqual(hardware.sequence,(expected+1)&0xffffffff)

    def test_initial_controls_and_real_keys_survive_echo_filter_without_initial_nav_override(self):
        for navigation in (False,True):
            with self.subTest(navigation=navigation):
                hardware,bridge,simulation,poll=self.fpga_main_controls(navigation,control=8)
                self.assertFalse(simulation.manual_moving)
                self.assertFalse(simulation.navigation_manual)
                self.assertEqual(simulation.nav,'STRAIGHT' if navigation else 'LEFT')
                simulation.command('RIGHT');simulation.command('MOVE');bridge.publish(simulation,False)
                hardware.key(12);poll()
                self.assertFalse(bridge.software_echo)
                self.assertEqual((simulation.nav,simulation.manual_moving),('LEFT',True))
                self.assertEqual(simulation.navigation_manual,navigation)
                hardware.key(8);poll()
                self.assertFalse(bridge.software_echo)
                self.assertEqual((simulation.nav,simulation.manual_moving),('LEFT',False))

    def test_camera_blue_card_touching_green_trees_keeps_its_own_boundary(self):
        reader=NavigationReader(None)
        for direction in ('LEFT','STRAIGHT','RIGHT'):
            frame=np.full((360,640,3),(40,100,50),np.uint8)
            frame[:90,:100]=card(direction,(90,100))
            self.assertEqual(reader.classify(frame)[0],direction)
            darker=np.rint(frame.astype(float)*.68).astype(np.uint8)
            self.assertEqual(reader.classify(darker)[0],direction)

    def test_compressed_scaled_arrows(self):
        reader=NavigationReader((0,0,1,1))
        for key in ('LEFT','STRAIGHT','RIGHT'):
            for size in ((180,200),(90,100),(300,320)):
                image=card(key,size)
                ok,data=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,55]);self.assertTrue(ok)
                predicted,score=reader.classify(cv2.imdecode(data,1))
                self.assertEqual(predicted,key,(key,size,score))

    def test_text_and_conflicting_arrows_do_not_grant_navigation(self):
        reader=NavigationReader((0,0,1,1))
        image=np.full((160,240,3),(130,70,10),np.uint8)
        cv2.putText(image,'250m',(15,100),0,1.8,(255,255,255),3)
        self.assertEqual(reader.classify(image)[0],'UNKNOWN')
        image=np.concatenate((card('LEFT'),card('RIGHT')),axis=1)
        self.assertEqual(reader.classify(image)[0],'UNKNOWN')

    def test_confirmation_current_capture_and_missing_arrow(self):
        reader=NavigationReader((0,0,1,1))
        self.assertEqual(reader.update(card('LEFT'),1,1)['direction'],'UNKNOWN')
        self.assertEqual(reader.update(card('LEFT'),1.21,1.21)['direction'],'LEFT')
        self.assertEqual(reader.update(card('RIGHT'),1.42,1.42)['direction'],'UNKNOWN')
        self.assertEqual(reader.update(card('RIGHT'),1.63,1.63)['direction'],'RIGHT')
        self.assertEqual(reader.update(card('RIGHT'),1.63,2.1)['direction'],'UNKNOWN')
        self.assertEqual(reader.update(np.zeros((100,100,3),np.uint8),2.4,2.4)['direction'],'UNKNOWN')

    def simulation(self):
        s=FusionSimulation();s.navigation_enabled=True;s.navigation_direction='LEFT';return s

    def test_route_turn_and_change_have_different_solid_line_rules(self):
        s=self.simulation();road={'LEFT':{'kind':'SOLID'}}
        self.assertEqual(s.update_fused(road,{},False),'GO_LEFT')
        s.command('CHANGE')
        self.assertEqual(s.update_fused(road,{},False),'ALARM_SOLID')
        self.assertFalse(s.guidance_visible)

    def test_route_cannot_override_red_unknown_signal_manual_stop_or_pedestrian_brake(self):
        s=self.simulation();road={'LEFT':{'kind':'DASHED'}}
        self.assertEqual(s.update_fused(road,{'LEFT':'red'},True),'STOP_RED')
        self.assertEqual(s.update_fused(road,{},True),'WAIT_UNKNOWN')
        self.assertEqual(s.update_fused(road,{'LEFT':'green'},True,'STOP'),'STOP_MANUAL')
        s.command('AUTO');s.update_fused(road,{'LEFT':'green'},True)
        PedestrianGuard().apply(s,{'status':'BRAKE'})
        self.assertEqual(s.decision,'BRAKE_PEDESTRIAN');self.assertFalse(s.guidance_visible)

    def test_manual_steering_override_and_resume_navigation(self):
        s=self.simulation();road={'LEFT':{'kind':'DASHED'},'RIGHT':{'kind':'DASHED'}}
        s.command('RIGHT');s.update_fused(road,{},False)
        self.assertEqual(s.nav,'RIGHT')
        s.command('AUTO');s.update_fused(road,{},False)
        self.assertEqual(s.nav,'LEFT')
        s.navigation_direction='UNKNOWN';s.update_fused(road,{},False)
        self.assertEqual(s.decision,'WAIT_NAVIGATION');self.assertFalse(s.guidance_visible)

    def test_hdmi_navigation_accepts_shared_manual_steering_and_auto_resume(self):
        primary=FusionSimulation();other=FusionSimulation();hdmi=self.simulation()
        primary.navigation_peers=(other,hdmi)
        self.assertTrue(primary.has_navigation_context())
        road={'LEFT':{'kind':'DASHED'},'RIGHT':{'kind':'DASHED'}}
        for action,direction,manual in (('RIGHT','RIGHT',True),('AUTO','LEFT',False)):
            primary.command(action)
            hdmi.requested_nav=primary.requested_nav;hdmi.navigation_manual=primary.navigation_manual
            hdmi.update_fused(road,{},False)
            self.assertEqual((hdmi.nav,hdmi.navigation_manual),(direction,manual))
        hdmi.navigation_enabled=False
        self.assertFalse(primary.has_navigation_context())
        other.navigation_enabled=True
        self.assertTrue(primary.has_navigation_context())

    def test_standalone_steering_without_navigation_does_not_latch_override(self):
        simulation=FusionSimulation()
        self.assertFalse(simulation.has_navigation_context())
        simulation.command('RIGHT')
        self.assertEqual(simulation.nav,'RIGHT')
        self.assertFalse(simulation.navigation_manual)

    def test_route_turn_does_not_require_visible_lane_paint(self):
        s=self.simulation()
        for direction in ('LEFT','STRAIGHT','RIGHT'):
            s.navigation_direction=direction
            for road in ({},{'LEFT':{'kind':'UNKNOWN'},'RIGHT':{'kind':'UNKNOWN'}}):
                self.assertEqual(s.update_fused(road,{},False),'GO_'+direction)
                self.assertTrue(s.guidance_visible)
        s.command('CHANGE')
        self.assertEqual(s.update_fused({}, {},False),'WAIT_UNKNOWN')
        self.assertFalse(s.guidance_visible)

    def test_navigation_without_lane_paint_publishes_valid_moving_command(self):
        from traffic_fpga import FPGAInteraction
        from types import SimpleNamespace
        s=self.simulation();s.update_fused({}, {},False)
        bridge=FPGAInteraction.__new__(FPGAInteraction)
        bridge.last_control=(1,0);bridge.hardware_control=1;bridge.source=0;bridge.sequence=0
        bridge.read_map=bytearray(0x2000)
        bridge.sequence_word=SimpleNamespace(value=0);bridge.command_word=SimpleNamespace(value=0)
        bridge.publish(s,s.guidance_visible)
        self.assertEqual(bridge.command_word.value&0x50,0x50)
        self.assertEqual(bridge.command_word.value&3,0)

    def test_missing_lane_paint_does_not_bypass_stop_conditions(self):
        s=self.simulation()
        self.assertEqual(s.update_fused({}, {'LEFT':'red'},True),'STOP_RED')
        self.assertEqual(s.update_fused({}, {'LEFT':'red'},True,'MOVE'),'ALARM_RED')
        self.assertFalse(s.guidance_visible)
        self.assertEqual(s.update_fused({}, {},False,'STOP'),'STOP_MANUAL')
        self.assertFalse(s.guidance_visible)
        s.command('AUTO');s.update_fused({}, {},False)
        PedestrianGuard().apply(s,{'status':'UNKNOWN'})
        self.assertFalse(s.guidance_visible)


if __name__=='__main__':unittest.main()
