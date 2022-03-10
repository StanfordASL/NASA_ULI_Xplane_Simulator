from agent import TaxiAgent
from sensors import * 
from atc import *
from evaluator import *
from poi_map import *
from holdline_detector import *
from openloop_controller import *

import sys
import time
import os
NASA_ULI_ROOT_DIR = os.environ['NASA_ULI_ROOT_DIR']
XPC3_DIR = NASA_ULI_ROOT_DIR + '/src/'
sys.path.append(XPC3_DIR)

import xpc3

def main():
    with xpc3.XPlaneConnect() as client:
        client.sendDREF("sim/flightmodel/controls/parkbrake", False)
        time.sleep(2.0)
        
        # Construct the agent
        map = POIMap(XPC3_DIR+"/sim_v2/data/grant_co_pois.csv")
        atc_listener = ATCAgent(client)
        planner = FixedWaypointPlanner(XPC3_DIR+"/sim_v2/data/waypoints.csv")
        controller = OpenLoopController(client, XPC3_DIR+"/sim_v2/data/openloop_control.csv")
        holdline_detector = HoldLineDetector()
        GPS_sensor = GPSSensor(client, 0.0001, 0.0001, 0.0001)
        camera_sensor = CameraSensor(64, 64, save_sample_screenshot=True, monitor_index=2)
        timer = Timer(client)
        agent = TaxiAgent(client, map, atc_listener, planner, controller, holdline_detector, GPS_sensor, camera_sensor, timer)
                
    
        # Construct the evaluator
        evaluator = Evaluator(client)
        max_time = 1000

        # Main loop
        while timer.time() < max_time:
            agent.step()
            evaluator.log()
            time.sleep(0.01)
            
        
        
        
if __name__ == "__main__":
    main()
