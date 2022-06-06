from agent import TaxiAgent
from sensors import * 
from atc import *
from evaluator import *
from poi_map import *
from holdline_detector import *
from trivial_controllers import *
from lookahead_controller import LookaheadController
from planner import *
from velocity_controller import *
from utils import gps

import sys
import time
import os
NASA_ULI_ROOT_DIR = os.environ['NASA_ULI_ROOT_DIR']
XPC3_DIR = os.path.join(NASA_ULI_ROOT_DIR, "src")
sys.path.append(XPC3_DIR)
 
import xpc3

# set the route
route_start = "Gate 3"
route_goal = "4 takeoff"

def main():
    with xpc3.XPlaneConnect() as client:
        client.sendDREF("sim/flightmodel/controls/parkbrake", False)
        coordinate_converter = gps.CoordinateConverter.from_sim(client)
        time.sleep(2.0)
        
        # Construct the agent
        DATA_DIR = os.path.join(XPC3_DIR, "sim_v2", "data")
        map = POIMap(os.path.join(DATA_DIR, "grant_co_pois.csv"))
        atc_listener = ATCAgent(client)
        planner = GraphPlanner(coordinate_converter, os.path.join(DATA_DIR, "grant_co_map.csv"), interp="linear")
        controller = TeleportController(client, 100)
        # planner = GraphPlanner(coordinate_converter, os.path.join(DATA_DIR, "grant_co_map.csv"), interp="spline")
        # controller = LookaheadController(client)
        holdline_detector = HoldLineDetector(os.path.join(DATA_DIR, "model_weights.pth"))
        GPS_sensor = GPSSensor(client, 0.0000, 0.0000, 0.0000)
        camera_sensor = CameraSensor(64, 32, save_sample_screenshot=True, monitor_index=0)
        timer = Timer(client)
        agent = TaxiAgent(client, map, atc_listener, planner, controller, holdline_detector, GPS_sensor, camera_sensor, timer)
        
        agent.set_route(route_start, route_goal)
    
        # Construct the evaluator
        evaluator = Evaluator(client)
        max_time = 1000

        # Main loop
        while timer.time() < max_time:
            agent.step()
            evaluator.log(agent)
            time.sleep(0.01)
            
        
        
        
if __name__ == "__main__":
    main()
