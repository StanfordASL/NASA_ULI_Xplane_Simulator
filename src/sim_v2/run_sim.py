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
XPC3_DIR = os.path.join(NASA_ULI_ROOT_DIR, "src")
sys.path.append(XPC3_DIR)

# Import julia stuff to access trained neural network
import julia
from julia import Main
Main.julia_file = "load_nn.jl"
Main.eval("include(julia_file)")

def main():
    with xpc3.XPlaneConnect() as client:
        client.sendDREF("sim/flightmodel/controls/parkbrake", False)
        time.sleep(2.0)
        
        # Construct the agent
        DATA_DIR = os.path.join(XPC3_DIR, "sim_v2", "data")
        map = POIMap(os.path.join(DATA_DIR, "grant_co_pois.csv"))
        atc_listener = ATCAgent(client)
        planner = FixedWaypointPlanner(os.path.join(DATA_DIR, "waypoints.csv"))
        
        controller = OpenLoopController(client, os.path.join(DATA_DIR, "openloop_control.csv"))
        holdline_detector = HoldLineDetector()
        GPS_sensor = GPSSensor(client, 0.0001, 0.0001, 0.0001)
        camera_sensor = CameraSensor(64, 32, save_sample_screenshot=True, monitor_index=2)
        timer = Timer(client)
        agent = TaxiAgent(client, map, atc_listener, planner, controller, holdline_detector, GPS_sensor, camera_sensor, timer)
        
        agent.set_route("Gate 3", "4 takeoff")
    
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
