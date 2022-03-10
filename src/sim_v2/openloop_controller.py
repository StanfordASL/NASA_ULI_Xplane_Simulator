import pandas
import numpy as np

class OpenLoopController:
    def __init__(self, client, controls_csv_file):
        self.client=client
        self.controls = pandas.read_csv(controls_csv_file)
        self.control_labels = ["Latitudinal Stick", "Longitudinal Stick", "Rudder Pedals", "Throttle", "Gear Action", "Flaps", "Speedbrakes"]
        self.times = (self.controls["Time"] - self.controls["Time"][0]).to_numpy()
    
    def get_control(self, belief, waypoints):
        t = belief['t']
        index = np.argmax(self.times > t) - 1
        controls = self.controls.iloc[index][self.control_labels].to_numpy().tolist()
        controls[4] = int(controls[4])
        
        return controls

class FixedWaypointPlanner:
    def __init__(self, filename):
        self.waypoints = pandas.read_csv(filename)
        
    def get_waypoints(self, map, belief, distance_to_holdline):
        return self.waypoints