import pandas
import numpy as np
import time

# Applies a fixed sequence of controls that are read in from a CSV file
class OpenLoopController:
    def __init__(self, client, controls_csv_file):
        self.client=client
        self.controls = pandas.read_csv(controls_csv_file)
        self.control_labels = ["Latitudinal Stick", "Longitudinal Stick", "Rudder Pedals", "Throttle", "Gear Action", "Flaps", "Speedbrakes"]
        self.times = (self.controls["Time"] - self.controls["Time"][0]).to_numpy()
    
    def get_control(self, belief, trajectory):
        t = belief['t']
        index = np.argmax(self.times > t) - 1
        controls = self.controls.iloc[index][self.control_labels].to_numpy().tolist()
        controls[4] = int(controls[4])
        
        return controls, False

# Teleports the aircraft to the trajectory waypoints in sequence
class TeleportController:
    def __init__(self, client, resolution=10):
        self.client=client
        self.us = np.linspace(0,1,resolution)
        self.ui = 0
    
    def get_control(self, belief, traj):
        self.client.pauseSim(True)
        lat, lon, head = traj(self.us[self.ui])
        if head < 0:
            head += 360.0
            
        time.sleep(.1)
        curr_agly = self.client.getDREF("sim/flightmodel/position/y_agl")[0]
        curr_localy = self.client.getDREF("sim/flightmodel/position/local_y")[0]
        self.client.sendDREF("sim/flightmodel/position/local_y",
                        curr_localy - curr_agly)
        
        self.client.sendPOSI([lat, lon, -998, 0, 0, head])
        done = self.us[self.ui] == 1
        if done:
            self.ui = 0
        else:
            self.ui += 1
        return None, done
            
