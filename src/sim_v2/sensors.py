import numpy as np
import cv2
import mss
import sys
import time
import os

NASA_ULI_ROOT_DIR = os.environ['NASA_ULI_ROOT_DIR']
XPC3_DIR = os.path.join(NASA_ULI_ROOT_DIR, "src")
sys.path.append(XPC3_DIR)
DATA_DIR = os.path.join(XPC3_DIR, "sim_v2", "data")

class GPSSensor:
    def __init__(self, client, noise_lat, noise_long, noise_alt):
        self.client = client
        self.noise_lat = noise_lat
        self.noise_long = noise_long
        self.noise_alt = noise_alt
        
    def sense(self):
        # get the state
        s = self.client.getPOSI()
        lat = s[0] + np.random.normal(0, self.noise_lat)
        lon = s[1] + np.random.normal(0, self.noise_long)
        alt = s[2] + np.random.normal(0, self.noise_alt)
        
        return {"Latitude":lat, "Longitude":lon, "Altitude":alt}
        

class CameraSensor:
    def __init__(self, width, height, monitor_index=0, save_sample_screenshot=False, save_filename="test.png"):
        self.width = width
        self.height = height
        self.screenshotter = mss.mss()
        self.monitor_index = monitor_index
        self.save = save_sample_screenshot
        self.save_filename = save_filename
        
    def sense(self, i=0):
        # img = cv2.cvtColor(np.array(self.screenshotter.grab(self.screenshotter.monitors[self.monitor_index])), cv2.COLOR_BGRA2BGR)[:, :, ::-1]
        img = cv2.cvtColor(np.array(self.screenshotter.grab(self.screenshotter.monitors[self.monitor_index])), cv2.COLOR_RGBA2BGR)[:, :, ::-1]
        img = img[530:, :, :]
        img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_AREA)

        # For now, just save the image to an output directory
        if self.save:
            cv2.imwrite(self.save_filename, img)
            # cv2.imwrite("data\\images\\img" + str(i) + ".png", img)
            cv2.imwrite(os.path.join(DATA_DIR, "images", "img" + str(i) + ".png"), img)
        
        return img

class Timer:
    def __init__(self, client):
        self.client = client
        self.t0 = self.client.getDREF("sim/time/local_time_sec")[0]
    
    def time(self):
        return self.client.getDREF("sim/time/local_time_sec")[0] - self.t0

class LocalSensor:
    def __init__(self, client):
        self.client = client

    def sense(self):
        x = get_local_x(self.client)
        z = get_local_z(self.client)
        heading = get_heading(self.client)
        v = get_ground_velocity(self.client)

        return {"x" : x, "z" : z, "Heading" : heading, "Velocity" : v}
        