import csv
import math
import numpy as np

class Evaluator:
    def __init__(self, client):
        self.client=client
        self.filename = "sim_log.csv"

        # Setup CSV
        fieldnames = ["idx", "t", "latitude", "longitude", "altitude", "speed", "check_for_holdline", "distance_to_holdline", "true_distance_to_holdline"]
        with open(self.filename, "w") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(fieldnames)
    
    def log(self, agent):
        # Check again for holdline (this is currently pretty redundant)
        check_for_holdline = agent.map.check_for_holdline(agent.belief)
        distance_to_holdline = math.inf
        if check_for_holdline:
            distance_to_holdline = agent.holdline_detector.get_distance(agent.camera_sensor.sense())

        # Get latitude and longitude of actual holdline - from Sydney's spreadsheet
        lat_line = 47.196231
        lon_line = -119.332108

        # Find true distance to holdline
        true_distance_to_holdline = agent.map.distance_to_holdline(agent.belief)
        
        # Get speed
        speed = agent.client.getDREF("sim/flightmodel/position/groundspeed")[0]
        
        # Write sim output to csv
        row = [agent.idx, agent.belief["t"], agent.belief["Latitude"], agent.belief["Longitude"], agent.belief["Altitude"], speed, check_for_holdline, distance_to_holdline, true_distance_to_holdline]
        with open(self.filename, "a") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(row)