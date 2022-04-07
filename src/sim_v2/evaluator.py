import csv
import math
class Evaluator:
    def __init__(self, client):
        self.client=client
        self.filename = "sim_log.csv"

        # Setup CSV
        fieldnames = ["idx", "t", "latitude", "longitude", "altitude", "check_for_holdline", "distance_to_holdline"]
        with open(self.filename, "w") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(fieldnames)
    
    def log(self, agent):
        check_for_holdline = agent.map.check_for_holdline(agent.belief)
        distance_to_holdline = math.inf
        if check_for_holdline:
            distance_to_holdline = agent.holdline_detector.get_distance(agent.camera_sensor.sense())
        
        row = [agent.idx, agent.belief["t"], agent.belief["Latitude"], agent.belief["Longitude"], agent.belief["Altitude"], check_for_holdline, distance_to_holdline]
        with open(self.filename, "a") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(row)