import pandas
import numpy as np
import math

class POIMap:
    def __init__(self, filename):
        self.pois = pandas.read_csv(filename)

        # Location of holdline in GPS coordinates
        self.lat_line = 47.196231
        self.lon_line = -119.332108
        self.alt_line = 361.8233947753906
        self.pitch_line = 1.227753758430481
        self.roll_line = 0.22528982162475586
        self.heading_line = 325.41326904296875
        self.line = [47.196231,-119.332108, 361.8233947753906, 1.227753758430481, 0.22528982162475586, 325.41326904296875, 1.0]

        # Location of holdline in local coordinates
        self.x_line = -25122.07421875
        self.z_line = 33763.4296875

        # Starting location at gate 3 in GPS coordinates
        self.lat_gate4 = 47.1898307800293
        self.lon_gate4 = -119.333251953125
        self.alt_gate4 = 364.3702697753906
        self.pitch_gate4 = 0.9692096710205078
        self.roll_gate4 = -0.019080888479948044
        self.heading_gate4 = 89.95248413085938
        self.gate4 = [47.1898307800293, -119.333251953125, 364.3702697753906, 0.9692096710205078, -0.019080888479948044, 89.95248413085938, 1.0]

    def haverside(self, lat1, lon1, lat2, lon2):
        # Distance between latitudes and longitudes in radians
        dLat = (lat2 - lat1) * math.pi / 180.0
        dLon = (lon2 - lon1) * math.pi / 180.0

        # Convert to original lats to radians
        lat1 = (lat1) * math.pi / 180.0
        lat2 = (lat2) * math.pi / 180.0

        # Compute haversine stuff
        a = (pow(math.sin(dLat / 2), 2) + pow(math.sin(dLon / 2), 2) * math.cos(lat1) * math.cos(lat2))
        rad = 6371
        c = 2 * math.asin(math.sqrt(a))
        return rad * c * 1000

    def distance_to_holdline(self, belief):
        # Get current latitude and longitude
        lat = belief["Latitude"]
        lon = belief["Longitude"]

        # Compute haverside distance to holdine
        d = self.haverside(lat, lon, self.lat_line, self.lon_line)

        return d

        
    def check_for_holdline(self, belief):
        # Get current latitude and longitude
        # lat = belief["Latitude"]
        # lon = belief["Longitude"]

        # # Get latitude and longitude of actual holdline - from Sydney's spreadsheet
        # lat_line = 47.196231
        # lon_line = -119.332108

        # # Haversine formula for distance
        # d = self.haverside(lat, lon, lat_line, lon_line)

        # Check if we are within range of holdline
        if self.distance_to_holdline(belief) < 30:
            check = True
        else:
            check = False

        return check
        