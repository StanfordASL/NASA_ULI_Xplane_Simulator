import pandas

class POIMap:
    def __init__(self, filename):
        self.pois = pandas.read_csv(filename)
        
    def check_for_holdline(self, belief):
        # Get current latitude and longitude
        lat = belief["Latitude"]
        lon = belief["Longitude"]

        # Get latitude and longitude of holdline sign
        lat_hold = self.pois.to_numpy()[1][1]
        lon_hold = self.pois.to_numpy()[1][2]

        # Get latitude and longitude of actual holdline
        lat_line = 47.19595887
        lon_line = -119.331832

        # Get distance between holdline and sign - need to refine this
        dist_sign_line = (lat_line - lat_hold)**2 + (lon_line - lon_hold)**2
        dist_to_line = (lat - lat_hold)**2 + (lon - lon_hold)**2

        # Check if we are closer to the line than the sign is
        if dist_to_line < 2*dist_sign_line:
            check = True
        else:
            check = False

        return check
        