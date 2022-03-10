import pandas

class POIMap:
    def __init__(self, filename):
        self.pois = pandas.read_csv(filename)
        
    def check_for_holdline(self, belief):
        return False #TODO
        