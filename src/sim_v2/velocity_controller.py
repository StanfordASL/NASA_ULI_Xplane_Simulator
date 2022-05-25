import math

class VelocityController:
    def __init__(self, client, v0, k):
        self.client=client
        self.v0 = v0
        self.k = k
    
    def get_control(self,  belief, waypoints):
        v = self.client.getDREF("sim/flightmodel/position/groundspeed")[0]
        u = math.tanh(-self.k*(v - self.v0))
        
        return [0, 0, 0, u]