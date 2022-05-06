import math
import queue

# This class describes the high-level agent that controls the taxiing process
class TaxiAgent:
    def __init__(self, client, map, atc, planner, controller, holdline_detector, GPS_sensor, camera_sensor, timer, initial_belief=None):
        
        self.client = client
        self.map = map
        self.planner = planner
        self.controller = controller
        self.holdline_detector = holdline_detector
        self.GPS_sensor = GPS_sensor
        self.camera_sensor = camera_sensor
        self.timer = timer
        if initial_belief is None:
            initial_belief=self.GPS_sensor.sense()
            
        self.belief = initial_belief
        self.atc = atc
        self.idx = 0
    
    def process_atc(self):
        self.atc.message # TODO Process ATC commands
        
    def update_belief(self, obs):
        self.belief = obs # TODO belief update?
        
    def step(self):
        # Sense
        gps = self.GPS_sensor.sense()
        img = self.camera_sensor.sense(self.idx)
        t = self.timer.time()
        
        
        obs = {**gps, "t" : t}
        self.update_belief(obs)
        
        # Process ATC commands
        self.process_atc()
        
        # Check to see if we are in range of a hold line
        check_for_holdline = self.map.check_for_holdline(self.belief)
        distance_to_holdline = math.inf
        if check_for_holdline:
            distance_to_holdline = self.holdline_detector.get_distance(img)
        
        # Get the set of waypoints
        waypoints = self.planner.get_waypoints(self.map, self.belief, distance_to_holdline)
        
        # Get the low-level control
        control = self.controller.get_control(self.belief, waypoints)
        
        # Act
        self.client.sendCTRL(control)

        # Update timestep 
        self.idx = self.idx + 1
        
        
        
        