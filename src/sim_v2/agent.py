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
    
    def set_route(self, start, goal):
        start = self.planner.get_node_by_desc(start)
        to = self.planner.get_node_by_desc(goal)
        route = self.planner.get_route(start, to)
        self.route_segments = self.planner.split_route(route)
        self.traj_segments = [self.planner.get_trajectory(s) for s in self.route_segments]
        self.current_segment = 0
    
    def process_atc(self):
        self.atc.message # TODO Process ATC commands
        
    def update_belief(self, obs):
        self.belief = obs # TODO belief update?
        
    def step(self):
        # Sense
        gps = self.GPS_sensor.sense()
        img = self.camera_sensor.sense()
        t = self.timer.time()
        
        # Speed and heading
        
        
        
        obs = {**gps, "t" : t}
        self.update_belief(obs)
        
        # Process ATC commands
        self.process_atc()
        
        # Check to see if we are in range of a hold line
        check_for_holdline = self.map.check_for_holdline(self.belief)
        distance_to_holdline = math.inf
        if check_for_holdline:
            distance_to_holdline = self.holdline_detector.get_distance(img)
        
        # Get the low-level control
        control, done_segment = self.controller.get_control(self.belief, self.traj_segments[self.current_segment])
        
        # Increment the trajectory segment
        if done_segment:
            self.current_segment += 1
        
        # Act
        if control is not None:
            self.client.sendCTRL(control)

        # Update timestep 
        self.idx = self.idx + 1
        
        
        
        