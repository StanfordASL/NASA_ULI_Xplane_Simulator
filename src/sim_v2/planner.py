import pandas
from scipy.interpolate import splprep, splev
import csv

class Trajectory:
    def __init__(self, tck):
        self.tck=tck
        
    def __call__(self, u):
        pt = splev([u], tck)
        return [pt[0][0], pt[1][0]]
        

class Node:
    def __init__(self, row):
        self.lat = row['Latitude']
        self.lon = row['Longitude']
        self.head = row['Heading']
        if self.head > 180:
            self.head -= 360
        self.label = row['Label']
        self.desc = row['Desc']
        self.neighbors = eval(row['Neighbors'].replace(";", ","))
        
    def is_stop(self):
        return self.label in ["hold short", "takeoff"]
        
        
class GraphPlanner:
    def __init__(self, graphfile):
        df = pandas.read_csv(graphfile)
        self.graph = [Node(df.iloc[i]) for i in range(len(df))]
        
    def __getitem__(self, index):
        return self.graph[index]
    
    def get_coord(self, n):
        return [self[n].lat, self[n].lon]
        
    def get_coords(self, rt):
        return [self.get_coord(n) for n in rt]
    
    # Copied from https://www.geeksforgeeks.org/building-an-undirected-graph-and-finding-shortest-path-using-dictionaries-in-python/
    def get_route(self, start, goal):
        explored = []
        queue = [[start]]
         
        if start == goal:
            raise Exception("Same Node")
         
        while queue:
            path = queue.pop(0)
            node = path[-1]
             
            if node not in explored:
                for neighbour in self[node].neighbors:
                    new_path = list(path)
                    new_path.append(neighbour)
                    queue.append(new_path)
                     
                    if neighbour == goal:
                        return new_path
                explored.append(node)
        raise Exception("No path between these nodes")

    def split_route(self, rt):
        rts = [[]]
        for n in rt:
            rts[-1].append(n)
            if self[n].is_stop() and n != rt[-1]:
                rts.append([n])
        return rts
    
    def write_coords_csv(self, coords, filename):
        with open(filename, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['Latitude','Longitude'])
            for c in coords:
                writer.writerow(c)
                
    def write_route_csv(self, route, filename):
        self.write_coords_csv(self.get_coords(route), filename)
        
    def get_node_by_desc(self, desc):
        for n in range(len(self.graph)):
            if self[n].desc == desc:
                return n
        raise Exception("No node with name " + name)
        
    def get_trajectory(self, route, k=3):
        coords = self.get_coords(route)
        lats = [c[0] for c in coords]
        lons = [c[1] for c in coords]
        lats = (lats - mean(lats)) / (maximum(lats) - minimum(lats))
        tck, u = splprep([, ], s=0, k=5)
        return Trajectory(tck)

g = GraphPlanner("data/grant_co_map.csv")
g.graph[0].desc

start = g.get_node_by_desc("Gate 3")
to = g.get_node_by_desc("4 takeoff")

rt = g.get_route(start, to)

g.write_route_csv(rt, "data/test_tr.csv")

traj = g.get_trajectory(rt, k=1)

coords = g.get_coords(rt)
lats = [c[0] for c in coords]
lons = [c[1] for c in coords]
# lats = (lats - np.mean(lats)) / (np.max(lats) - np.min(lats))
# lons = (lons - np.mean(lons)) / (np.max(lons) - np.min(lons))
tck, u = splprep([lats, lons], s=0, k=2)
traj = Trajectory(tck)

import matplotlib.pyplot as plt
plt.figure()
us = np.linspace(0,1,100)
lats = [traj(u)[0] for u in us]
lons = [traj(u)[1] for u in us]

plt.plot([c[0] for c in coords], [c[1] for c in coords], 'x', lats, lons, 'r-')
fig

g.write_coords_csv([traj(u) for u in np.linspace(0,1,100)], "data/dense_tr.csv")
