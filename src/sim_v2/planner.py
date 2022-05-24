import pandas
import scipy.interpolate as interp

class Node:
    def __init__(self, row):
        self.lat = row['Latitude']
        self.lon = row['Longitude']
        self.head = row['Heading']
        self.label = row['Label']
        self.desc = row['Desc']
        self.neighbors = eval(row['Neighbors'].replace(";", ","))
        
    def is_stop(self):
        return self.label in ["hold short", "takeoff"]
        
        
class GraphPlanner:
    def __init__(self, graphfile):
        df = pandas.read_csv(graphfile)
        self.graph = [Node(df.iloc[i]) for i in range(len(res))]
        
    def __getitem__(self, index):
        return self.graph[index]
    
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
    
    def get_node_by_desc(self, desc):
        for n in range(len(self.graph)):
            if self[n].desc == desc:
                return n
        raise Exception("No node with name " + name)
        
    def get_trajectory(self, route):
        #TODO
        return None

g = GraphPlanner("src/sim_v2/data/grant_co_map.csv")
g.graph[0].desc

g.get_node_by_desc("Gate 3")

rt = g.get_route(0, 4)

g.split_route(rt)


import numpy as np 

phi = np.linspace(0, 2.*np.pi, 40)
r = 0.5 + np.cos(phi)         # polar coords
x, y = r * np.cos(phi), r * np.sin(phi)

from scipy.interpolate import splprep, splev
tck, u = splprep([x, y], s=0)
new_points = splev(u, tck)


import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(x, y, 'ro')
ax.plot(new_points[0], new_points[1], 'r-')
fig
