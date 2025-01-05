import numpy as np
import pandas as pd 
import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from distance_matrix.distance_matrix import get_distance_matrix

class Graph:
    def __init__(self,df, weights = {}, max_deviations = {}):
        self.df = df.copy()
        self.num_nodes = self.df.shape[0]  # Number of nodes (group members)
        self.distance_matrix = get_distance_matrix(df1 = self.df, df2 = self.df, weights = weights, max_deviations = max_deviations)
        # Initialize pheromones for each path between nodes (same size as distances)
        self.pheromones = np.ones((self.num_nodes, self.num_nodes))# Start with equal pheromones






