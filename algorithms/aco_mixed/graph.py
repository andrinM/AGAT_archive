import numpy as np
import pandas as pd 
from sklearn.metrics.pairwise import pairwise_distances


class Graph:
    def __init__(self,df):
        self.df = df
        self.df_hom = df.filter(regex="^hom").copy()
        self.df_het = df.filter(regex="^het").copy()
        self.df_hot_hom = df.filter(regex="^hot_hom").copy()
        self.df_hot_het = df.filter(regex="^hot_het").copy()
        self.euclidean_matrix = self.euclidean_distance_matrix()
        self.het_matrix = self.het_distance_matrix()
        self.hot_hom_matrix = self.hot_hom_distance_matrix()
        self.hot_het_matrix = self.hot_het_distance_matrix()
        self.num_nodes = self.df.shape[0]  # Number of nodes (group members)
        # Initialize pheromones for each path between nodes (same size as distances)
        self.pheromones = np.ones((self.num_nodes, self.num_nodes))# Start with equal pheromones


    def euclidean_distance_matrix(self):
        if self.df_hom.empty:
            return None
        euclidean_dist = pairwise_distances(self.df_hom, metric="euclidean")
        mask = np.eye(euclidean_dist.shape[0], dtype=bool)  # Diagonalmaske
        euclidean_dist[~mask & (euclidean_dist == 0)] = 0.001
        return euclidean_dist


    def het_distance_matrix(self):
        if self.df_het.empty: 
            return None 
        features_array = np.array(self.df_het)
        max_deviations = np.max(features_array, axis=0) - np.min(features_array, axis=0)
        print("max_deviation", max_deviations)
        num_points = features_array.shape[0]
        dist_matrix = np.zeros((num_points, num_points))
        for i in range(num_points):
            for j in range(i + 1, num_points):
                inverse_diffs = abs(np.abs(features_array[i] - features_array[j])- max_deviations)
                distance = np.sqrt(np.sum(inverse_diffs ** 2))
                if distance == 0 and i != j:
                    dist_matrix[i, j] = dist_matrix[j, i] = 0.001
                else:
                    dist_matrix[i, j] = dist_matrix[j, i] = distance
        return dist_matrix
    
    # Hamming distances for categorial features 
    def hot_hom_distance_matrix(self):
        if self.df_hot_hom.empty:
            return None
        features_array = np.array(self.df_hot_hom)
        num_points = features_array.shape[0]
        hamming_matrix = np.zeros((num_points, num_points))
        for i in range(num_points):
            for j in range(num_points):
                dist = sum(x != y for x,y in zip(features_array[i],features_array[j]))
                if dist == 0 and i !=j:
                    hamming_matrix[i, j] = hamming_matrix[j,i] = 0.001
                else: 
                    hamming_matrix[i, j] = hamming_matrix[j,i] = dist      
        return hamming_matrix
  
    def hot_het_distance_matrix(self):
        if self.df_hot_het.empty:
            return None
        features_array = np.array(self.df_hot_het)
        num_points = features_array.shape[0]
        hamming_matrix = np.zeros((num_points, num_points))
        for i in range(num_points):
            for j in range(num_points):
                dist = sum(x == y for x,y in zip(features_array[i],features_array[j]))
                if dist == 0 and i !=j:
                    hamming_matrix[i, j] = hamming_matrix[j,i] = 0.001
                else: 
                    hamming_matrix[i, j] = hamming_matrix[j,i] = dist      
        return hamming_matrix

