import numpy as np

# Graph class represents the environment where ants will travel
class Graph:
    def __init__(self,df):
        self.df = df
        # Initialize distance matrix 
        self.euclidean_dist = self.euclidean_distance_matrix(self.df)
        self.num_nodes = len(self.euclidean_dist)  # Number of nodes (group members)
        # Initialize pheromones for each path between nodes (same size as distances)
        self.pheromones = np.ones_like(self.euclidean_dist, dtype=float)  # Start with equal pheromones

    # adapted functioon to compute Euclidean distance matrix for data points
    # if x !=y and distance(x,y) = 0 the distance is set to 0.001 to avoid dividing by 0
    def euclidean_distance_matrix(self,df): 
        # assuming that the first column contains the names from members 
        df_features = df.iloc[:,1:]
        # convert to np array 
        features_array = np.array(df_features)
       
        num_points = features_array.shape[0]
        dist_matrix = np.zeros((num_points, num_points))  # Initialize a square matrix
        for i in range(num_points):
            for j in range(i + 1, num_points):
            # Calculate distance between points i and j
                distance = np.sqrt(np.sum((features_array[i] - features_array[j])**2))
                if distance == 0 and i != j:
                    dist_matrix[i, j] = dist_matrix[j, i] = 0.001
                else:
                    dist_matrix[i, j] = dist_matrix[j, i] = distance
        return dist_matrix