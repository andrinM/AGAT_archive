import pandas as pd
import numpy as np
from sklearn import metrics
import pulp
from scipy.spatial.distance import cdist
from scipy.spatial.distance import squareform
import sys
import os
current_dir = os.getcwd()  
src_dir = os.path.abspath(os.path.join(current_dir, 'src'))  # Parent directory of import_destination (i.e., src)
sys.path.append(src_dir)
from Data import testData
from algorithms import functions
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.patches as mpatches
import random
import matplotlib.colors as mcolors
from matplotlib import cm
from constraints import preprocess_constraints

"""Test Data"""
df_A1 = testData.df_A1

ml = []
cl = []
neighborhoods = []
ml_graph, cl_graph, neighborhoods = preprocess_constraints(ml, cl, 30)
n_clusters = 10
group_size = 3
""" End of Test Data"""
 
# Calculate initial Cluster Centers based on neighborhoods

def kmeans_plus_plus(df, k, distance_func):
    """
    KMeans++ cluster initialization with unique centers.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing the data points.
        k (int): Number of clusters.
        distance_func (function): A custom distance function to calculate distances between points.
    
    Returns:
        list: List of indices of the selected initial cluster centers.
    """
    
    # Step 1: Randomly choose the first center from the data points
    n_samples = df.shape[0]
    first_center_idx = np.random.randint(0, n_samples)
    centers_indices = [first_center_idx]
    
    # Keep track of chosen indices to ensure uniqueness
    chosen_indices = set(centers_indices)
    
    # Step 2: Select the remaining k-1 centers
    for _ in range(1, k):
        distances = []
        
        # Calculate the distance of each point to the nearest existing center
        for idx, point in df.iterrows():
            min_dist = float('inf')
            for center_idx in centers_indices:
                center_point = df.iloc[center_idx]
                dist = distance_func(df_A1.loc[[point.name]], df_A1.loc[[center_point.name]])
                if dist < min_dist:
                    min_dist = dist
            distances.append(min_dist)
        
        # Convert distances to a numpy array
        distances = np.array(distances)
        
        # Compute probabilities proportional to the square of distances
        squared_distances = distances ** 2
        probabilities = squared_distances / squared_distances.sum()
        
        # Choose a new center based on the weighted probability distribution
        while True:
            new_center_idx = np.random.choice(range(n_samples), p=probabilities)
            if new_center_idx not in chosen_indices:
                centers_indices.append(new_center_idx)
                chosen_indices.add(new_center_idx)
                break  # Ensure that a unique point is selected
    
    return centers_indices

""" Transforms a dictionare to a 2d array and remove duplicates.
"""
def dict_to_array(dict):
    # Flatten the dictionary into a list of [key, value] pairs
    flattened_list = [[key, value] for key, values in dict.items() for value in values]

    # Convert the list to a numpy array (optional)
    flattened_array = np.array(flattened_list)

    # Use a set to store sorted pairs
    unique_pairs = {tuple(sorted(pair)) for pair in flattened_array}

    # Convert the set back to a 2D array
    unique_array = np.array([list(pair) for pair in unique_pairs])
    
    return unique_array

def check_cluster_match(df, index, cluster_number, cluster_dict):
    # Get the indices of rows where the "cluster" column matches the given cluster_number
    cluster_indices = df[df['cluster'] == cluster_number].index
    # Check if any of these indices are in the values of the provided index in the dictionary
    if any(idx in cluster_dict.get(index, []) for idx in cluster_indices):
        return True
    
    return False


### MAIN ###
if (len(neighborhoods) >= n_clusters):
    lengths = [len(inner_array) for inner_array in neighborhoods]

    # Step 1: Calculate the lengths of each inner array
    lengths = [len(inner_array) for inner_array in neighborhoods]

    # Step 2: Sort the indices of the 2D array by the length of the inner arrays in descending order
    sorted_indices = sorted(range(len(neighborhoods)), key=lambda x: lengths[x], reverse=True)
    
    # Step 3: Select the top k elements based on the sorted indices
    selected_neighborhoods = [neighborhoods[i] for i in sorted_indices[:n_clusters]]
    cluster_centers = functions.get_cluster_centers_from_neighborhoods(df_A1,selected_neighborhoods)
else:
    if(len(neighborhoods) != 0):
        neighborhood_centers = functions.get_cluster_centers_from_neighborhoods(df_A1, neighborhoods)
        data_points_to_remove = [item for sublist in neighborhoods for item in sublist]

        # Make a copy of the DataFrame to avoid modifying the original
        df_without_neighboors = df_A1.copy()
        df_without_neighboors = df_without_neighboors.drop(data_points_to_remove)

        k = n_clusters - len(neighborhoods)
        kmean_centers = df_A1.iloc[kmeans_plus_plus(df_without_neighboors,k, functions.distance_between_two_datapoints)]
        cluster_centers = pd.concat([neighborhood_centers, kmean_centers], ignore_index=True)
    if(len(neighborhoods) == 0):
        cluster_centers = kmeans_plus_plus(df_A1, n_clusters, functions.distance_between_two_datapoints)
        cluster_centers = pd.DataFrame(cluster_centers, columns= df_A1)
        print(cluster_centers)
# Convert all hots to arrays
hot_columns = [col for col in cluster_centers.columns if col.startswith('hot')]
for col in hot_columns:
    cluster_centers[col] = cluster_centers[col].apply(lambda x: np.array(x) if isinstance(x, tuple) else x)

ml_2d_array = dict_to_array(ml_graph)
cl_2d_array = dict_to_array(cl_graph)
ml_1d_array = ml_2d_array.flatten()
cl_1d_array = cl_2d_array.flatten()
distance_matrix = functions.get_distance_matrix(df_A1, cluster_centers)
# Assigne  data points to the mustlink connected data points until they reach group_size

group_assembly = pd.DataFrame(columns=df_A1.columns)

if len(neighborhoods) != 0:
    for i in range(len(neighborhood_centers)):
        group_assembly = pd.concat([group_assembly, df_A1.loc[neighborhoods[i]]], ignore_index=False)
        df_A1 = df_A1.drop(index=neighborhoods[i])
        iteration = 0
        stop_search = False
        while not stop_search:
            iteration += 1
            
            min_index = distance_matrix[i].idxmin()

            if np.isin(min_index, ml_1d_array):
                
                distance_matrix = distance_matrix.drop(index=min_index)
                continue

            for j in ml_2d_array[i]:  # j is an index of a must-link group
                if j in cl_graph and min_index in cl_graph[j]:
                    distance_matrix = distance_matrix.drop(index=min_index)
                    continue
            # Add data point to group assembly
            group_assembly = pd.concat([group_assembly, df_A1.loc[min_index].to_frame().T], ignore_index=False)
            # Remove data point from distance matrix
            distance_matrix = distance_matrix.drop(index=min_index)
            

            # Remove data point from df
            df_A1 = df_A1.drop(index=min_index)
            stop_search = True
        # FIXME add check if enough data points per Group

        cluster_centers = cluster_centers.drop(index = i)

group_assembly["cluster"] = np.repeat(range(len(group_assembly) // group_size + (len(group_assembly) % group_size > 0)), group_size)[:len(group_assembly)]

distance_matrix = functions.get_distance_matrix(df_A1, cluster_centers)

# Reindex distance matrix so it matches with df_A1
distance_matrix = distance_matrix.reindex(index = df_A1.index, columns = cluster_centers.index, method = "nearest")

while(len(distance_matrix) !=0):
    for i in distance_matrix.columns:
        
        min_row_index_found = False
        local_distance_matrix = distance_matrix.copy()
        while not min_row_index_found:
            min_row_index = local_distance_matrix[i].idxmin()
            if not check_cluster_match(group_assembly, min_row_index, i, cl_graph):
                row_to_add_to_group_assembly = df_A1.loc[min_row_index]
                row_to_add_to_group_assembly["cluster"] = i
                df_A1 = df_A1.drop(index=min_row_index)
                distance_matrix = distance_matrix.drop(index = min_row_index)
                group_assembly = pd.concat([group_assembly, row_to_add_to_group_assembly.to_frame().T], ignore_index=False)
                min_row_index_found = True

            local_distance_matrix = local_distance_matrix.drop(index = min_row_index)

group_assembly["hom_1"] = group_assembly["hom_1"].astype(float)
group_assembly["hom_2"] = group_assembly["hom_2"].astype(float)
group_assembly["het_1"] = group_assembly["het_1"].astype(float)
group_assembly["het_2"] = group_assembly["het_2"].astype(float)

report = functions.get_within_group_distance(group_assembly, 10)
print("Totale Distance", report.sum())
