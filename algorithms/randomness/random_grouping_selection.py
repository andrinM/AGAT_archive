import pandas as pd
import numpy as np
from sklearn import metrics
from scipy.spatial.distance import cdist
from scipy.spatial.distance import squareform
import sys
import os
current_dir = os.getcwd()  
src_dir = os.path.abspath(os.path.join(current_dir, 'src'))  # Parent directory of import_destination (i.e., src)
sys.path.append(src_dir)
from Data import testData
from algorithms import functions
import random
import time
import pickle


def random_grouping_selection(distance_matrix, n_groups, n_runs, top_s_solutions, time_limit=None):
    # Convert distance_matrix from df to np array
    distance_matrix = distance_matrix.to_numpy()

    # Create array with indices for each member
    n_indices = distance_matrix.shape[0]
    indices = np.arange(n_indices)

    # To track the top_s_solutions results (store tuples of (total_distance, groups))
    list_of_top_s_solutions = []

    # Start the CPU timer if time_limit is provided
    if time_limit is not None:
        start_time = time.process_time()

    # Run the procedure n_runs times or until time limit is exceeded
    for run in range(n_runs):
        # Check if time limit is exceeded
        if time_limit is not None and (time.process_time() - start_time) >= time_limit:
            print("Time limit exceeded, stopping early.")
            break  # Exit if time limit is exceeded
        
        np.random.seed(None)
        # Shuffle indices and assign them to groups
        np.random.shuffle(indices)
        groups = {i: indices[i::n_groups] for i in range(n_groups)}
        
        # Calculate total distance for this run
        total_distance = 0
        for group_id, group_indices in groups.items():
            # Extract submatrix for the group
            submatrix = distance_matrix[np.ix_(group_indices, group_indices)]
            # Sum pairwise distances (excluding diagonal)
            total_distance += np.sum(np.triu(submatrix, k=1))
        
        # If there is space in the top_s_solutions or the current distance is better than the worst one in the list
        if len(list_of_top_s_solutions) < top_s_solutions:  
            # Add the current grouping and distance to the top_s_solutions list
            list_of_top_s_solutions.append((total_distance, {group_id: group_indices.tolist() for group_id, group_indices in groups.items()}))
        else:
            # If top_s_solutions is full, check if the current grouping is better than the worst one
            worst_distance = max(list_of_top_s_solutions, key=lambda x: x[0])[0]
            if total_distance < worst_distance:
                # Replace the worst one with the new best grouping
                list_of_top_s_solutions = [entry for entry in list_of_top_s_solutions if entry[0] != worst_distance]
                list_of_top_s_solutions.append((total_distance, {group_id: group_indices.tolist() for group_id, group_indices in groups.items()}))
        
        # Keep top_s_solutions sorted by total distance
        list_of_top_s_solutions = sorted(list_of_top_s_solutions, key=lambda x: x[0])
    
    return list_of_top_s_solutions



