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
import copy



""" Randomly creates solutions and validates them. Returns an array with dictionaies. Each dictionary contains one solution.
Keyword arguments:
distance_matrix -- The distance matrix for all members
n_groups -- number of groups per solution
n_runs -- number of runs (iterations)
top_s_solutions -- number of how many of the best solutions to return
"""
def random_grouping_selection_ml(distance_matrix, n_groups, n_runs, top_s_solutions, fixed_groups):
    # Convert distance_matrix from df to np array
    distance_matrix = distance_matrix.to_numpy()

    # Create array with indices for each member
    n_indices = distance_matrix.shape[0]
    indices = np.arange(n_indices)
    group_size = distance_matrix.shape[0] / n_groups

    initial_fixed_groups = copy.deepcopy(fixed_groups)  # Save the initial state

    # Get the integers already assigned to the dictionary
    existing_indices = [val for sublist in fixed_groups.values() for val in sublist]
    original_remaining_indices = [val for val in indices if val not in existing_indices]
   
    # To track the top_s_solutions results (store dictionaries of groups)
    list_of_top_s_solutions = []

    # Run the procedure n_runs times
    for run in range(n_runs):
        np.random.seed(None)

        fixed_groups = copy.deepcopy(initial_fixed_groups)
        remaining_indices = copy.deepcopy(original_remaining_indices)
        # Shuffle the random indices
        np.random.shuffle(remaining_indices)

        # Distribute values to ensure each key has exactly group_size values
        for key in fixed_groups:
            while len(fixed_groups[key]) < group_size:
                fixed_groups[key].append(remaining_indices.pop())
                
        
        # Calculate total distance for this run
        total_distance = 0
        for group_id, group_indices in fixed_groups.items():
            # Extract submatrix for the group
            submatrix = distance_matrix[np.ix_(group_indices, group_indices)]
            # Sum pairwise distances (excluding diagonal)
            total_distance += np.sum(np.triu(submatrix, k=1))
        
        # If there is space in the top_s_solutions or the current distance is better than the worst one in the list
        if len(list_of_top_s_solutions) < top_s_solutions:
            # Add the current grouping and distance to the list
            list_of_top_s_solutions.append({
                'total_distance': total_distance, 
                'groups': {group_id: group_indices for group_id, group_indices in fixed_groups.items()}
            })
        else:
            # If top_s_solutions is full, check if the current grouping is better than the worst one
            worst_distance = max(list_of_top_s_solutions, key=lambda x: x['total_distance'])['total_distance']
            if total_distance < worst_distance:
                # Replace the worst one with the new best grouping
                list_of_top_s_solutions = [entry for entry in list_of_top_s_solutions if entry['total_distance'] != worst_distance]
                list_of_top_s_solutions.append({
                    'total_distance': total_distance, 
                    'groups': {group_id: group_indices for group_id, group_indices in fixed_groups.items()}
                })
        
        # Keep top_s_solutions sorted by total distance
        list_of_top_s_solutions = sorted(list_of_top_s_solutions, key=lambda x: x['total_distance'])
    
    return list_of_top_s_solutions




