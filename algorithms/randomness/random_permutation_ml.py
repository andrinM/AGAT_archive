import pandas as pd
import numpy as np
from sklearn import metrics
from scipy.spatial.distance import cdist
from scipy.spatial.distance import squareform
import sys
import os
import random
import time
import pickle

current_dir = os.getcwd()  
src_dir = os.path.abspath(os.path.join(current_dir, 'src'))  # Parent directory of import_destination (i.e., src)
sys.path.append(src_dir)

from Data import testData
from algorithms import functions


""" Permutates a given grouping by swapping two members of two different groups.
If the total_distance is better than before, keep the new grouping, otherwise discard it.
Keyword arguments:
grouping -- dictionairy of pre-defined grouping, where the key is the group number and the values are the group members
distance_matrix -- dataframe that holds all distances
n_runs -- number of runs
"""
def random_permutation_ml(grouping, distance_matrix, n_runs, disallowed_indices):
    # Convert distance_matrix from df to np array
    distance_matrix = distance_matrix.to_numpy()

    # Tracking the best result
    best_total_distance = float('inf')
    best_groups = None

    # Initialize random seed for both numpy and random
    np.random.seed(None)
    random.seed(None)

    # Main loop for swapping and optimizing
    for run in range(n_runs):
        # Randomly select two different groups
        key1, key2 = random.sample(list(grouping.keys()), 2)

        # Randomly select an index from the first group's list, ensuring it's not disallowed
        while True:
            inner_idx1 = random.randint(0, len(grouping[key1]) - 1)
            if grouping[key1][inner_idx1] not in disallowed_indices:
                break

        # Randomly select an index from the second group's list, ensuring it's not disallowed
        while True:
            inner_idx2 = random.randint(0, len(grouping[key2]) - 1)
            if grouping[key2][inner_idx2] not in disallowed_indices:
                break

        # Perform the swap
        grouping[key1][inner_idx1], grouping[key2][inner_idx2] = (
            grouping[key2][inner_idx2],
            grouping[key1][inner_idx1],
        )
        
        # Calculate the total distance for the current grouping
        total_distance = 0  
        for group_id, group_indices in grouping.items():
            # Extract the submatrix for the group
            submatrix = distance_matrix[np.ix_(group_indices, group_indices)]
            # Sum pairwise distances (excluding diagonal)
            total_distance += np.sum(np.triu(submatrix, k=1))
        
        # Check if the new configuration improves the total distance
        if total_distance < best_total_distance:
            best_total_distance = total_distance
            best_grouping = {group_id: group_indices[:] for group_id, group_indices in grouping.items()}
        else:
            # Revert the swap if it doesn't improve the total distance
            grouping[key1][inner_idx1], grouping[key2][inner_idx2] = (
                grouping[key2][inner_idx2],
                grouping[key1][inner_idx1],
            )
    return best_total_distance, best_grouping

