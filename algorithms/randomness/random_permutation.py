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
import numpy as np
import random

def random_permutation(grouping, distance_matrix, n_runs, n_solutions=1):
    """
    Randomly optimizes groupings to minimize total intra-group distances.

    Parameters:
        grouping (dict): Dictionary of groups, where keys are group IDs and values are lists of indices.
        distance_matrix (pd.DataFrame): Distance matrix as a DataFrame.
        n_runs (int): Number of random permutations to attempt.
        n_solutions (int): Number of best solutions to return.

    Returns:
        list[tuple]: A list of tuples, each containing (total_distance, best_grouping).
    """
    # Convert distance_matrix from df to np array
    distance_matrix = distance_matrix.to_numpy()

    # List to store the top n solutions
    best_solutions = []

    # Initialize random seed for both numpy and random
    np.random.seed(None)
    random.seed(None)

    # Main loop for swapping and optimizing
    for run in range(n_runs):
        # Randomly select two different groups
        key1, key2 = random.sample(list(grouping.keys()), 2)

        # Randomly select an index from each group's list
        inner_idx1 = random.randint(0, len(grouping[key1]) - 1)
        inner_idx2 = random.randint(0, len(grouping[key2]) - 1)

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

        # Store the solution if it is among the n best
        current_solution = (total_distance, {group_id: group_indices[:] for group_id, group_indices in grouping.items()})
        best_solutions.append(current_solution)
        best_solutions = sorted(best_solutions, key=lambda x: x[0])[:n_solutions]  # Keep only n best

        # Revert the swap if the solution wasn't the best
        if current_solution not in best_solutions:
            grouping[key1][inner_idx1], grouping[key2][inner_idx2] = (
                grouping[key2][inner_idx2],
                grouping[key1][inner_idx1],
            )

    return best_solutions
