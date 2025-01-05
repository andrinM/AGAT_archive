import pandas as pd
import numpy as np
from sklearn import metrics
from scipy.spatial.distance import cdist
from scipy.spatial.distance import squareform
import sys
import os
import itertools


current_dir = os.getcwd()  
src_dir = os.path.abspath(os.path.join(current_dir, 'src'))  # Parent directory of import_destination (i.e., src)
sys.path.append(src_dir)
from Data import testData
from algorithms.distance_matrix.distance_matrix import get_distance_matrix as dm
from algorithms.randomness.possible_groups import possible_groups
from algorithms.randomness.random_permutation import random_permutation


def random_dst_rank(distance_matrix, group_size, n_iterations_optimization, n_solutions = 1):

    column_sums = {col: distance_matrix[col].sum() for col in distance_matrix.columns}

    sorted_column_sums = dict(sorted(column_sums.items(), key=lambda item: item[1],reverse=True))
    # Rank distances
    ranking_top_distances = list(sorted_column_sums.keys())

    # Create greedily an initiale grouping based on the member_fitness
    initiale_grouping = greedy_grouping(distance_matrix= distance_matrix, members = ranking_top_distances, group_size= group_size)
    
    # Random permutate one two members for k times
    best_solutions = random_permutation(grouping = initiale_grouping,
                                        distance_matrix= distance_matrix,
                                        n_runs= n_iterations_optimization,
                                        n_solutions = n_solutions)

    return best_solutions


def greedy_grouping(distance_matrix, members, group_size):
    """
    Group indices based on proximity from a distance matrix, 
    updating the matrix to exclude already-used indices. First member
    of members gets grouped with the g closest other members.

    Parameters:
        distance_matrix (np.ndarray): Square distance matrix (n x n).
        members (list[int]): List of indices corresponding to the distance matrix.
        group_size (int): Number of closest integers to find in each group.

    Returns:
        dict[int, list[int]]: Dictionary of groups, where keys are group indices and 
                              values are lists of grouped indices.
    """
    # Copy the distance matrix to avoid modifying the original
    distance_matrix = distance_matrix.to_numpy()
    dist_matrix = distance_matrix.copy()
    
    n = dist_matrix.shape[0]
    
    # Set diagonal to infinity to prevent self-selection
    np.fill_diagonal(dist_matrix, np.inf)
    
    groups = {}  # Dictionary to store groups
    remaining_indices = set(members)  # Use a set to track remaining members, but keep order intact
    group_id = 0  # Initialize group ID

    for current_index in members:  # Iterate directly over the original list (members)
        if current_index not in remaining_indices:
            continue  # Skip if the current index is already used

        remaining_indices.remove(current_index)

        # Get distances for the current index
        distances = dist_matrix[current_index]
        
        # Find the g closest indices among remaining ones
        closest_indices = np.argsort(distances)[:group_size-1]
        closest_indices = [idx for idx in closest_indices if idx in remaining_indices]
        
        # Form the group (including the current index)
        group = [current_index] + closest_indices
        groups[group_id] = group  # Store the group in the dictionary
        group_id += 1  # Increment group ID

        # Remove used indices from remaining_indices
        remaining_indices.difference_update(group)

        # Update the distance matrix: set rows and columns of used indices to np.inf
        for idx in group:
            dist_matrix[idx, :] = np.inf
            dist_matrix[:, idx] = np.inf

    return groups

