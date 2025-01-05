import pandas as pd
import numpy as np
from sklearn import metrics
from scipy.spatial.distance import cdist
from scipy.spatial.distance import squareform
import sys
import os
import itertools
import pickle


current_dir = os.getcwd()  
src_dir = os.path.abspath(os.path.join(current_dir, 'src'))  # Parent directory of import_destination (i.e., src)
sys.path.append(src_dir)
from algorithms.distance_matrix.distance_matrix import get_distance_matrix as dm
from algorithms.randomness.possible_groups import possible_groups
from algorithms.randomness.random_permutation import random_permutation
from Data import testData


def random_occdst_rank(distance_matrix, group_size, top_k_groups, n_iterations_optimization, n_solutions = 1):

    # Create all possible groups and there inner distance
    all_possible_groups, distance_list = possible_groups(distance_matrix= distance_matrix, group_size= group_size) 

    # Create a dic of each member and their number of occurences in the top_k_groups (k groups with the lowest inner distance) 
    member_fitness = get_top_k_member_counts(all_possible_groups,top_k_groups)

    # Sort the member_fitness acsending (unfitest firts)
    sorted_member_fitness = dict(sorted(member_fitness.items(), key=lambda item: item[1], reverse=True))

    column_sums = {col: distance_matrix[col].sum() for col in distance_matrix.columns}

    sorted_column_sums = dict(sorted(column_sums.items(), key=lambda item: item[1],reverse=False))
    
    ranked_by_distance = {key: index for index, key in enumerate(sorted_column_sums)}
    ranked_by_fitness = {key: index for index, key in enumerate(sorted_member_fitness)}

    ranked_sum = {key: ranked_by_distance[key] + ranked_by_fitness[key] 
              for key in ranked_by_distance if key in ranked_by_fitness}
    ranked_sum_sorted = [key for key, value in sorted(ranked_sum.items(), key=lambda item: item[1])]

    # Create greedily an initiale grouping based on the member_fitness and distance
    initiale_grouping = greedy_grouping(distance_matrix= distance_matrix, members = ranked_sum_sorted, group_size= group_size)


    # Random permutate one two members for k times
    best_solutions = random_permutation(grouping = initiale_grouping,
                                        distance_matrix= distance_matrix,
                                        n_runs= n_iterations_optimization,
                                        n_solutions = n_solutions)

    return best_solutions

def get_top_k_member_counts(possible_groups, top_k_groups):
    """
    Sorts the dictionary by values and calculates the count of each member
    across the top k tuples (based on the smallest values).

    Parameters:
    - possible_groups: Dictionary with tuples as keys and numeric values.
    - top_k_groups: Number of top k groups to consider.

    Returns:
    - A dictionary with integers as keys and their counts as values.
    """
    # Step 1: Sort the dictionary by its values
    sorted_items = sorted(possible_groups.items(), key=lambda x: x[1])
    
    # Step 2: Extract the top k tuples
    top_k_tuples = [key for key, _ in sorted_items[:top_k_groups]]
    
    # Step 3: Count occurrences of each integer in the top k tuples
    top_k_members = {}
    for tup in top_k_tuples:
        for num in tup:
            top_k_members[num] = top_k_members.get(num, 0) + 1
    
    return top_k_members

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

