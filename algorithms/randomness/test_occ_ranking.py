import pandas as pd
import numpy as np
import math
import sys
import os
import random
import pulp
from concurrent.futures import ThreadPoolExecutor
import itertools


current_dir = os.getcwd()  
src_dir = os.path.abspath(os.path.join(current_dir, 'src'))  # Parent directory of import_destination (i.e., src)
sys.path.append(src_dir)

from algorithms.distance_matrix.distance_matrix import get_distance_matrix as dm
from Data import testData
from algorithms import functions

def possible_groups(distance_matrix, group_size):
    """
    Calculate all possible groups of size k and their total distances using submatrices.

    Args:
        distance_matrix (pd.DataFrame): A square DataFrame (nxn) where each entry represents 
                                        the distance between two points.
        group_size (int): The desired group size (k).

    Returns:
        dict: A dictionary where keys are tuples representing groups, and values are the total distance.
        lst: of all distances
    """
    # Ensure the input is a DataFrame and square
    if not isinstance(distance_matrix, pd.DataFrame):
        raise ValueError("distance_matrix must be a pandas DataFrame.")
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square matrix.")

    # Get the indices of the distance matrix
    all_indices = list(distance_matrix.index)

    # Generate all combinations of size group_size (k)
    all_groups = list(itertools.combinations(all_indices, group_size))
    
    # Calculate the distance for each group using submatrices
    group_distances = {}
    for group in all_groups:
        group_indices = list(group)  # Convert tuple to list for submatrix indexing
        # Extract submatrix for the group
        submatrix = distance_matrix.to_numpy()[np.ix_(group_indices, group_indices)]
        # Sum pairwise distances (excluding diagonal)
        total_distance = np.sum(np.triu(submatrix, k=1))
        group_distances[group] = total_distance
        
    distance_list = list(group_distances.values())

    return group_distances, distance_list


#### MAIN ####
df_A1 = testData.df_A1
distance_matrix = dm(df_A1,df_A1)
group_size = 3

def possible_groups_distances(distance_matrix, group_size, cl_graph):

    # Create a list of indices from 0 to n-1
    all_indices = list(range(len(distance_matrix)))
    
    # Create a 2d list of all possible groups
    all_groups = list(itertools.combinations(all_indices, group_size))

    # Filter out tuples that violate cannot-link constraints
    filtered_groups = [
        group for group in all_groups
        if not any(
            other in cl_graph.get(item, set())
            for item in group for other in group if item != other
        )
    ]

    group_distances = {}
    for group in filtered_groups:
        group_indices = list(group)  # Convert tuple to list for submatrix indexing
        # Extract submatrix for the group
        submatrix = distance_matrix[np.ix_(group_indices, group_indices)]
        # Sum pairwise distances (excluding diagonal)
        total_distance = np.sum(np.triu(submatrix, k=1))
        group_distances[group] = total_distance
            
    distance_list = list(group_distances.values())
    return group_distances, distance_list

def greedy_grouping(distance_matrix, sorted_member_occurance, group_size, cl_graph, ml_graph):
    """
    Group indices based on proximity from a distance matrix, adhering to cannot-link and must-link constraints.

    Parameters:
        distance_matrix (np.ndarray): Square distance matrix (n x n).
        sorted_member_occurance (list[int]): List of indices corresponding to the distance matrix.
        group_size (int): Number of members per group.
        cl_graph (dict[int, list[int]]): Cannot-link constraints; keys are indices, values are lists of indices 
                                         that cannot be grouped with the key.
        ml_graph (dict[int, list[int]]): Must-link constraints; keys are indices, values are lists of indices 
                                         that must be grouped with the key.

    Returns:
        dict[int, list[int]]: Dictionary of groups, where keys are group indices and values are lists of grouped indices.
    """
    # Copy the distance matrix to avoid modifying the original
    dist_matrix = distance_matrix.copy()
    
    # Set diagonal to infinity to prevent self-selection
    np.fill_diagonal(dist_matrix, np.inf)
    
    groups = {}  # Dictionary to store groups
    remaining_indices = set(sorted_member_occurance)  # Track remaining members
    group_id = 0  # Initialize group ID

    for current_index in sorted_member_occurance:
        if current_index not in remaining_indices:
            continue  # Skip if the current index is already used
        
        remaining_indices.remove(current_index)

        # Initialize the group with the current index
        group = [current_index]

        # Add must-link members first
        must_link_members = ml_graph.get(current_index, [])
        for ml_member in must_link_members:
            if ml_member in remaining_indices:
                group.append(ml_member)
                remaining_indices.remove(ml_member)
                if len(group) >= group_size:
                    break  # Stop adding if the group is full

        # If the group is not full, add closest indices
        if len(group) < group_size:
            distances = dist_matrix[current_index]  # Get distances for the current index
            closest_indices = np.argsort(distances)  # Sort by distance

            for idx in closest_indices:
                if idx in remaining_indices and idx not in cl_graph.get(current_index, []):  # Check cannot-link
                    group.append(idx)
                    remaining_indices.remove(idx)
                    if len(group) >= group_size:
                        break  # Stop adding if the group is full

        # Store the group
        groups[group_id] = group
        group_id += 1

        # Update the distance matrix: set rows and columns of used indices to np.inf
        for idx in group:
            dist_matrix[idx, :] = np.inf
            dist_matrix[:, idx] = np.inf

    return groups

def random_permutation_ml(grouping, distance_matrix, n_runs, disallowed_indices):
   
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

def random_permutation_ml_cl(grouping, distance_matrix, n_runs, cl_graph, ml_graph, n_solutions=1):
    # Tracking the best results
    best_solutions = []

    # Initialize random seed for both numpy and random
    np.random.seed(None)
    random.seed(None)

    # Main loop for swapping and optimizing
    for run in range(n_runs):
        # Randomly select two different groups
        key1, key2 = random.sample(list(grouping.keys()), 2)

        # Ensure neither key1 nor key2 have must-link relationships
        while (ml_graph.get(key1) or ml_graph.get(key2)):
            # If either key has must-link relationships, pick new keys
            key1, key2 = random.sample(list(grouping.keys()), 2)

        # Get the two indices to be swapped
        idx1 = random.randint(0, len(grouping[key1]) - 1)
        idx2 = random.randint(0, len(grouping[key2]) - 1)

        # Get the actual values from the groups
        value1 = grouping[key1][idx1]
        value2 = grouping[key2][idx2]

        # Check if value1 is in cl_graph[value2]
        if value1 in cl_graph.get(value2, set()):
            # Skip this swap if it's a cannot-link pair
            continue

        # Perform the swap
        grouping[key1][idx1], grouping[key2][idx2] = grouping[key2][idx2], grouping[key1][idx1]

        # Calculate the total distance for the current grouping
        total_distance = 0
        for group_id, group_indices in grouping.items():
            # Extract the submatrix for the group
            submatrix = distance_matrix[np.ix_(group_indices, group_indices)]
            # Sum pairwise distances (excluding diagonal)
            total_distance += np.sum(np.triu(submatrix, k=1))

        # Add the current solution to the list of best solutions if it qualifies
        if len(best_solutions) < n_solutions:
            best_solutions.append((total_distance, {group_id: group_indices[:] for group_id, group_indices in grouping.items()}))
            best_solutions.sort(key=lambda x: x[0])  # Sort by total distance
        elif total_distance < best_solutions[-1][0]:
            best_solutions[-1] = (total_distance, {group_id: group_indices[:] for group_id, group_indices in grouping.items()})
            best_solutions.sort(key=lambda x: x[0])  # Sort by total distance

        # Revert the swap if it doesn't improve the total distance
        else:
            grouping[key1][idx1], grouping[key2][idx2] = grouping[key2][idx2], grouping[key1][idx1]

    # Return the top n_solutions
    return best_solutions



#all_possible_groups = possible_groups_distances(distance_matrix, 3)
cl_graph = {
    0: set(), 
    1: [10,5],     
    3: set(),
    4: set(),
    5: [10,1], 
    6: set(),     
    7: set(),
    8: set(),
    9: set(), 
    10: [5,1],     
    11: set() 
}

ml_graph = {
    0: set(), 
    1: [10,5],     
    3: set(),
    4: set(),
    5: [10,1], 
    6: set(),     
    7: set(),
    8: [10,11],
    9: set(), 
    10: [11,8],     
    11: [10,8] 
}

df_A1 = testData.df_A1
df_A = pd.DataFrame(df_A1.loc[:11])
distance_matrix = dm(df_A,df_A)
#grouping = greedy_grouping(distance_matrix, [10,1,2,3,4,5,6,7,8,9,0,11],3, cl_1, {})
grouping = {0: [10, 5, 1], 1: [2, 4, 7], 2: [3, 11, 0], 3: [6, 8, 9]}


best_total_distance = random_permutation_ml_cl(grouping, distance_matrix, 10,cl_graph,ml_graph, 10)
print(best_total_distance)