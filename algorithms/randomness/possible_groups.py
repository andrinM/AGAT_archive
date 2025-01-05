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
