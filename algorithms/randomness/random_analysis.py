import pandas as pd
import numpy as np
from sklearn import metrics
from scipy.spatial.distance import cdist
from scipy.spatial.distance import squareform
import sys
import os
import random
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


current_dir = os.getcwd()  
src_dir = os.path.abspath(os.path.join(current_dir, 'src'))  # Parent directory of import_destination (i.e., src)
sys.path.append(src_dir)

from Data import testData
from algorithms import functions
import pickle
from random_grouping_selection import random_grouping_selection
from random_grouping_selection_ml import random_grouping_selection_ml
from random_permutation_ml import random_permutation_ml
from random_permutation import random_permutation
from possible_groups import possible_groups
from algorithms.distance_matrix.distance_matrix import get_distance_matrix as dm


# df_A4 holds 120 members 
df_A1 = testData.df_A1
# Scale Data
columns_to_scale = ['hom__1', 'hom__2']
scaler = MinMaxScaler()

df_A1[columns_to_scale] = scaler.fit_transform(df_A1[columns_to_scale])

# Distance Matrix
distance_matrix = dm(df_A1,df_A1)
distance_matrix = pd.DataFrame(distance_matrix)

group_distances_df_A1, distance_list = possible_groups(distance_matrix, 3)

# Get the occurances of each member in the k best groups
top_k_groups = 10

top_indice_df_A1 = functions.get_top_k_integer_counts(group_distances_df_A1, top_k_groups)


# Sort the memhbers acsending (unfitest firts)
sorted_top_indice_df_A1 = dict(sorted(top_indice_df_A1.items(), key=lambda item: item[1], reverse=True))
# Rank occurences
ranking_top_indices_df_A1 = {key: idx for idx, key in enumerate(sorted_top_indice_df_A1)}

# Creat ranking with total distance on distance_matrix
column_sums = {col: distance_matrix[col].sum() for col in distance_matrix.columns}
sorted_column_sums = dict(sorted(column_sums.items(), key=lambda item: item[1]))
# Rank distances
ranking_top_distances = {key: idx for idx, key in enumerate(sorted_column_sums)}

ranking_result = {key: ranking_top_indices_df_A1.get(key, 0) + ranking_top_distances.get(key, 0) for key in ranking_top_indices_df_A1}
sorted_ranking_result = dict(sorted(ranking_result.items(), key=lambda item: item[1]))
ranking_list = list(sorted_ranking_result.keys())


reversed_ranking_list = ranking_list[::-1]

# Make a list out of the dictionary
list_sorted_top_indice_df_A1 = list(sorted_top_indice_df_A1.keys())

# Greedy algorithm, unfitest first
initiale_grouping = functions.group_by_closest(distance_matrix, 
                                               reversed_ranking_list,
                                               2)
# Create dictionary with the initiale group
initiale_grouping_dict =  {i: initiale_grouping[i] for i in range(len(initiale_grouping))}

# Random permutate one two members for k times
#best_total_distance, best_grouping = random_permutation(initiale_grouping_dict, distance_matrix, 1)

print(initiale_grouping)

row1 = pd.DataFrame(df_A1.loc[0]).T
row2 = pd.DataFrame(df_A1.loc[1]).T
print(row1)
print(row2)
distanzmatrix = dm(row1,df_A1)

