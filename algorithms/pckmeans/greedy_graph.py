import pandas as pd
import numpy as np
import pandas as pd
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
import random

random.seed(None)

def greedy_partition_it(distance_matrix, k, t=100):
    n = len(distance_matrix)
    best_partition = None
    best_total_distance = float('inf')  # Initialize with a large value
    np.random.seed(None)

    for _ in range(t):  # Repeat t times
        # Shuffle the nodes randomly at the start of each iteration
        nodes = list(distance_matrix.index)
        np.random.shuffle(nodes)  # Shuffle the nodes randomly

        subgraphs = []  # List to store nodes of each subgraph
        group_assignments = pd.DataFrame(index=distance_matrix.index, columns=["group"])

        # Greedy partitioning strategy
        nodes_copy = set(nodes)  # Copy of nodes to track which are not yet assigned
        while nodes_copy:
            subgraph_nodes = []
            first_node = nodes_copy.pop()  # Select the first node for the subgraph
            subgraph_nodes.append(first_node)

            # Greedily add nodes to this subgraph
            while len(subgraph_nodes) < k and nodes_copy:
                min_distance = float('inf')
                next_node = None
                for node in nodes_copy:
                    # Calculate the total distance to nodes already in the subgraph
                    dist_sum = sum(distance_matrix.loc[node, subgraph_node] for subgraph_node in subgraph_nodes)
                    if dist_sum < min_distance:
                        min_distance = dist_sum
                        next_node = node

                # If next_node is None, something went wrong, break out of the loop
                if next_node is None:
                    break

                subgraph_nodes.append(next_node)
                nodes_copy.remove(next_node)

            # Assign group number to the subgraph nodes
            group_number = len(subgraphs)  # Group number is based on subgraph order
            subgraphs.append(subgraph_nodes)

            # Update group_assignments DataFrame
            for node in subgraph_nodes:
                group_assignments.loc[node, "group"] = group_number

        # Calculate total distance for the current partitioning
        total_distance = 0
        for subgraph in subgraphs:
            subgraph_distance = sum(distance_matrix.loc[node1, node2] for i, node1 in enumerate(subgraph) for node2 in subgraph[i+1:])
            total_distance += subgraph_distance

        # If this partitioning has a smaller total distance, save it
        if total_distance < best_total_distance:
            best_total_distance = total_distance
            best_partition = group_assignments.copy()

    return best_partition, best_total_distance

df_A1 = testData.df_A1
distance_matrix = functions.get_distance_matrix(df_A1,df_A1)

k = 3  # Size of each subgraph
best_partition, best_total_distance = greedy_partition_it(distance_matrix, k, 1000)

print(best_partition)
print(best_total_distance)
