import pandas as pd
import numpy as np
import pandas as pd
from sklearn import metrics
import pulp
from scipy.spatial.distance import cdist
from scipy.spatial.distance import squareform
import random


""" Returns a distance matrix, based on the two dataframes given.
Data Frames need the same amount of columns.
df1 and df2 can be the same, but then the diagonal is not 0!
"""
def get_distance_matrix(df1, df2):
    # Check if the column "cluster" is in the df, if so drop it
    if "cluster" in df1.columns:
        df1 = df1.drop(columns=["cluster"])
    if "cluster" in df2.columns:
        df2 = df2.drop(columns=["cluster"])
    
    homogenous_distances = (cdist(df1.loc[:, df1.columns.str.startswith('hom')], df2.loc[:, df2.columns.str.startswith('hom')], 'euclidean'))**2
    heterogenous_distances = cdist((df1.loc[:, df1.columns.str.startswith('het')]),(df2.loc[:, df2.columns.str.startswith('het')]),ninja_distance)
    one_hot_distances = get_one_hot_distances(df1, df2)  

    distance_matrix = np.sqrt(homogenous_distances + heterogenous_distances + one_hot_distances)

    return distance_matrix

def get_distance_matrix_keep_indices(df1, df2):
    # Check if the column "cluster" is in the df, if so drop it
    if "cluster" in df1.columns:
        df1 = df1.drop(columns=["cluster"])
    if "cluster" in df2.columns:
        df2 = df2.drop(columns=["cluster"])
    
    homogenous_distances = (cdist(df1.loc[:, df1.columns.str.startswith('hom')], df2.loc[:, df2.columns.str.startswith('hom')], 'euclidean'))**2
    heterogenous_distances = cdist((df1.loc[:, df1.columns.str.startswith('het')]),(df2.loc[:, df2.columns.str.startswith('het')]),ninja_distance)
    one_hot_distances = get_one_hot_distances(df1, df2)  

    distance_matrix = np.sqrt(homogenous_distances + heterogenous_distances + one_hot_distances)
    distance_matrix = pd.DataFrame(distance_matrix, index=df1.index, columns=df2.index)

    return distance_matrix

""" Calculates the mean of all data points in an assigned cluster.
Returns a Dataframe with n_cluster data points.
Keyword arguments:
df: Data Frame of data points WITH a column "cluster", where each data point is assigned to a cluster
n_cluster: the total number of clusters 
"""
def get_cluster_centers(df, n_clusters):
    # Initialize an empty DataFrame for new_cluster_centers
    new_cluster_centers = pd.DataFrame()
    for i in range(n_clusters):
        # Filter rows where "cluster" equals i, and only include columns starting with 'hom'
        homogenous_mean = df.loc[df["cluster"] == i, df.columns.str.startswith('hom')].mean()
        # Filter rows where "cluster" equals i, and only include columns starting with 'het'
        heterogenous_mean = df.loc[df["cluster"] == i, df.columns.str.startswith('het')].mean()
        # Filter rows where "cluster" equals i, and only include columns starting with 'hot'
        one_hot_mean = get_one_hot_mean(df.loc[df["cluster"] == i, df.columns.str.startswith('hot')])

        # Combine the means into one Series
        combined_mean = pd.concat([homogenous_mean, heterogenous_mean, one_hot_mean], axis=0)
        # Add the combined_mean as a new row in new_cluster_centers
        new_cluster_centers = pd.concat([new_cluster_centers, combined_mean.to_frame().T], ignore_index=True)
        
    # Make sure the values are floats
    columns_to_convert = df.filter(regex='^(hom|het)').columns
    new_cluster_centers[columns_to_convert] = new_cluster_centers[columns_to_convert].astype(float)
    return new_cluster_centers

""" Group the data points in the distance_matrix into groups of group_size. 
"""
def lp_solver(distance_matrix, group_size):

    if (len(distance_matrix) % group_size != 0):
        raise ValueError("number of data points is not divisible by desired group size")

    # Initialize the ILP problem
    prob = pulp.LpProblem("MinimizeColumnSums", pulp.LpMinimize)

    # Define decision variables
    variables = pulp.LpVariable.dicts("x", ((i, j) for i in range(distance_matrix.shape[0]) for j in range(distance_matrix.shape[1])), cat="Binary")

    # Objective function: Minimize the total sum of chosen elements
    prob += pulp.lpSum(distance_matrix.iloc[i, j] * variables[(i, j)] for i in range(distance_matrix.shape[0]) for j in range(distance_matrix.shape[1]))

    # Constraint 1: Select exactly 3 values in each column
    for j in range(distance_matrix.shape[1]):
        prob += pulp.lpSum(variables[(i, j)] for i in range(distance_matrix.shape[0])) == group_size

    # Constraint 2: Each row can be used at most once
    for i in range(distance_matrix.shape[0]):
        prob += pulp.lpSum(variables[(i, j)] for j in range(distance_matrix.shape[1])) <= 1

    # Solve the ILP
    prob.solve()

    # Extract the chosen values
    chosen_values = [(i, j) for i in range(distance_matrix.shape[0]) for j in range(distance_matrix.shape[1]) if pulp.value(variables[(i, j)]) == 1]
    assigned_clusters = pd.DataFrame(chosen_values, columns=["index",'cluster'])
    assigned_clusters = assigned_clusters.drop(columns = "index")
    
    return assigned_clusters

# FIXME max_feature_distance could have different values for different features
def ninja_distance(x,y,max_feature_distance = 2):
        return (np.sum(np.abs(np.abs(x - y) - max_feature_distance)**2))


""" Jaccard Distance returns 0 if the arrays are the same and 1 otherwise
FIXME if we use more than one 1 there will be a different distance.
"""
def get_one_hot_distances(df, cluster_centers, weight = 2):
    result = pd.DataFrame(0, index=range(len(df)), columns=range(len(cluster_centers)))

    for col in df.columns:
        if col.startswith("hot"):
            dist_matrix_one_hot = pd.DataFrame(cdist(np.vstack(df[col]),np.vstack(cluster_centers[col]), "jaccard"))
            dist_matrix_one_hot *= weight
            result += (dist_matrix_one_hot**2) 
    return result

""" Calculate the most common permutation in each column.
The most comon permutation is the mean, if there is a tie, chose random.
Returns a pd.Series with a mean for each column.
Keyword arguments:
df -- allready filtered (hot) Data Frame
"""
def get_one_hot_mean(df):
    # Initialize a dictionary to store the most common permutation for each column
    most_common_permutations = {}

    # Iterate through each column
    for col in df.columns:
        # Get the counts of each permutation in the column
        counts = df[col].apply(tuple).value_counts()

        # Get the permutation with the highest count
        most_common = counts.idxmax()  # This is the permutation tuple

        # Store the most common permutation using the column name
        most_common_permutations[col] = most_common

    # Convert the dictionary into a Series aligned with the column names
    result_series = pd.Series(most_common_permutations)
    
    return result_series

""" Returns True if df1 has the same rows as df2
"""
def compare_data_frames(df1,df2, n_clusters):
    # Reset the indices to 0 to n_clusters
    df1 = df1.reset_index(drop=True)
    df2 = df2.reset_index(drop=True)
    count = 0
    for i in range(n_clusters):
        for j in range(n_clusters):
            if (df1.loc[i].equals(df2.loc[j])):
                count +=1
    return (count == n_clusters)

""" Calculates the distance within one group.
"""
def calculate_within_distance(df1,df2):
    within_distance_df = get_distance_matrix(df1,df2)
    np.fill_diagonal(within_distance_df.values, 0)

    return within_distance_df.to_numpy().sum() / 2

""" Returns a Data Frame. The index corresponds to the cluster and the column distance
to the within cluster (or group) distance. Calculates the within group distance for all groups
at once.
"""
def get_within_group_distance(df, n_groups):
    within_group_distance = pd.DataFrame(columns=["distance"])
    for i in range(n_groups):
        within_group_distance.loc[i] = calculate_within_distance(df.loc[df["cluster"] == i],df.loc[df["cluster"] == i])
    return within_group_distance

""" Return n_cluster random datapoints from the given df
"""
def get_random_cluster_centers(df, n_clusters):
    cluster_centers = df.sample(n=n_clusters, replace=False, random_state = 1)

    return cluster_centers

""" df is the distance_matrix between the data_points and the cluster_centers.
We search the smallest value in each row and assigne the column index (cluster_center)
to the dataframe
"""
def assign_cluster(df):
    # Get the column name (index) corresponding to the minimum value for each row
    min_columns = df.idxmin(axis=1)
    # Create a new DataFrame with one column 'cluster' that stores these column names
    result_df = pd.DataFrame(min_columns, columns=['cluster'])
    
    return result_df

""" This is a basic kmean that returns the final cluster centers.
The return is for further processing with the lp solver
"""
def kmean(df, n_clusters, max_iterations):
    iteration = 0
    for i in range(max_iterations):

        cluster_centers = get_random_cluster_centers(df, n_clusters)

        prev_cluster_centers = cluster_centers

        distance_matrix = get_distance_matrix(df, cluster_centers)

        assigned_clusters =  assigne_clusters_with_distance(distance_matrix)

        df[["cluster", "distance"]] = assigned_clusters
        
        iteration += 1

        cluster_centers = get_cluster_centers(df, n_clusters)
        if (compare_data_frames(prev_cluster_centers,cluster_centers, n_clusters)):
            break
    return cluster_centers, df

def greedy_partition(distance_matrix, k):
    n = len(distance_matrix)
    subgraphs = []  # List to store nodes of each subgraph
    nodes = set(distance_matrix.index)  # Use DataFrame index instead of range(n)

    # Create a DataFrame to store the group assignments (initialize with NaN)
    group_assignments = pd.DataFrame(index=distance_matrix.index, columns=["group"])

    # Greedy partitioning strategy
    while nodes:
        subgraph_nodes = []
        first_node = nodes.pop()  # Select the first node for the subgraph
        subgraph_nodes.append(first_node)

        # Greedily add nodes to this subgraph
        while len(subgraph_nodes) < k and nodes:
            min_distance = float('inf')
            next_node = None
            for node in nodes:
                # Calculate the total distance to nodes already in the subgraph
                dist_sum = sum(distance_matrix.loc[node, subgraph_node] for subgraph_node in subgraph_nodes)
                if dist_sum < min_distance:
                    min_distance = dist_sum
                    next_node = node

            # If next_node is None, something went wrong, break out of the loop
            if next_node is None:
                break

            subgraph_nodes.append(next_node)
            nodes.remove(next_node)

        # Assign group number to the subgraph nodes
        group_number = len(subgraphs)  # Group number is based on subgraph order
        subgraphs.append(subgraph_nodes)

        # Update group_assignments DataFrame
        for node in subgraph_nodes:
            group_assignments.loc[node, "group"] = group_number

    return group_assignments


def assigne_clusters_with_distance(df):
    # Get the column name (index) corresponding to the minimum value for each row
    min_columns = df.idxmin(axis=1)

    # Get the actual minimum value
    min_values = df.min(axis=1)
    # Create a new DataFrame with one column 'cluster' that stores these column names
    result_df = pd.DataFrame({'cluster': min_columns, 'distance': min_values})
    
    return result_df

def distance_between_two_datapoints(df1, df2):
    within_distance_df = get_distance_matrix(df1,df2)

    return within_distance_df.to_numpy().sum()

def get_cluster_centers_from_neighborhoods(df, neighboorhoods):
    # Initialize an empty DataFrame for new_cluster_centers
    new_cluster_centers = pd.DataFrame(columns=df.columns)

    for neighbrhood in neighboorhoods:
        # Filter rows where "cluster" equals i, and only include columns starting with 'hom'
        homogenous_mean = df.iloc[neighbrhood].loc[:, df.columns.str.startswith('hom')].mean()
        # Filter rows where "cluster" equals i, and only include columns starting with 'het'
        heterogenous_mean = df.iloc[neighbrhood].loc[:, df.columns.str.startswith('het')].mean()
        # Filter rows where "cluster" equals i, and only include columns starting with 'hot'
        one_hot_mean = get_one_hot_mean(df.iloc[neighbrhood].loc[:, df.columns.str.startswith('hot')])
        # Combine the means into one Series
        combined_mean = pd.concat([homogenous_mean, heterogenous_mean, one_hot_mean], axis=0)
        # Add the combined_mean as a new row in new_cluster_centers
        new_cluster_centers = pd.concat([new_cluster_centers, combined_mean.to_frame().T], ignore_index=True)
        
    # Make sure the values are floats
    columns_to_convert = df.filter(regex='^(hom|het)').columns
    new_cluster_centers[columns_to_convert] = new_cluster_centers[columns_to_convert].astype(float)
    return new_cluster_centers

def greedy_partition_it(distance_matrix, k, t=100):
    n = len(distance_matrix)
    best_partition = None
    best_total_distance = float('inf')  # Initialize with a large value

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

def get_top_k_integer_counts(input_dict, k):
    """
    Sorts the dictionary by values and calculates the count of each integer
    across the top k tuples (based on the smallest values).

    Parameters:
    - input_dict: Dictionary with tuples as keys and numeric values.
    - k: Number of top entries to consider.

    Returns:
    - A dictionary with integers as keys and their counts as values.
    """
    # Step 1: Sort the dictionary by its values
    sorted_items = sorted(input_dict.items(), key=lambda x: x[1])
    
    # Step 2: Extract the top k tuples
    top_k_tuples = [key for key, _ in sorted_items[:k]]
    
    # Step 3: Count occurrences of each integer in the top k tuples
    counts = {}
    for tup in top_k_tuples:
        for num in tup:
            counts[num] = counts.get(num, 0) + 1
    
    return counts

import numpy as np

def group_by_closest(distance_matrix, indices, g):
    """
    Group indices based on proximity from a distance matrix, 
    updating the matrix to exclude already-used indices.

    Parameters:
        distance_matrix (np.ndarray): Square distance matrix (n x n).
        indices (list[int]): List of indices corresponding to the distance matrix.
        g (int): Number of closest integers to find in each group.

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
    remaining_indices = set(indices)
    group_id = 0  # Initialize group ID

    while remaining_indices:
        # Take the first remaining index
        current_index = next(iter(remaining_indices))
        remaining_indices.remove(current_index)

        # Get distances for the current index
        distances = dist_matrix[current_index]
        
        # Find the g closest indices among remaining ones
        closest_indices = np.argsort(distances)[:g]
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

def randomly_assign_groups(n_groups, n_members):
    # Generate a list of member indices
    members = list(range(n_members))

    random.seed(None)
    # Shuffle the list randomly
    random.shuffle(members)
    
    # Calculate the size of each group
    group_size = n_members // n_groups
    
    # Create the dictionary with groups as keys
    group_dict = {}
    for i in range(n_groups):
        group_dict[i] = members[i*group_size:(i+1)*group_size]
    
    return group_dict