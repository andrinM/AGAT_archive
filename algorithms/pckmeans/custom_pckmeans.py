import pandas as pd
import numpy as np
import math
import sys
import os
import random
import pulp
from concurrent.futures import ThreadPoolExecutor

current_dir = os.getcwd()  
src_dir = os.path.abspath(os.path.join(current_dir, 'src'))  # Parent directory of import_destination (i.e., src)
sys.path.append(src_dir)

from exceptions import EmptyClustersException
from constraints import preprocess_constraints
from algorithms.distance_matrix.distance_matrix import get_distance_matrix as dm
from Data import testData



class custom_PCKMeans:
    def __init__(self, df, group_size, n_clusters=3, max_iter=100, ml_list=None, cl_list=None):
        """
        Initialize the custom_PCKMeans instance.

        Parameters:
        - df (pd.DataFrame): The dataset to be clustered.
        - n_clusters (int): The number of clusters to form. Default is 3.
        - max_iter (int): Maximum number of iterations for the clustering algorithm. Default is 100.
        - ml_list (list): Must-link constraints. Default is None.
        - cl_list (list): Cannot-link constraints. Default is None.
        """
        self.df = df  # Store the dataset
        self.group_size = group_size
        self.n_clusters = n_clusters  # Number of clusters
        self.max_iter = max_iter  # Maximum number of iterations
        self.ml_list = ml_list if ml_list is not None else []  # Must-link constraints
        self.cl_list = cl_list if cl_list is not None else []  # Cannot-link constraints
        self.labels_ = None  # Placeholder for cluster labels
        self.cluster_centers_ = None  # Placeholder for cluster centers
        

    def fit(self):
        """
        Fit the PCKMeans model to the dataset.
        This function will implement the clustering logic.
        """
        n = self.df.shape[0]
        ml_graph, cl_graph, neighborhoods = preprocess_constraints(ml = self.ml_list, cl = self.cl_list, n = n)
        
        if self.ml_list:
            # neighborhood_clusters are all groups with must-links the same size as group_size
            neighborhood_clusters, df_without_neighborhood_members = self.fill_neighborhoods(df = self.df,
                                                                                         neighborhoods=neighborhoods,
                                                                                         group_size = self.group_size,
                                                                                         cl_graph = cl_graph)
            # Run kmean
            unbalanced_final_clusters, new_centroids, distance_matrix_initial_centroids = self.kmeans_with_constraints(df = df_without_neighborhood_members,
                                                                k = self.n_clusters,
                                                                cl_graph= cl_graph,
                                                                max_iter=self.max_iter)
            
            # Balance cluster_size (divisible by group_size)
            balanced_clusters = self.balance_clusters_cl(unbalanced_final_clusters = unbalanced_final_clusters,
                                                     group_size= self.group_size,
                                                     cl_graph=cl_graph)
            
            # Create distance_matrices, based on balanced clusters
            all_distance_matrices = self.get_all_distance_matrices(balanced_clusters= balanced_clusters)

            # Solve lp for all distance matrices
            lp_solution = self.solve_all(all_distance_matrices, self.group_size)

            # Formate solution
            final_grouping = [[[balanced_clusters[i][index] for index in group] for group in sublist]
            for i, sublist in enumerate(lp_solution)]
            final_grouping = [inner_list for sublist in final_grouping for inner_list in sublist]
            final_grouping = final_grouping + neighborhood_clusters

            return final_grouping, lp_solution, neighborhood_clusters, balanced_clusters
         
        else: # Case, where there are no must-links
            df_without_neighborhood_members = self.df
            unbalanced_final_clusters, new_centroids, distance_matrix_initial_centroids = self.kmeans_with_constraints(df = df_without_neighborhood_members,
                                                                k = self.n_clusters,
                                                                cl_graph= cl_graph,
                                                                max_iter=self.max_iter)
            
            # Balance cluster_size (divisible by group_size)
            balanced_clusters = self.balance_clusters_cl(unbalanced_final_clusters = unbalanced_final_clusters,
                                                     group_size= self.group_size,
                                                     cl_graph=cl_graph)
            
            # Create distance_matrices, based on balanced clusters
            all_distance_matrices = self.get_all_distance_matrices(balanced_clusters= balanced_clusters)

            # Solve lp for all distance matrices
            lp_solution = self.solve_all(all_distance_matrices, self.group_size)

            # Formate solution
            final_grouping = [[[balanced_clusters[i][index] for index in group] for group in sublist]
            for i, sublist in enumerate(lp_solution)]
            final_grouping = [inner_list for sublist in final_grouping for inner_list in sublist]
            
            return final_grouping, lp_solution, balanced_clusters

    def get_all_distance_matrices(self, balanced_clusters):
        all_distance_matrices = []
        for balanced_cluster in balanced_clusters:
            distance_matrix = dm(self.df.iloc[balanced_cluster], self.df.iloc[balanced_cluster])
            all_distance_matrices.append(distance_matrix)
        return all_distance_matrices

    def solve_all(self, distance_matrices, group_size):
        """
        Solve the LP problems for all distance matrices using multithreading.
        """
        results = []
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.solve_lp, dm, group_size) for dm in distance_matrices]
            for future in futures:
                results.append(future.result())
        return results

    def solve_lp(self, distance_matrix, k):
            
        n = len(distance_matrix)
        num_groups = n // k

        # Problem definition
        problem = pulp.LpProblem("Minimize_Grouping_Distance", pulp.LpMinimize)

        # Decision variables
        # y[i, g] is 1 if point i is assigned to group g, 0 otherwise
        y = pulp.LpVariable.dicts('y', ((i, g) for i in range(n) for g in range(num_groups)), 
                                lowBound=0, upBound=1, cat=pulp.LpBinary)
        
        # Auxiliary variables
        # z[i, j, g] is 1 if both i and j are in group g, 0 otherwise
        z = pulp.LpVariable.dicts('z', ((i, j, g) for i in range(n) for j in range(n) if i < j for g in range(num_groups)),
                                lowBound=0, upBound=1, cat=pulp.LpBinary)

        # Objective function: minimize the sum of distances for grouped points
        problem += pulp.lpSum(distance_matrix[i, j] * z[i, j, g] 
                            for i in range(n) for j in range(n) if i < j 
                            for g in range(num_groups))

        # Constraint 1: Each point must be in exactly one group
        for i in range(n):
            problem += pulp.lpSum(y[i, g] for g in range(num_groups)) == 1

        # Constraint 2: Each group must have exactly k members
        for g in range(num_groups):
            problem += pulp.lpSum(y[i, g] for i in range(n)) == k

        # Constraint 3: Define the auxiliary variable z[i, j, g]
        for i in range(n):
            for j in range(n):
                if i < j:
                    for g in range(num_groups):
                        # z[i, j, g] can only be 1 if both y[i, g] and y[j, g] are 1
                        problem += z[i, j, g] <= y[i, g]
                        problem += z[i, j, g] <= y[j, g]
                        problem += z[i, j, g] >= y[i, g] + y[j, g] - 1

        # Solve the problem
        solver = pulp.PULP_CBC_CMD(msg=1, timeLimit= 60)
        status = problem.solve(solver)

        # If the problem is solved, extract the results
        if status == pulp.LpStatusOptimal:
            # Create a list to store the groups
            groups = [[] for _ in range(num_groups)]

            # Assign points to groups
            for i in range(n):
                for g in range(num_groups):
                    if pulp.value(y[i, g]) == 1:
                        groups[g].append(i)
                        break
            
            return groups  # Return the groups as a 2D list of indices
        else:
            print(f"Problem Status: {pulp.LpStatus[status]}")
            print("No optimal solution found.")
            return None
        
    def balance_clusters_cl(self, unbalanced_final_clusters, group_size, cl_graph):
        """
        Balances clusters so that every cluster is divisible by group_size, 
        while respecting cannot-link constraints.

        Args:
        - unbalanced_final_clusters: A list of lists where each list represents a cluster.
        - group_size: Integer representing the required group size.
        - cl_graph: Dictionary where keys are indices and values are lists of indices they cannot link with.

        Returns:
        - unbalanced_final_clusters: Balanced list of clusters.
        """
        
        while True:
            give = {}
            take = {}
            cluster_sizes = [len(cluster) for cluster in unbalanced_final_clusters]

            for idx, cluster_size in enumerate(cluster_sizes):
                r = cluster_size % group_size
                if (r != 0) and (r >= group_size / 2):
                    take[idx] = r  # Add a key-value pair
                elif (r != 0) and (r < group_size / 2):
                    give[idx] = r  # Add a key-value pair

            # Take an element from the first give cluster and add it to the first element of the first take cluster
            if give and take:
                # Case 1: Both `give` and `take` are not empty
                give_keys = list(give.keys())
                take_keys = list(take.keys())
                for g_key in give_keys:
                    for t_key in take_keys:
                        if unbalanced_final_clusters[g_key]:  # Ensure the list is not empty
                            chosen_element = random.choice(unbalanced_final_clusters[g_key])

                            # Check for cannot-link constraints
                            if all(
                                chosen_element not in cl_graph.get(member, [])
                                for member in unbalanced_final_clusters[t_key]
                            ):
                                unbalanced_final_clusters[t_key].append(chosen_element)
                                unbalanced_final_clusters[g_key].remove(chosen_element)
                                break
                    else:
                        continue
                    break

            elif not take and give:
                # Case 2: `take` is empty and `give` is not
                give_keys = list(give.keys())
                for g_key in give_keys:
                    for g_key_2 in give_keys:
                        if g_key != g_key_2 and unbalanced_final_clusters[g_key]:
                            chosen_element = random.choice(unbalanced_final_clusters[g_key])

                            # Check for cannot-link constraints
                            if all(
                                chosen_element not in cl_graph.get(member, [])
                                for member in unbalanced_final_clusters[g_key_2]
                            ):
                                
                                unbalanced_final_clusters[g_key_2].append(chosen_element)
                                unbalanced_final_clusters[g_key].remove(chosen_element)
                                break
                    else:
                        continue
                    break

            elif not give and take:
                # Case 3: `give` is empty and `take` is not
                take_keys = list(take.keys())
                for t_key in take_keys:
                    for t_key_2 in take_keys:
                        if t_key != t_key_2 and unbalanced_final_clusters[t_key]:
                            chosen_element = random.choice(unbalanced_final_clusters[t_key])

                            # Check for cannot-link constraints
                            if all(
                                chosen_element not in cl_graph.get(member, [])
                                for member in unbalanced_final_clusters[t_key_2]
                            ):
                                unbalanced_final_clusters[t_key_2].append(chosen_element)
                                unbalanced_final_clusters[t_key].remove(chosen_element)
                                break
                    else:
                        continue
                    break

            elif not give and not take:
                # Case 4: Both `give` and `take` are empty
                break
        # Remove empty clusters (lists)
        unbalanced_final_clusters = [cluster for cluster in unbalanced_final_clusters if cluster]

        return unbalanced_final_clusters

    def kmeans_with_constraints(self, df, k, cl_graph, max_iter=100):
        # Initialize centroids using KMeans++
        distance_matrix = dm(df,df)
        original_indices = df.index
        initial_centroids = self.kmeans_plus_plus(distance_matrix=distance_matrix, original_indices = original_indices, k=k)
        
        # Initialize the distance matrix
        distance_matrix_initial_centroids = dm(df, df.loc[initial_centroids])
        
        # Create a list to store clusters
        clusters = self.assign_points_to_centroids_with_constraints(distance_matrix_initial_centroids, original_indices, k, cl_graph)
        
        # Store the previous cluster assignments to check for convergence
        prev_clusters = None

        for _ in range(max_iter):
            # If clusters haven't changed, break the loop (convergence)
            if clusters == prev_clusters:
                break
            if prev_clusters is not None:
                if set(map(tuple, clusters)) == set(map(tuple, prev_clusters)):
                    break

            prev_clusters = clusters
            
            # Recalculate centroids
            new_centroids = pd.DataFrame(columns=df.columns)
            
            for cluster_points in clusters:
                # Get the new centroid for the cluster
                if len(cluster_points) > 0:
                    points_in_cluster = df.loc[cluster_points]
                    centroid = self.get_cluster_centers(points_in_cluster)
                    new_centroids = pd.concat([new_centroids, centroid], ignore_index=True)
            
            # Update centroids
            distance_matrix_initial_centroids = dm(df, new_centroids)
            
            # Reassign points to new centroids considering constraints
            clusters = self.assign_points_to_centroids_with_constraints(distance_matrix_initial_centroids, original_indices, k, cl_graph)
            distance_matrix_initial_centroids = pd.DataFrame(distance_matrix_initial_centroids)
            distance_matrix_initial_centroids.index = df.index
        
        return clusters, new_centroids, distance_matrix_initial_centroids

    def assign_points_to_centroids_with_constraints(self, distance_matrix, original_indices, n_clusters, cl_graph):
        """
        Assigns each point to the closest centroid while ensuring cannot-link constraints are respected.
        
        Args:
        - distance_matrix: A 2D numpy array representing the pairwise distances between points and centroids.
        - original_indices: A list of the original indices of the points in the dataset.
        - k: The number of centroids (clusters).
        - cl_graph: A dictionary where the key is the index of a point, and the value is a set of points that it cannot be assigned to the same cluster with.
        
        Returns:
        - centroid_assignments: A list of lists where each list contains the original indices of the data points 
        closest to a specific centroid.
        """
        # Initialize a list of empty lists for each centroid
        centroid_assignments = [[] for _ in range(n_clusters)]
        
        # Initialize a dictionary to keep track of the points already assigned to each centroid
        centroids_assigned = {i: [] for i in range(n_clusters)}
        
        # For each point, assign it to the closest centroid while respecting the cannot-link constraints
        for i in range(len(distance_matrix)):
            # Get the distances for the i-th point to all centroids
            distances_to_centroids = distance_matrix[i]
            
            # Check if the current point has any cannot-link constraint with existing points in any cluster
            valid_assignment_found = False
            
            while not valid_assignment_found:
                # Find the index of the closest centroid (minimum distance)
                closest_centroid_idx = np.argmin(distances_to_centroids)
                
                # Check if assigning this point to the closest centroid violates any cannot-link constraint
                violates_cannotlink = False
                for assigned_index in centroids_assigned[closest_centroid_idx]:
                    if original_indices[i] in cl_graph.get(assigned_index, set()):
                        violates_cannotlink = True
                        break
                
                if not violates_cannotlink:
                    # If no violation, assign the point to the closest centroid
                    centroid_assignments[closest_centroid_idx].append(original_indices[i])
                    centroids_assigned[closest_centroid_idx].append(original_indices[i])
                    valid_assignment_found = True
                else:
                    # If violated, mark that centroid as invalid and find the second closest centroid
                    distances_to_centroids[closest_centroid_idx] = np.inf  # Disable this centroid
        
        return centroid_assignments

    def kmeans_plus_plus(self, distance_matrix, original_indices, k):
        """
        KMeans++ initialization on a distance matrix with preserved indices.
        
        Parameters:
            distance_matrix (np.ndarray): Square distance matrix (n x n).
            original_indices (np.ndarray): The original indices of the DataFrame.
            k (int): Number of clusters.
        
        Returns:
            list: Indices of initial centroids (from the original DataFrame).
        """
        # Ensure the distance matrix is 2D
        if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
            raise ValueError("The distance matrix must be a square 2D array.")

        n_points = distance_matrix.shape[0]
        centroids = []

        # Randomly select the first centroid using the original index
        first_centroid = np.random.choice(n_points)
        centroids.append(first_centroid)

        for _ in range(1, k):
            # Calculate minimum distances to the existing centroids
            # Ensure centroids is a list of indices
            min_distances = np.min(distance_matrix[:, centroids], axis=1)

            # Select the next centroid with probability proportional to the distance
            probabilities = min_distances / np.sum(min_distances)
            cumulative_probs = np.cumsum(probabilities)
            rand_val = np.random.rand()

            # Select the next centroid based on the probability distribution
            for i, prob in enumerate(cumulative_probs):
                if rand_val <= prob:
                    next_centroid = i  # Index of the new centroid in the distance matrix
                    break

            # Ensure the selected centroid is valid and not already in the list
            if next_centroid in centroids:
                continue
            centroids.append(next_centroid)

        # Map centroids back to original indices using original_indices
        original_centroids = original_indices[centroids]
        original_centroids = original_centroids.tolist()


        return original_centroids

    def fill_neighborhoods(self, df, neighborhoods, group_size, cl_graph):

        # Get centers of neighborhoods (mean)
        neighborhood_centers = self.get_neighborhood_centers(self.df, neighborhoods)
        # Flatten neighborhoods into a 1d list
        neighborhood_members = [index for sublist in neighborhoods for index in sublist]

        # Remove indices if they are in a neighborhood
        df_without_neighborhood_members = df.drop(neighborhood_members)

        # Store the original indices (they get removed by creating the distance matrix)
        original_indices = df_without_neighborhood_members.index
        
        
        # Creating distance_matrix to neighborhood_centers
        distance_matrix = pd.DataFrame(dm(df_without_neighborhood_members, neighborhood_centers))

        # Reindex
        distance_matrix.index = original_indices

        # Assigne members to neighborhoods
        neighborhood_clusters = self.snake_draft_selection(distance_matrix= distance_matrix, clusters=neighborhoods, group_size =group_size, cl_graph= cl_graph)

        # Remove assigned members from the original dataframe
        neighborhood_members = [index for sublist in neighborhood_clusters for index in sublist]
        df_without_neighborhood_members = df.drop(neighborhood_members)

        return neighborhood_clusters, df_without_neighborhood_members

    def snake_draft_selection(self, distance_matrix, clusters, group_size, cl_graph):
        """
        Perform a snake draft selection from the distance matrix, considering cannot-link constraints.

        Parameters:
        - distance_matrix: pd.DataFrame, rows are datapoints, columns are clusters.
        - clusters: list of lists, initialized 2D array to store indices for each cluster.
        - group_size: int, the size of each cluster.
        - cl_graph: dict, where keys are indices and values are sets of indices
        that the key index cannot be in the same cluster with.

        Returns:
        - clusters: updated 2D list with selected indices.
        - distance_matrix: updated distance matrix with selected rows removed.
        """
        # Create a copy of the distance matrix to avoid modifying the original
        temp_matrix = distance_matrix.copy()

        # Total number of clusters
        num_clusters = len(clusters)

        # Continue until all clusters have the desired group_size
        while any(len(cluster) < group_size for cluster in clusters):
            # Snake order: left to right for even rounds, right to left for odd rounds
            for cluster_index in range(num_clusters):
                # Reverse order for odd rounds
                if len(clusters[0]) % 2 != 0:
                    cluster_index = num_clusters - 1 - cluster_index

                # Skip this cluster if it's already full
                if len(clusters[cluster_index]) >= group_size:
                    continue

                # Get a sorted list of indices based on distances (closest first)
                sorted_indices = temp_matrix.iloc[:, cluster_index].sort_values().index.tolist()

                # Find the closest valid index that satisfies cannot-link constraints
                for candidate_index in sorted_indices:
                    # Check if the candidate_index violates any cannot-link constraints
                    if not any(
                        candidate_index in cl_graph.get(existing_index, set())
                        for existing_index in clusters[cluster_index]
                    ):
                        # Assign the valid candidate to the cluster
                        clusters[cluster_index].append(candidate_index)

                        # Remove the selected index from the distance matrix
                        temp_matrix = temp_matrix.drop(index=candidate_index)
                        break

        return clusters

    def get_neighborhood_centers(self, df , neighborhoods):
        """Parameters:
        df (pd.DataFrame): DataFrame containing the data points.
        neighborhoods: output of fucntion preprocess_constraints 
        
        Returns:
        neighborhood_centers (df): DataFrame containing the center of each neighborhood. The row index corresponds to the list (neighborhoods)
        index.
        """
        neighborhood_centers = pd.DataFrame(columns=df.columns)  # Initialize with column names

        for neighborhood in neighborhoods:
            subset_df = self.df.loc[neighborhood, :]  # Select rows from df based on neighborhood indices
            center = self.get_cluster_centers(subset_df)  # Calculate the cluster center
            
            # Convert the center (which could be a Series or DataFrame) into a DataFrame and concatenate
            neighborhood_centers = pd.concat([neighborhood_centers, center], ignore_index=True)
            
        return neighborhood_centers

    def get_cluster_centers(self, df):
        means = {}
        
        for column in df.columns:
            if column.startswith('hom'):
                # Regular mean for integer columns
                means[column] = df[column].mean()
            
            elif column.startswith('hot'):
                # Mean for one-hot encoded columns
                values = df[column].apply(lambda x: np.where(x == 1.0)[0][0] if 1.0 in x else None).dropna()
                array_length = len(df[column].iloc[0])
                result_array = [0] * array_length
                
                if len(values) > 0:
                    mode = values.mode()
                    if len(mode) > 1:
                        selected_index = np.random.choice(mode)
                    else:
                        selected_index = mode[0]
                    
                    result_array[selected_index] = 1
                means[column] = result_array
            
            elif column.startswith('mult'):
                # Mean for multi-hot encoded columns
                summed = np.sum(df[column].tolist(), axis=0)
                row_count = len(df)
                avg = np.round(summed / row_count).astype(int)

                    # Modify logic to use 0.5 as the threshold for setting the value to 1
                avg = (summed / row_count >= 0.5).astype(int)

                    # If all values in the average are 0
                if np.sum(avg) == 0:
                    non_zero_positions = np.where(summed > 0)[0]
                    if len(non_zero_positions) > 0:
                        random_indices = np.random.choice(non_zero_positions, size=1, replace=False)
                        avg[random_indices] = 1
                           
                means[column] = avg
        
        # Convert the means dictionary into a DataFrame
        means_df = pd.DataFrame({col: [val] if isinstance(val, (int, float)) else [val] for col, val in means.items()})
        
        return means_df
    

 
if __name__ == "__main__":

    df_A1 = testData.df_A1
    df_A2 = df_A1.iloc[:12]
    model = custom_PCKMeans(df = df_A1,
                            group_size = 3,
                            n_clusters = 3,
                            max_iter = 100,
                            ml_list = [],
                            cl_list= [])
    
    final_grouping, grouping_solution, balanced_clusters = model.fit()
    print("LP OUTPUT :", grouping_solution)
    print("FINAL GROUPING: ", final_grouping)
    print("BALANCED CLUSTERS :", balanced_clusters)
    #print("NEIGHBORHOODS :", neighborhood_clusters)
   
