import pandas as pd
import numpy as np
from sklearn import metrics
from scipy.spatial.distance import cdist
from scipy.spatial.distance import squareform
import sys
import os
import itertools
import random
current_dir = os.getcwd()  
src_dir = os.path.abspath(os.path.join(current_dir, 'src'))  # Parent directory of import_destination (i.e., src)
sys.path.append(src_dir)
from Data import testData
from algorithms.distance_matrix import distance_matrix as dm
from algorithms.randomness.possible_groups import possible_groups
from algorithms.randomness.random_permutation import random_permutation

class Occurance_Ranking:
    def __init__(self, df, group_size, alpha, max_iter=100, n_solutions = 1, ml_list=[], cl_list=[]):
        """
        Initialize the custom_PCKMeans instance.

        Parameters:
        - df (pd.DataFrame): The dataset to be clustered.
        - alpha (int): The number of best pairwise groups considered for the occurance ranking.
        - max_iter (int): Maximum number of iterations for the optimization algorithm. Default is 100.
        - ml_list (list): Must-link constraints. Default is None.
        - cl_list (list): Cannot-link constraints. Default is None.
        """
        self.df = df  # Store the dataset
        self.group_size = group_size
        self.alpha = alpha  # Number of clusters
        self.max_iter = max_iter  # Maximum number of iterations
        self.n_solutions = n_solutions
        self.ml_list = ml_list if ml_list is not None else []  # Must-link constraints
        self.cl_list = cl_list if cl_list is not None else []  # Cannot-link constraints
        

    def fit(self):

        distance_matrix = dm(self.df, self.df)
        ml_graph, cl_graph, neighborhoods = self.preprocess_constraints(nl = self.ml_list, cl= self.cl_list, n = len(self.df))

        # Create dictionary with all possible groups (with respect to cannot-link constraint) 
        all_possible_groups = self.possible_groups_distances(distance_matrix, self.group_size)

        # Get dictionary with member ID as key and number of occurances in the top alpha groups
        member_occurance = self.get_fittest_members(possible_groups=all_possible_groups, alpha=self.alpha)

        # Sort the member_occurance acsending (the one with the least occurances firts)
        sorted_member_occurance = dict(sorted(member_occurance.items(), key=lambda item: item[1], reverse=False))

        # Create greedily an initiale grouping based on the member_fitness
        initiale_grouping = self.greedy_grouping(distance_matrix=distance_matrix, sorted_member_occurance= sorted_member_occurance,
                                        group_size =  self.group_size,
                                        ml_graph = ml_graph,
                                        cl_graph = cl_graph)

        best_solutions = self.random_permutation_ml_cl(grouping=initiale_grouping,
                                                       distance_matrix=distance_matrix,
                                                       n_runs=self.max_iter,
                                                       cl_graph=cl_graph,
                                                       ml_graph=ml_graph,
                                                       n_solutions=self.n_solutions)
        return best_solutions


    def random_occ_rank(distance_matrix, group_size, top_k_groups, n_iterations_optimization, n_solutions = 1):

        # Create all possible groups and there inner distance
        all_possible_groups, distance_list = possible_groups(distance_matrix= distance_matrix, group_size= group_size) 

        # Create a dic of each member and their number of occurences in the top_k_groups (k groups with the lowest inner distance) 
        member_fitness = get_top_k_member_counts(all_possible_groups,top_k_groups)

        # Sort the member_fitness acsending (unfitest firts)
        sorted_member_fitness = dict(sorted(member_fitness.items(), key=lambda item: item[1], reverse=False))

        # Create greedily an initiale grouping based on the member_fitness
        initiale_grouping = greedy_grouping(distance_matrix= distance_matrix, members = sorted_member_fitness,group_size= group_size)

        # Random permutate one two members for k times
        best_solutions = random_permutation(grouping = initiale_grouping,
                                                                distance_matrix= distance_matrix,
                                                                n_runs= n_iterations_optimization,
                                                                n_solutions = n_solutions)

        return best_solutions
    
    def random_permutation_ml_cl(self, grouping, distance_matrix, n_runs, cl_graph, ml_graph, n_solutions=1):
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

    def get_fittest_members(self, possible_groups, alpha):
        """
        Sorts the dictionary by values and calculates the count of each member
        across the top k tuples (based on the smallest values).

        Parameters:
        - possible_groups: Dictionary with tuples as keys and numeric values.
        - alpha: Number of top alpha groups to consider.

        Returns:
        - A dictionary with indices (member ID) as keys and their counts as values.
        """
        # Step 1: Sort the dictionary by its values
        sorted_items = sorted(possible_groups.items(), key=lambda x: x[1])
        
        # Step 2: Extract the top k tuples
        top_alpha_tuples = [key for key, _ in sorted_items[:alpha]]
        
        # Step 3: Count occurrences of each indice (member ID) in the top alpha tuples
        top_alpha_members = {}
        for tup in top_alpha_tuples:
            for num in tup:
                top_alpha_members[num] = top_alpha_members.get(num, 0) + 1
        
        return top_alpha_members

    def greedy_grouping(slef, distance_matrix, sorted_member_occurance, group_size, ml_graph, cl_graph):
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
    
    def possible_groups_distances(self, distance_matrix, group_size, cl_graph):
        """
        Computes the total pairwise distances for all possible groups of a given size 
        from a distance matrix, while adhering to cannot-link constraints.

        Args:
            distance_matrix (numpy.ndarray): A 2D array representing pairwise distances 
                                            between data points.
            group_size (int): The size of each group to be formed.
            cl_graph (dict): A dictionary representing cannot-link constraints. Keys 
                            are indices, and values are sets of indices that cannot 
                            be grouped together with the key.

        Returns:
            dict: A dictionary where keys are tuples representing valid groups 
                (indices of group members) and values are the total pairwise 
                distances within those groups.
        """
        
        # Step 1: Create a list of all indices from 0 to n-1
        all_indices = list(range(len(distance_matrix)))
        
        # Step 2: Generate all possible groups (combinations of indices of the given group size)
        all_groups = list(itertools.combinations(all_indices, group_size))

        # Step 3: Filter out groups that violate cannot-link constraints
        # For each group, check if any pair of indices violates the cannot-link graph
        filtered_groups = [
            group for group in all_groups
            if not any(
                other in cl_graph.get(item, set())  # Check if 'other' is in the cannot-link set for 'item'
                for item in group for other in group if item != other  # Compare all pairs in the group
            )
        ]

        # Step 4: Compute total pairwise distances for valid groups
        group_distances = {}
        for group in filtered_groups:
            group_indices = list(group)  # Convert the tuple of group indices to a list
            
            # Extract the submatrix corresponding to the group (rows and columns for group indices)
            submatrix = distance_matrix[np.ix_(group_indices, group_indices)]
            
            # Sum the pairwise distances within the group, excluding diagonal elements
            # np.triu(submatrix, k=1) extracts the upper triangular part of the matrix, excluding the diagonal
            total_distance = np.sum(np.triu(submatrix, k=1))
            
            # Store the total distance for the group in the dictionary
            group_distances[group] = total_distance

        # Step 5: Return the dictionary of valid groups and their corresponding total distances
        return group_distances
    
    def preprocess_constraints(ml, cl, n):
        "Create a graph of constraints for both must- and cannot-links"

        """ Represent the graphs using adjacency-lists. This creates two empty dictionaries of size n
        and with keys from 0 to n. Each key is associated with an empty set.
        Keyword arguments:
        ml -- list of must-links
        cl-- list of cannot-links
        n -- number of rows (data points)
        """
        ml_graph, cl_graph = {}, {}
        for i in range(n):
            ml_graph[i] = set()
            cl_graph[i] = set()

        def add_both(d, i, j):
            d[i].add(j)
            d[j].add(i)

        """ Key of dict acts as index of a data point. The set (value) represent the neighboors
        of the index. Example: 
        ml = [(1,2),(2,4)]
        ml_graph = {0:set(), 1:{2}, 2:{1,4}, 3:set(), 4:{2}}
        returns adjacency list
        """
        for (i, j) in ml:
            ml_graph[i].add(j)
            ml_graph[j].add(i)

        for (i, j) in cl:
            cl_graph[i].add(j)
            cl_graph[j].add(i)
        
        def dfs(i, graph, visited, component): # Depth First Search Algorithm
            visited[i] = True
            for j in graph[i]:
                if not visited[j]:
                    dfs(j, graph, visited, component)
            component.append(i)

        """ Keyword arguments:
        component -- list of connected nodes (data points)
        neighborhoods -- list of lists. Each within list is a component
        """
        # Run DFS from each node to get all the graph's components
        # and add an edge for each pair of nodes in the component (create a complete graph)
        # See http://www.techiedelight.com/transitive-closure-graph/ for more details
        visited = [False] * n
        neighborhoods = []
        for i in range(n): # traverse each data point (node)
            if not visited[i] and ml_graph[i]: # If ml_graph[i] has values then we get true otherwise false
                component = []
                dfs(i, ml_graph, visited, component)
                for x1 in component:
                    for x2 in component:
                        if x1 != x2:
                            ml_graph[x1].add(x2)
                neighborhoods.append(component) # neighborhoods is a list of lists. Each within list is a component
        
        """ This for loop adds nodes (data points) to the cl_graph if they have a transitive
        inference of a cannot-link constrans. It basically adds some of the must-links to the
        cl_graph, if they violate consistency 
        """
        for (i, j) in cl:
            for x in ml_graph[i]: # i is the key of ml_graph and x is the corresponding value
                add_both(cl_graph, x, j)

            for y in ml_graph[j]:
                add_both(cl_graph, i, y)

            for x in ml_graph[i]:
                for y in ml_graph[j]:
                    add_both(cl_graph, x, y)

        """ This for loop checks if any tuple is a must-link AND a cannot-link constraint. 
        If this is the case, an exception gets thrown.
        """
        for i in ml_graph: # iterate over the keys
            for j in ml_graph[i]:
                if j != i and j in cl_graph[i]:
                    raise ValueError('Inconsistent constraints between {} and {}'.format(i, j))

        return ml_graph, cl_graph, neighborhoods

   
