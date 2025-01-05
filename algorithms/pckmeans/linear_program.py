import numpy as np
import pandas as pd
import pulp
import sys
import os
current_dir = os.getcwd() 
src_dir = os.path.abspath(os.path.join(current_dir, 'src'))  # Parent directory of import_destination (i.e., src)
sys.path.append(src_dir)
from Data import testData
from algorithms import functions

def solve_grouping(distance_matrix, k):
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
    solver = pulp.PULP_CBC_CMD(msg=1)
    status = problem.solve(solver)

    # If the problem is solved, extract the results
    if status == pulp.LpStatusOptimal:
        # Assign points to groups
        group_assignments = [-1] * n
        for i in range(n):
            for g in range(num_groups):
                if pulp.value(y[i, g]) == 1:
                    group_assignments[i] = g
                    break
        return group_assignments
    else:
        print(f"Problem Status: {pulp.LpStatus[status]}")
        print("No optimal solution found.")
        return None




df_A0 = testData.create_random_dataframe(n_rows= 21, hom=2, het=2, feature_range=3, 
                                         one_hot=2, random_state= True)

distance_matrix = functions.get_distance_matrix(df_A0, df_A0)
np.fill_diagonal(distance_matrix.values, 0)
distance_matrix = np.array(distance_matrix)

df_A0["cluster"] = solve_grouping(distance_matrix, 3)
print(functions.get_within_group_distance(df_A0, 7))
