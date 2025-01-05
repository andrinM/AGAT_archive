from multiprocessing import Process
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, value
import numpy as np

def group_distance_minimization(distance_matrix, g, k, process_id):
    """
    Solves the grouping problem using linear programming.

    Parameters:
        distance_matrix (list of list of float): nxn distance matrix.
        g (int): Number of groups.
        k (int): Number of members in each group.
        process_id (int): Process ID for logging.
    
    Returns:
        dict: Optimal group assignments for each item.
    """
    print(f"Process {process_id} started.")
    n = len(distance_matrix)
    assert n % k == 0, "n must be divisible by k"
    assert g == n // k, "g must equal n/k"

    # Create the LP problem
    prob = LpProblem(f"Group_Distance_Minimization_{process_id}", LpMinimize)

    # Decision variables: x[i][j] = 1 if item i is assigned to group j, else 0
    x = [[LpVariable(f"x_{i}_{j}", cat="Binary") for j in range(g)] for i in range(n)]

    # Auxiliary variables for pairwise distances within groups
    z = [[[LpVariable(f"z_{i}_{j}_{m}", cat="Binary") for m in range(g)] for j in range(n)] for i in range(n)]

    # Objective function: Minimize the total distance within groups
    prob += lpSum(
        distance_matrix[i][j] * z[i][j][m]
        for m in range(g) for i in range(n) for j in range(n) if i != j
    )

    # Constraint 1: Each item is assigned to exactly one group
    for i in range(n):
        prob += lpSum(x[i][m] for m in range(g)) == 1

    # Constraint 2: Each group has exactly k members
    for m in range(g):
        prob += lpSum(x[i][m] for i in range(n)) == k

    # Constraint 3: Define z[i][j][m] to link pairwise group membership
    for m in range(g):
        for i in range(n):
            for j in range(n):
                if i != j:
                    prob += z[i][j][m] <= x[i][m]
                    prob += z[i][j][m] <= x[j][m]
                    prob += z[i][j][m] >= x[i][m] + x[j][m] - 1

    # Solve the problem
    prob.solve()

    # Extract the results
    assignments = {i: None for i in range(n)}
    for i in range(n):
        for m in range(g):
            if value(x[i][m]) == 1:
                assignments[i] = m

    print(f"Process {process_id} completed.")
    print(f"Group assignments for process {process_id}: {assignments}")


if __name__ == "__main__":
    # Example distance matrix (symmetric)
    distance_matrix_1 = np.random.rand(10, 10)
    np.fill_diagonal(distance_matrix_1, 0)  # Ensure diagonal is zero
    distance_matrix_2 = np.random.rand(10, 10)
    np.fill_diagonal(distance_matrix_2, 0)  # Ensure diagonal is zero

    g = 5
    k = 2

    # Create processes for two LP problems
    process_1 = Process(target=group_distance_minimization, args=(distance_matrix_1, g, k, 1))
    process_2 = Process(target=group_distance_minimization, args=(distance_matrix_2, g, k, 2))

    # Start processes
    process_1.start()
    process_2.start()

    # Wait for processes to finish
    process_1.join()
    process_2.join()

    print("Both processes have completed.")
