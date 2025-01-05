import pandas as pd
import numpy as np
import pandas as pd
from sklearn import metrics
import pulp
from scipy.spatial.distance import cdist
from scipy.spatial.distance import squareform
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from Data import testData



def get_random_cluster_ceters(df, n_clusters):
    cluster_centers = df.sample(n=n_clusters, replace=False, random_state = 1)

    return cluster_centers

# Returns a #members * #n_clusters dataframe with the distances from each data point to all cluster_centers
def calculate_distance_matrix_to_cluster_centers(df, cluster_centers):
    # Check if the column "cluster" is in the df, if so drop it
    if "cluster" in df.columns:
        df = df.drop(columns=["cluster"])
    if "cluster" in cluster_centers.columns:
        cluster_centers = cluster_centers.drop(columns=["cluster"])
    
    homogenous_distances = (cdist(df.loc[:, df.columns.str.startswith('hom')], cluster_centers.loc[:, cluster_centers.columns.str.startswith('hom')], 'euclidean'))**2
    heterogenous_distances = cdist((df.loc[:, df.columns.str.startswith('het')]),(cluster_centers.loc[:, df.columns.str.startswith('het')]),ninja_distance)
    one_hot_distances = get_one_hot_distances(df, cluster_centers)  

    distance_matrix = np.sqrt(homogenous_distances + heterogenous_distances + one_hot_distances)

    return distance_matrix



""" This function uses the output of the ilp_solver to calculate the mean of each cluster, to get new cluster_centers. 
"""
def get_cluster_centers(df, n_clusters):
    new_cluster_centers = pd.DataFrame(index= range(n_clusters),columns=df.columns)    

    for i in range(n_clusters):
        # Filter rows where "cluster" equals i, and only include columns starting with 'hom'
        homogenous_mean = df.loc[df["cluster"] == i, df.columns.str.startswith('hom')].mean()
        heterogenous_mean = df.loc[df["cluster"] == i, df.columns.str.startswith('het')].mean()
        one_hot_mean = get_one_hot_mean(df.loc[df["cluster"] == i, df.columns.str.startswith('hot')])
        
        new_cluster_centers.loc[i, homogenous_mean.index] = homogenous_mean
        new_cluster_centers.loc[i, heterogenous_mean.index] = heterogenous_mean
        new_cluster_centers.loc[i, one_hot_mean.index] = one_hot_mean

    # Cast specific columns to float after assignment
    hom_columns = new_cluster_centers.columns[new_cluster_centers.columns.str.startswith('hom')]
    het_columns = new_cluster_centers.columns[new_cluster_centers.columns.str.startswith('het')]

    new_cluster_centers[hom_columns] = new_cluster_centers[hom_columns].astype(float)
    new_cluster_centers[het_columns] = new_cluster_centers[het_columns].astype(float)
    
    return new_cluster_centers.drop(columns="cluster")




#def assigne_data_points_to_clustercenter(df, cluster_centers):


""" Assignes each datapoint to a  cluster, based on the distance_matrix 
Keyword arugemtns:
df -- rows represent the data points(by index), columns represent the clusters. The values represent the distance from data point x_i
to cluster center c_i
"""
def lp_solver(distance_matrix, group_size):
    # Initialize the ILP problem
    prob = pulp.LpProblem("MinimizeColumnSums", pulp.LpMinimize)

    # Define decision variables
    variables = pulp.LpVariable.dicts("x", ((i, j) for i in range(distance_matrix.shape[0]) for j in range(distance_matrix.shape[1])), cat="Binary")

    # Objective function: Minimize the total sum of chosen elements
    prob += pulp.lpSum(distance_matrix.iloc[i, j] * variables[(i, j)] for i in range(distance_matrix.shape[0]) for j in range(distance_matrix.shape[1]))

    # Constraint 1: Select exactly 3 values in each column FIXME 3 is the groupsize variable
    for j in range(distance_matrix.shape[1]):
        prob += pulp.lpSum(variables[(i, j)] for i in range(distance_matrix.shape[0])) == group_size

    # Constraint 2: Each row can be used at most once
    for i in range(distance_matrix.shape[0]):
        prob += pulp.lpSum(variables[(i, j)] for j in range(distance_matrix.shape[1])) <= 1

    # Solve the ILP
    prob.solve()

    # Extract the chosen values
    chosen_values = [(i, j) for i in range(distance_matrix.shape[0]) for j in range(distance_matrix.shape[1]) if pulp.value(variables[(i, j)]) == 1]
    assigned_clusters = pd.DataFrame(chosen_values, columns=['index', 'cluster'])
    
    return assigned_clusters




#one_hot_distance = cdist(df.loc[:, df.columns.str.startswith('hot')], cluster_centers.loc[:, df.columns.str.startswith('hot')], 'hamming')
#squareform(res)
#print(pd.DataFrame(squareform(res), index=df.index, columns= df.index))
#print(pd.DataFrame(one_hot_distance))



# FIXME max_feature_distance could have different values for different features
def ninja_distance(x,y):
        max_feature_distance =  2
        return (np.sum(np.abs(np.abs(x - y) - max_feature_distance)**2))

def hamming_distance(a, b):
    # Ensure the input arrays have the same shape
    if a.shape != b.shape:
        raise ValueError("Input arrays must have the same shape")
    
    # Calculate element-wise Hamming distance and sum it up
    return np.sum(a != b)
          


def get_one_hot_distances(df, cluster_centers, weight = 2):
    result = pd.DataFrame(0, index=range(len(df)), columns=range(len(cluster_centers)))

    for col in df.columns:
        if col.startswith("hot"):
            dist_matrix_one_hot = pd.DataFrame(cdist(np.vstack(df[col]),np.vstack(cluster_centers[col]), "jaccard"))
            dist_matrix_one_hot *= weight
            result += (dist_matrix_one_hot**2) 
    return result



#print(pd.DataFrame(cdist(df.loc[:, df.columns.str.startswith('hom')], cluster_centers.loc[:, df.columns.str.startswith('hom')], 'sqeuclidean')))
#distance_matrix = calculate_distance_matrix_to_cluster_centers(df, cluster_centers).apply(np.sqrt)
#df = lp_solver(distance_matrix,df)


# Function to calculate the most common permutation in each column
def get_one_hot_mean(df):
    # Dictionary to store the most common permutation for each column
    common_permutations = {}

    for col in df.columns:
        # Convert lists to tuples for easier counting
        counts = df[col].apply(tuple).value_counts()
        # Get the most common permutation(s)
        max_count = counts.max()
        most_common = counts[counts == max_count].index
        
        # Select a random one if there's a tie and ensure it's a list
        most_common_choice = np.random.choice(most_common)  # This is now a tuple
        common_permutations[col] = list(most_common_choice)  # Convert to list
        
    # Convert the dictionary to a Series (1D)
    return pd.Series(common_permutations)

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

#######TEST#######
"""

print((df_1.loc[:, df_1.columns.str.startswith('hom')]).shape)
print((new_cluster_centers.loc[:, new_cluster_centers.columns.str.startswith('hom')]).shape)

print((df_1.loc[:, df_1.columns.str.startswith('hom')]).dtypes)
print((new_cluster_centers.loc[:, new_cluster_centers.columns.str.startswith('hom')]).dtypes)

print("OLD",(cluster_centers.loc[:, cluster_centers.columns.str.startswith('hom')]).dtypes 


converged = False
df = testData.df_6
n_clusters = 10
max_iter = 0
iteration_count = 0

#numbers = np.tile(np.arange(0, 10), 3)  # np.arange(0, 11) gives numbers from 0 to 10
#np.random.shuffle(numbers)  # Shuffle the numbers randomly

# Create the DataFrame with random clusters
#df["cluster"] = numbers

# Initialize cluster centers
cluster_centers = get_cluster_centers(df,n_clusters)

empty_df = pd.DataFrame(columns=df.columns)
empty_df["cluster"] = None

for iteration in range(max_iter):
    
    # Calculate first distance matrix
    distance_matrix = calculate_distance_matrix_to_cluster_centers(df, cluster_centers)       
    # Find best solution to split in groups of size 3
    assigned_clusters = lp_solver(distance_matrix, df)
    df["cluster"] = assigned_clusters["cluster"]
    prev_cluster_centers = cluster_centers

    # Calculate new clustercenters from each group of 3
    cluster_centers = get_cluster_centers(df, n_clusters)
    iteration_count += 1
    rows_to_add = df.loc[df["cluster"] == 1].copy()  # Make sure to copy
    
    # Add a new column 'iteration' that holds the current iteration number
    rows_to_add['iteration'] = iteration
    
    # Concatenate the rows with empty_df, reset index to keep it sequential
    empty_df = pd.concat([empty_df, rows_to_add])    

    if (compare_data_frames(prev_cluster_centers,cluster_centers, n_clusters)):
        break


#sh_coefficent = metrics.silhouette_score(df[["hom_1", "hom_2", "het_1", "het_2"]], df["cluster"], metric='euclidean')


#print(empty_df.sort_values(by="iteration"))

"""