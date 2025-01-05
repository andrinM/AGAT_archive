from sklearn.cluster import SpectralClustering
from group_size_extension import calculate_distance_matrix_to_cluster_centers, get_cluster_centers, lp_solver
import sys
import os
import numpy as np
import pandas as pd
import pulp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from Data import testData

def calculate_within_distance(df1,df2):
    within_distance_df = calculate_distance_matrix_to_cluster_centers(df1,df2)
    np.fill_diagonal(within_distance_df.values, 0)

    return within_distance_df.to_numpy().sum() / 2

# Load your data
df = testData.df_6

# Calculate the distance matrix
distance_matrix = calculate_distance_matrix_to_cluster_centers(df, df)

n_clusters = 10

np.fill_diagonal(distance_matrix.values, 1e6)

sc = SpectralClustering(n_clusters, affinity='precomputed', n_init=200,
                        assign_labels='cluster_qr').fit_predict(distance_matrix)


df["cluster"] = sc

within_group_distance = pd.DataFrame(columns=["distance"])
for i in range(n_clusters):
    within_group_distance.loc[i] = calculate_within_distance(df.loc[df["cluster"] == i],df.loc[df["cluster"] == i])

cluster_centers = get_cluster_centers(df, n_clusters)
new_distance_matrix = calculate_distance_matrix_to_cluster_centers(df,cluster_centers)

result = lp_solver(new_distance_matrix,3)
result = result.drop(columns="index")
print(result)
df["cluster"] = result

within_group_distance = pd.DataFrame(columns=["distance"])
for i in range(n_clusters):
    within_group_distance.loc[i] = calculate_within_distance(df.loc[df["cluster"] == i],df.loc[df["cluster"] == i])


print(df.sort_values(by="cluster"))
print(within_group_distance)


