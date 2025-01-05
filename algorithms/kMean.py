import pandas as pd
import numpy as np
from matplotlib.animation import FuncAnimation
from sklearn.decomposition import PCA # reduces dimensions for visualization
import matplotlib.pyplot as plt

# Import testData
from Data import testData
members = testData.df

#features we want to cluster
features = ["f1", "f2"]

#data will be used for the clustering
data = members[features].copy()

# 1. Scale the data from 1 to 10

# Set the minimum value from each column to zero by subtracting
# the minimum value from each column. Then divide by the scale, multiply by 9 so we have 0-9,
#add +1 for 1-10.
data = (data - data.min()) / (data.max()- data.min()) * 9 + 1 

# 2. Initialzie random centroids
# Randomly sample one data point
def random_centroids(data, k):
    centroids = []
    for i in range(k):
        centroid = data.apply(lambda x: float(x.sample())) #returns pd series
        centroids.append(centroid) #add sampled centroid to list
    return pd.concat(centroids, axis = 1) #concats pd series into df



# 3. Label each data point
# distance is a df that holds each datapoints distance to the k randomly selected centroids
# columns: datapoints, rows: centroids
def get_labels(data, centroids):
    distances = centroids.apply(lambda x: np.sqrt(((data - x)**2).sum(axis = 1)))
    return(distances.idxmin(axis=1)) # Min value of each colum represents the cluster


# Calculate geom. mean to get the center of each cluster.
# We use arithmetic mean in logscale, fo efficency
# First we group each data point from data into its corresponding cluster.
# Then we calculate the geom. mean for each feature and this will be the new cluster
def new_centroids(data, labels, k): 
    return data.groupby(labels).apply(lambda x: np.exp(np.log(x).mean())).T

# Data Visualization

def plot_clusters(data, labels, centroids):
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(data)
    centroids_2d = pca.transform(centroids.T)

    plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, cmap='viridis')
    plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1], color='red', marker='X', edgecolor='black', label='Centroids')
    plt.legend()
    plt.show()


# Body of the kmean clustering algo.
max_iterations = 100
k = 5

centroids = random_centroids(data, k)
old_centroids = pd.DataFrame()
iteration = 1

while iteration < max_iterations and not centroids.equals(old_centroids):
    old_centroids = centroids

    labels = get_labels(data, centroids)
    centroids = new_centroids(data, labels, k)
    #plot_clusters(data, labels, centroids, iteration)
    iteration += 1
print(labels)
#members["cluster"] = labels
#print(members.sort_values(by="cluster"))