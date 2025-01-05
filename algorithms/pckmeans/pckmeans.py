""" This is an implementation of a PCKMeans algorithm
"""

import numpy as np
import math

from exceptions import EmptyClustersException
from constraints import preprocess_constraints


class PCKMeans:
    def __init__(self, X, homogenous_features, heterogenous_features,one_hot_features, n_clusters=3, max_iter=100, w=1):
        self.X = X
        self.homogenous_features = homogenous_features
        self.heterogenous_features = heterogenous_features
        self.one_hot_features = one_hot_features
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.w = w
        

    """ The first iteration takes the initialized clusters. Then each data point gets assigned
    to a cluster. We then safe those cluster_centers in prev_cluster_centers. Then for each cluster
    We calculate a new cluster_center, based on the freshly asigned data points (mean of those).
    This is done with the function _get_cluster_centers. We then compare the new centers with the
    old ones.

    Keyword arguments:
    converged --This returns True if all elements in the difference array are within the defined tolerances
    (atol=1e-6 and rtol=0) of the corresponding elements in the zero array.
    If any element is outside this tolerance, it returns False.
    """
    def fit(self, y=None, ml=[], cl=[]):
        # Preprocess constraints
        ml_graph, cl_graph, neighborhoods = preprocess_constraints(ml, cl, self.homogenous_features.shape[0]) # homogenous_features.shape[0] = nrows

        # Initialize centroids
        cluster_centers = self._initialize_cluster_centers(self.X, neighborhoods)

        # Repeat until convergence
        for iteration in range(self.max_iter):
            # Assign clusters
            labels = self._assign_clusters(cluster_centers, ml_graph, cl_graph, self.w)
            
            # Estimate means
            prev_cluster_centers = cluster_centers
            # Get the new cluster_centers
            cluster_centers = self._get_cluster_centers(labels)

            # Check for convergence, this compares the difference with zero 
            #difference = (prev_cluster_centers - cluster_centers)
            converged = self.lists_are_close(prev_cluster_centers,cluster_centers)

            if converged: break

        self.cluster_centers_, self.labels_ = cluster_centers, labels

        return self
    
    """ This function returns a set of cluster_centers equal to the size of k.
        neighboorhood_centers is the mean of each center based on the feature values
        neighboorhood_size is the amount of data points per neighboorhood
        If #neighboorhods > k then select the k biggest neighborhoods
        If #neighboorhods > 0 then cluster_centers = neighborhood_centers
        If #neighboorhods < k then add k-#neighboorhods random centroids
    """
    def _initialize_cluster_centers(self, X, neighborhoods):
        neighborhood_centers = []
        for neighborhood in neighborhoods:
            if (isinstance(neighborhood, int)):
                print("MISSING BRACKETS")
            homog_mean = self.homogenous_features[neighborhood].mean(axis=0)
            heterog_mean = self.heterogenous_features[neighborhood].mean(axis=0)
            one_hot_mean = self.get_one_hot_mean(self.one_hot_features[neighborhood])
            centroid_i = np.atleast_1d(homog_mean).tolist() + np.atleast_1d(heterog_mean).tolist() + np.atleast_1d(one_hot_mean).tolist()
            neighborhood_centers.append(centroid_i)
            

        neighborhood_sizes = np.array([len(np.atleast_1d(neighborhood)) for neighborhood in neighborhoods])
        cluster_centers = []

        #neighborhood_centers is the mean of all data points from this neighborhood.
        if len(neighborhoods) > self.n_clusters:
            # Select K largest neighborhoods' centroids
            cluster_centers = [neighborhood_centers[i] for i in np.argsort(neighborhood_sizes)[::-1][:self.n_clusters]] #orderd descending and pick first n_cluster indices
        else:
            if len(neighborhoods) > 0:
                cluster_centers = neighborhood_centers
            else:
                random_indices = np.random.choice(range(0, len(X)-1), size=self.n_clusters, replace=False)
                cluster_centers = [X[i] for i in random_indices]

            # FIXME look for a point that is connected by cannot-links to every neighborhood set

            if len(neighborhoods) < self.n_clusters and len(neighborhoods) != 0:
                random_indices = np.random.choice(range(0, len(X)-1), size=self.n_clusters - len(neighborhoods), replace=False)
                remaining_cluster_centers = [X[i] for i in random_indices]
                cluster_centers = cluster_centers + remaining_cluster_centers # FIXME There can be two of the same centroids!
        return cluster_centers
    
    """Keyword arguments:
    X -- data set
    x_i -- current inspected data point
    centroids -- list of cluster_centers
    labels -- list #rows filled with -1
    c_i -- index of current cluster (calculate cost if x_i is in c_i)

    The ml_penalty gets calculated as follows: We look at all data points must-link with
    x_i (called y_i). Then for each y_i figure out if it allready has a
    cluster (if labels[y_i] != -1) and if it is the same cluster c_i that we currently
    look at. If y_i has allready a cluster and its not c_i, then we get a penalty.

    The cl_penalty gets added, if a data point who y_i cannot-link with x_i is already in cluster
    c_i
    """
    def _objective_function(self, x_i, centroids, c_i, labels, ml_graph, cl_graph, w):
        homogenous_distance = (self.homogenous_features[x_i] - self.get_homogenous_features(centroids[c_i]))**2
        
        heterogenous_distance = abs(abs(self.heterogenous_features[x_i]
                                    -self.get_heterogenous_features(centroids[c_i]))
                                    -np.max(self.heterogenous_features, axis=0))
    
        one_hot_distance = self.hamming_distance(self.one_hot_features[x_i],
                                        self.get_one_hot_feauters(centroids[c_i]))
        distance = np.sqrt(np.sum(homogenous_distance) + np.sum(heterogenous_distance) + one_hot_distance)
        """
        ml_penalty = 0
        for y_i in ml_graph[x_i]: # iterate over all must-links from data point x_i
            if labels[y_i] != -1 and labels[y_i] != c_i:
                ml_penalty += w

        cl_penalty = 0
        for y_i in cl_graph[x_i]:
            if labels[y_i] == c_i:
                cl_penalty += w
        """
        
        return distance #+ ml_penalty + cl_penalty
    
    # Helpers for the new Objective Function
    def hamming_distance(self, a, b):
        # Ensure the input arrays have the same shape
        if a.shape != b.shape:
            raise ValueError("Input arrays must have the same shape")
        
        # Calculate the Hamming distance
        return np.sum(a != b)

    def get_homogenous_features(self,datapoint ):
        return np.array(datapoint[0:self.homogenous_features.shape[1]])

    def get_heterogenous_features(self,datapoint ):
        return np.array(datapoint[self.homogenous_features.shape[1]:(self.homogenous_features.shape[1]
                                                        + self.heterogenous_features.shape[1])])

    def get_one_hot_feauters(self, datapoint):
        extracted_one_hot_features= np.array(datapoint[(self.homogenous_features.shape[1]+self.heterogenous_features.shape[1]):
                            (self.homogenous_features.shape[1]
                            + self.heterogenous_features.shape[1]
                            + self.one_hot_features.shape[1])])
        return(np.squeeze(extracted_one_hot_features))
    

    """ Keyword arguments:
    labels -- initially a list with #data points filled with -1, this list will have the assigned clusters
    index -- to randomlly choose a sequence of data point indices, to make the decision of when
    which data point gets clustered random.
    x_i -- current data point we want to asign a cluster
    c_i -- current cluster we look at.

    We itterate over the number of clusters (k) in the function. We calculate with the objective
    function the cost of x_i to each centroid. Then we take the minimum with argmin
    argmin assignes the cluster center index, which has the lowest cost from x_i to labels.
    """
    def _assign_clusters(self, cluster_centers, ml_graph, cl_graph, w):
        labels = np.full(self.homogenous_features.shape[0], fill_value=-1) # numpy array with #rows filled with -1
        index = list(range(self.homogenous_features.shape[0])) # list with 0,1 ...,n where n is the #rows
        np.random.shuffle(index)
        for x_i in index: # for every data point with every cluster
            labels[x_i] = np.argmin([self._objective_function(x_i, cluster_centers, c_i, labels, ml_graph, cl_graph, w) for c_i in range(self.n_clusters)]) #c_i with the smallest cost will be assigned to labels[x_i]
        
        # Handle empty clusters
        # See https://github.com/scikit-learn/scikit-learn/blob/0.19.1/sklearn/cluster/_k_means.pyx#L309
        n_samples_in_cluster = np.bincount(labels, minlength=self.n_clusters)
        empty_clusters = np.where(n_samples_in_cluster == 0)[0] #stores the indices of the empty clusters
        # distance_to_cluster_center is a list of data point indices. The index at position 0 is the data point, who has the highest distance to its assigned cluster_center
        if len(empty_clusters) > 0:
            distance_to_cluster_center = np.argsort(
            [self._objective_function(i, cluster_centers, labels[i], labels, ml_graph, cl_graph, w) for i in range(self.homogenous_features.shape[0])])

        for i, cluster_id in enumerate(empty_clusters):
            new_center = self.X[distance_to_cluster_center[i]]
            cluster_centers[cluster_id] = new_center
            labels[distance_to_cluster_center[i]] = cluster_id
        
        return labels
        
    """ Getting all elements from X that are in cluster i and calculate the mean. The mean is
    the centroid for the next iteration. This function returns an np array with new centroids
    """
    def _get_cluster_centers(self, labels):
        cluster_centers = []
        #return np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])
        for i in range(self.n_clusters):
            homog_mean = self.homogenous_features[labels == i].mean(axis=0)
            heterog_mean = self.heterogenous_features[labels == i].mean(axis=0)
            one_hot_mean = self.get_one_hot_mean(self.one_hot_features[labels == i])
            centroid_i = homog_mean.tolist() + heterog_mean.tolist() + one_hot_mean.tolist()
            cluster_centers.append(centroid_i)
        return cluster_centers

    

    def get_one_hot_mean(self, one_hot_features):
        # Sum of each position for all the selected inner arrays
        sum_selected_inner_arrays = np.sum(one_hot_features, axis=0)

        # Initialize one_hot_mean with zeros, having the same shape as a row in one_hot_features
        one_hot_mean = np.zeros_like(one_hot_features[0])

        # Iterate over each inner array and set the maximum index to 1
        for i in range(sum_selected_inner_arrays.shape[0]):
            max_index = np.argmax(sum_selected_inner_arrays[i])
            one_hot_mean[i, max_index] = 1

        return one_hot_mean

    def lists_are_close(self,a, b, tol=1e-9):
        for sublist_a, sublist_b in zip(a, b):
            if all(isinstance(item, float) for item in sublist_a + sublist_b):
                if not all(math.isclose(item_a, item_b, abs_tol=tol) for item_a, item_b in zip(sublist_a, sublist_b)):
                    return False
            else:
                if sublist_a != sublist_b:
                    return False
        return True