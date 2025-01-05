from pckmeans import PCKMeans
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from Data import testData
import pandas as pd

df = testData.df_6
homogenous_features = df[["f1", "f2"]].to_numpy()
heterogenous_features = df[["f3", "f4"]].to_numpy()
df_one_hot = df[["f5", "f6"]]

# Create a new NumPy array with paired arrays
pair_array_list = []
for index, row in df_one_hot.iterrows():
    pair_array_list.append([row['f5'], row['f6']])  # Pairing the arrays

# Convert the list of pairs to a NumPy array
one_hot_features = np.array(pair_array_list)

# Creating X
#homog_heterog = np.stack((homogenous_features, heterogenous_features), axis= 1).tolist()
homog_heterog = np.hstack((homogenous_features, heterogenous_features)).tolist()
one_hot_features_list = one_hot_features.tolist()
X = [a + b for a, b in zip(homog_heterog, one_hot_features_list)]


clusterer = PCKMeans(X, homogenous_features, heterogenous_features,one_hot_features, 10)
clusterer.fit()
result = pd.DataFrame(clusterer.labels_)
df_result = df
df_result["cluster"] = result
print(df_result.sort_values(by = "cluster"))



