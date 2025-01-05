from faker import Faker
import pandas as pd
import numpy as np
import random
import pytest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.algorithms.aco_mixed.aco  import ACO
from src.algorithms.aco_mixed.graph import Graph 
from src.algorithms.aco_mixed.aco import ACO 


@pytest.fixture
def create_names():
    # Create faker instance
    fake = Faker()
    fake.seed_instance(42)
    random.seed(42)

    # Create df with 30 random names
    df_names = [fake.name() for _ in range(18)]
    return pd.DataFrame(df_names, columns=['Name'])

@pytest.fixture
def feature_hom(): 
    values_hom = [x for x in [1, 1, 2, 2, 3, 3]*3] 
    return pd.DataFrame({'hom': values_hom})
    

@pytest.fixture 
def feature_het():
    values_het = [x for x in [1, 3] * 9]
    return pd.DataFrame({'het_hot': values_het }) 
 
       
@pytest.fixture
def feature_hot(): 
    return pd.DataFrame({'hot_hom': [[1, 0, 0]] * 6 + [[0, 1, 0]] * 6 + [[0, 0, 1]] * 6})
    
@pytest.fixture
def create_df(create_names, feature_hom, feature_het, feature_hot):
    # df created, such that there is a perfect solution 
    df = pd.concat([create_names, feature_hom, feature_het, feature_hot], axis = 1)
    return df 


@pytest.fixture
def grouped_df(create_df):
    graph = Graph(create_df)
    aco = ACO(graph, num_ants=10, num_iterations=10, group_size=2)
    best_path, best_solution = aco.run()
    grouped_df = aco.add_groups(best_path, create_df)
    return grouped_df 

  
def test_check_homogeneity(grouped_df):
    grouped = grouped_df.groupby("Group")["hom"]
    assert grouped.nunique().eq(1).all()

def test_check_heterogenity(grouped_df):
    grouped = grouped_df.groupby("Group")["het_hot"]
    assert grouped.nunique().eq(2).all()

def test_check_hot_hom(grouped_df): 
    grouped = grouped_df.groupby("Group")["hot_hom"]
    for group, values in grouped:
        arrays = np.array(values.tolist())
        assert np.all(arrays == arrays[0])
   







