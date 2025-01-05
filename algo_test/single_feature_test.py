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
    df_names = [fake.name() for _ in range(30)]
    return pd.DataFrame(df_names, columns=['Name'])

@pytest.fixture
def df_hom(create_names):
    # Feature to test if algorithm finds perfect solution for homogeneous groups 
    # Feature contains exactly 6 times each number 
    values_hom = [x for x in [1, 2, 3, 4, 5] for _ in range(6)] 
    random.shuffle(values_hom)
    feature_hom = pd.DataFrame({'hom': values_hom})
    return pd.concat([create_names, feature_hom], axis=1) 

@pytest.fixture 
def df_het(create_names):
    # Feature to test if algorithm finds perfect solution for heterogeneous groups 
    # Feature contains exactly 10 times each number
    values_het = [x for x in [1, 3, 5] * 10]
    random.shuffle(values_het) 
    feature_het = pd.DataFrame({'het_hot': values_het }) 
    return pd.concat([create_names, feature_het], axis = 1)
       
@pytest.fixture
def df_hot_hom(create_names):
    # Feature to test if algorithm finds perfect solution for one hot homogeneous groups 
    feature_hot_hom = pd.DataFrame({'hot_hom': [[0, 0, 1]] * 12 + [[0, 1, 0]] * 9 + [[1, 0, 0]] * 9})
    return pd.concat([create_names,feature_hot_hom], axis = 1 )
@pytest.fixture
def df_hot_het(create_names): 
    # Feature to test if algorithm finds perfect solution for one hot heterogeneous groups
    feature_hot_het = pd.DataFrame({'hot_het': [[1, 0, 0], [0, 1, 0], [0, 0, 1]] * 10})
    return pd.concat([create_names,feature_hot_het], axis = 1)

def set_up(num_ants, num_iterations,df):
    graph = Graph(df)
    aco = ACO(graph, num_ants, num_iterations, group_size = 3)
    best_path, best_solution = aco.run()
    return aco.add_groups(best_path,df)

  
def test_check_homogeneity(df_hom):
    members_grouped = set_up(10,10,df_hom)
    members_grouped.sort_values(['Group'])
    grouped = members_grouped.groupby("Group")["hom"]
    assert grouped.nunique().eq(1).all()

def test_check_heterogenity(df_het):
    members_grouped = set_up(40,20,df_het)
    grouped = members_grouped.groupby("Group")["het_hot"]
    assert grouped.nunique().eq(3).all()

def test_check_hot_hom(df_hot_hom): 
    members_grouped = set_up(10,10,df_hot_hom)
    members_grouped.sort_values(['Group'])
    grouped = members_grouped.groupby("Group")["hot_hom"]
    for group, values in grouped:
        arrays = np.array(values.tolist())
        assert np.all(arrays == arrays[0])

def test_check_hot_het(df_hot_het): 
    members_grouped = set_up(10,10,df_hot_het)
    members_grouped.sort_values(['Group'])
    grouped = members_grouped.groupby("Group")["hot_het"]
    expected_values = [tuple([0, 0, 1]), tuple([0, 1, 0]), tuple([1, 0, 0])]

    for group, values in grouped:
        group_values = values.apply(tuple).tolist()
        group_values_sorted = sorted(group_values)
        expected_values_sorted = sorted(expected_values)
        print(f"Group {group}: group_values_sorted = {group_values_sorted}")
        print(f"Group {group}: expected_values_sorted = {expected_values_sorted}")
        assert group_values_sorted == expected_values_sorted    




