from faker import Faker
import pandas as pd
import numpy as np
import random

# Create faker instance
fake = Faker()
fake.seed_instance(42)
random.seed(42)
np.random.seed(42)
# Create df with 30 random names
df_names = [fake.name() for _ in range(30)]
df_names = pd.DataFrame(df_names, columns=['Name'])

# Feature 1: What are your goals for this project/course? 1: grade 6, 2: grade 5, 3: just pass
df_feature1 = [random.randint(1,3) for _ in range(30)]
df_feature1 = pd.DataFrame(df_feature1, columns=['hom_1'])

# Feature 2: What are your preferences regarding where group meetings should take
# place?
# 1: online, 2: In-Person 3: Hybrid
df_feature2 = [random.randint(1,3) for _ in range(30)]
df_feature2 = pd.DataFrame(df_feature2, columns=['hom_2'])

# Feature 3: What skills and strengths would you bring to your team?
# 1: structured planning, 2: presentation skills 3: writing skills
df_feature3 = [random.randint(1,3) for _ in range(30)]
df_feature3 = pd.DataFrame(df_feature3, columns=['het_1'])


#Create one df out of the previously defined df's
df = pd.concat([df_names, df_feature1, df_feature2, df_feature3], axis =1)

# Feature 4: to test heterogenity 
df_feature4 = [random.randint(1,3) for _ in range(30)]
df_feature4 = pd.DataFrame(df_feature4, columns=['het_2'])

# df to test aco algorithm 
df_hom = pd.concat([df_names, df_feature1, df_feature2, df_feature3], axis =1)

# Feature 5
f_5 = [np.random.permutation([1, 0, 0]) for _ in range(30)]

# Feature 6
f_6 = [np.random.permutation([1, 0, 0]) for _ in range(30)]

def create_random_dataframe(n_rows, hom, het, feature_range, one_hot, random_state=False):
    # If random_state is True, set a fixed seed for reproducibility
    if random_state:
        np.random.seed(42)
    
    # Creating 'hom' columns with random numbers between 1 and feature_range
    hom_data = {
        f'hom_{i+1}': np.random.randint(1, feature_range + 1, size=n_rows)
        for i in range(hom)
    }
    
    # Creating 'het' columns with random numbers between 1 and feature_range
    het_data = {
        f'het_{i+1}': np.random.randint(1, feature_range + 1, size=n_rows)
        for i in range(het)
    }
    
    # Creating 'one_hot' columns with random permutations of [1, 0, 0]
    one_hot_data = {
        f'hot_{i+1}': [np.random.permutation([1, 0, 0]) for _ in range(n_rows)]
        for i in range(one_hot)
    }
    
    # Combine all the data into a single dictionary
    data = {**hom_data, **het_data, **one_hot_data}
    
    # Create the DataFrame
    df = pd.DataFrame(data)
    
    return df

def generate_dataframe(n_rows, feature_list, random_state = False):
    """
    Generate a DataFrame with n_rows rows and len(feature_list) columns based on the feature_list input.

    Args:
        n_rows (int): Number of rows for the DataFrame.
        feature_list (list): List of tuples (x, y), where x is a feature type and y is an integer.

    Returns:
        pd.DataFrame: Generated DataFrame with specified columns and values.
    """
    if random_state:
        random.seed(42)
        np.random.seed(42)
    # Initialize an empty dictionary to store column data
    data = {}

    # Track counts for enumerating each feature category
    feature_counts = {}

    for feature, y in feature_list:
        # Increment the feature count to ensure proper enumeration
        if feature not in feature_counts:
            feature_counts[feature] = 1
        else:
            feature_counts[feature] += 1

        # Create the column name with enumeration
        col_name = f"{feature}_{feature_counts[feature]}"

        if feature in ["hom_", "het_ninja", "het_hot"]:
            # Random integers from 1 to y
            data[col_name] = np.random.randint(1, y + 1, size=n_rows)

        elif feature in ["hot_hom", "hot_het"]:
            # Arrays of length y with all zeros and one 1 at a random position
            data[col_name] = [np.eye(1, y, k=np.random.randint(0, y)).flatten() for _ in range(n_rows)]

        elif feature in ["mult_hot_het", "mult_hot_hom"]:
            # Arrays of length y with random 0s and 1s
            data[col_name] = [np.random.choice([0, 1], size=y).tolist() for _ in range(n_rows)]

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(data)

    return df

# Create dataframe with 6 features
df_6 = pd.concat([df_feature1, df_feature2, df_feature3, df_feature4],axis=1)
df_6['f5'] = f_5
df_6["f6"] = f_6
# print(df_6)

# Feature 7: How would you assess your skills?
# 1: very low, 2: ok, 3: very good 
# for testing purposes the feature contains exactly 10 1's, 2's and 3's (ensure that evry group has 1,2,3)
df_feature7 = [x for x in [1, 2, 3] for _ in range(10)]
df_feature7 = pd.DataFrame(df_feature7, columns=['f7'])

# df to test aco for heterogen groups 
df_het = pd.concat([df_names, df_feature7], axis =1)

# df to test aco mixed: 
# Feature 8 and Feature 9: heterogen 
# Feature 10: homogen 
# df is created such that optimal groups have the same values in feature 10 and 
# different values in feature 8 and 9 
df_feature8 = [x for x in [1, 2, 3] * 10]
df_feature8 = pd.DataFrame(df_feature8, columns=['f8'])

df_feature9 = [x for x in [0, 1, 2] * 10]
df_feature9 = pd.DataFrame(df_feature9, columns=['f9'])

df_feature10 = [x for x in [1,2,3,4,5,6,7,8,9,10] for _ in range(3)]
df_feature10 = pd.DataFrame(df_feature10, columns=['f10'])

df_hom_het = pd.concat([df_names, df_feature8, df_feature9, df_feature10], axis=1)

# IMPORTENT DO NOT DELET!
tuple_list = [("hom_", 3), ("hom_", 4), ("hot_het", 3),("hot_het", 4), ("mult_hot_hom", 4), ("mult_hot_het", 4)]

df_A1 = generate_dataframe(n_rows= 60, feature_list = tuple_list)

df_A2 = generate_dataframe(n_rows= 180, feature_list = tuple_list)

df_A3 = generate_dataframe(n_rows= 420, feature_list = tuple_list)

df_A4 = generate_dataframe(n_rows= 900, feature_list = tuple_list)

