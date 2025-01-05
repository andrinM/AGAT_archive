import pandas as pd 
import time 
import os 
import sys
sys.path.insert(0, 'src')
from Data.testData import create_random_dataframe

# importing different algorithms

from algorithms.aco_complete.aco import ACO 
from algorithms.aco_complete.graph import Graph 
from algorithms. aco_complete.ant import Ant 

# Output of each algorithm should be from the form: run_time, best_distance

def run_aco(group_size):
    start_time = time.perf_counter()
    graph = Graph(members)
    aco = ACO(graph, num_ants=20, num_iterations=20, group_size=group_size)
    best_path, best_distance = aco.run()
    run_time = time.perf_counter() - start_time
    return run_time, best_distance 

def run_pck(group_size):
    start_time = time.perf_counter()

    run_time = time.perf_counter() - start_time
    return run_time,0 

def run_random(group_size):
    start_time = time.perf_counter()
    run_time = time.perf_counter() - start_time
    return run_time,0 


def run_algorithms(members):
    results = []
    avg_results = []
    iterations = 10

    for group_size in range(2, 5):
        time_dist_aco = {'run_times': [], 'distances': []}
        time_dist_pck = {'run_times': [], 'distances': []}
        time_dist_random = {'run_times': [], 'distances': []}
      
        for i in range(iterations):
            run_time_aco, distance_aco = run_aco(group_size)
            run_time_pck, distance_pck = run_pck(group_size)
            run_time_random, distance_random = run_random(group_size)

            time_dist_aco['run_times'].append(run_time_aco)
            time_dist_aco['distances'].append(distance_aco)
            time_dist_pck['run_times'].append(run_time_pck)
            time_dist_pck['distances'].append(distance_pck)
            time_dist_random['run_times'].append(run_time_random)
            time_dist_random['distances'].append(distance_random)
           
            results.extend([
                {'algorithm': 'ACO', 'group_size': group_size, 'time': run_time_aco, 'distance': distance_aco},
                {'algorithm': 'PCK', 'group_size': group_size, 'time': run_time_pck, 'distance': distance_pck},
                {'algorithm': 'Random', 'group_size': group_size, 'time': run_time_random, 'distance': distance_random}
            ])

        avg_run_time_aco = sum(time_dist_aco['run_times']) / iterations
        avg_run_time_pck = sum(time_dist_pck['run_times']) / iterations
        avg_run_time_random = sum(time_dist_random['run_times']) / iterations

        avg_distance_aco = sum(time_dist_aco['distances']) / iterations
        avg_distance_pck = sum(time_dist_pck['distances']) / iterations
        avg_distance_random = sum(time_dist_random['distances']) / iterations
       

        avg_results.extend([
                {'algorithm': 'ACO', 'group_size': group_size, 'average_time': avg_run_time_aco, 'average_distance': avg_distance_aco},
                {'algorithm': 'PCK', 'group_size': group_size, 'average_time': avg_run_time_pck, 'average_distance': avg_distance_pck},
                {'algorithm': 'Random', 'group_size': group_size, 'average_time': avg_run_time_random, 'average_distance': avg_distance_random}
            ])

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by=['group_size', 'algorithm'])

    avg_results_df = pd.DataFrame(avg_results)
    avg_results_df = avg_results_df.sort_values(by=['group_size', 'algorithm'])
  
    csv_files = 'csv_files'
    file_path = os.path.join(csv_files, 'results_12_members.csv')
    results_df.to_csv(file_path, index=False)
    file_path = os.path.join(csv_files, 'avg_results_12_members.csv')
    avg_results_df.to_csv(file_path, index=False)
   

members = create_random_dataframe(n_rows = 12, hom = 2, het = 2, feature_range = 3, hot_het = 2, hot_hom = 2)
run_algorithms(members)