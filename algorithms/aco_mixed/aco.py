import numpy as np
import itertools
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ant import Ant

# ACO (Ant Colony Optimization) class runs the algorithm to find the best path
class ACO:
    def __init__(self, graph, num_ants, num_iterations, group_size, must_links = None, decay=0.5, alpha=1.0):
        self.graph = graph
        self.num_ants = num_ants  # Number of ants in each iteration
        self.num_iterations = num_iterations  # Number of iteration
        self.group_size = group_size
        self.must_links = must_links
        self.decay = decay  # Rate at which pheromones evaporate
        self.alpha = alpha  # Strength of pheromone update
        self.best_solution_history = [] # Track best solution found in each iteration
        self.best_path_history = [] # Track best path in each iteration
       
   
    # Main function to run the ACO algorithm
    def run(self):
        best_path = None
        best_solution = np.inf 
        # Run the algorithm for the specified number of iterations
        for _ in range(self.num_iterations):
            ants = [Ant(self.graph, self.group_size, self.must_links) for _ in range(self.num_ants)]  # Create a group of ants
            for ant in ants:
                ant.group()  # Let each ant form groups
                ant.solution_score = self.evaluate_solution(ant)
                if best_solution > ant.solution_score:
                    best_solution = ant.solution_score
                    best_path = ant.path
                    print('best solution: ', best_solution)     
            self.update_pheromones(ants)  # Update pheromones based on the ants' paths
            # Visualisierung der Pheromone als Heatmap
            #self.plot_pheromone_heatmap(self.graph.pheromones, _)
            self.best_solution_history.append(best_solution)  # Save the best solution for each iteration
            self.best_path_history.append(best_path)
        return best_path, best_solution
  
    # Update the pheromones on the paths after all ants have completed their trips
    def update_pheromones(self, ants):
        self.graph.pheromones *= self.decay  # Reduce pheromones on all paths (evaporation)
        # For each ant, increase pheromones on the paths they took, based on how good their path was
        for ant in ants:
            for group in ant.path:
                combinations = [(int(x),int(y)) for x,y in itertools.combinations(group,2)]
                for x, y in combinations:
                    self.graph.pheromones[x, y] += self.alpha / ant.solution_score 
                    self.graph.pheromones[y, x] += self.alpha / ant.solution_score                   

    #TODO: weight distances 
    def evaluate_solution(self, ant):
        euclidean_dist = ant.euclidean_dist
        het_dist = ant.het_dist
        hot_hom_dist = ant.hot_hom_dist
        hot_het_dist = ant.hot_het_dist
        return euclidean_dist + het_dist + hot_hom_dist + hot_het_dist
                                    

       
    # Assign groups from best solution to DataFrame after algorithm runs 
    def add_groups(self, path, df):
        df['Group'] = None
        # Iterate through each group and update the 'Group' column in the Dat3aFrame
        for group_number, group in enumerate(path):
            for index in group:
                df.at[index, 'Group'] = group_number
        return df      
          