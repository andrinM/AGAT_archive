import numpy as np
import itertools
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ant import Ant
# ACO (Ant Colony Optimization) class runs the algorithm to find the best path
class ACO:
    def __init__(self, graph, num_ants, num_iterations, group_size, mode, decay=0.5, alpha=1.0):
        self.graph = graph
        self.num_ants = num_ants  # Number of ants in each iteration
        self.num_iterations = num_iterations  # Number of iteration
        self.group_size = group_size
        self.mode = mode
        self.decay = decay  # Rate at which pheromones evaporate
        self.alpha = alpha  # Strength of pheromone update
        self.best_distance_history = [] # Track best distance found in each iteration
        self.best_path_history = [] # Track best path in each iteration
        self.w_g = 0
        self.w_cv = 0 
        self.w_ed = 1
   
    
    def set_weights(self,w_g, w_cv, w_ed):
        self.w_g = w_g
        self.w_cv = w_cv
        self.w_ed = w_ed


    # Main function to run the ACO algorithm
    def run(self):
        best_path = None
        val = self.set_value()
        best_distance = val
        best_solution = val
        # Run the algorithm for the specified number of iterations
        for _ in range(self.num_iterations):
            ants = [Ant(self.graph, self.group_size, self.graph.df, self.mode) for _ in range(self.num_ants)]  # Create a group of ants
            for ant in ants:
                ant.group()  # Let each ant form groups
                ant.solution_score = self.evaluate_solution(ant)
                compare = (lambda a, b: a < b) if self.mode == 'het' else (lambda a, b: a > b)
                if compare(best_solution, ant.solution_score):
                    best_solution = ant.solution_score
                    best_path = ant.path
                    best_distance = ant.total_distance   
                     
            self.update_pheromones(ants)  # Update pheromones based on the ants' paths
            self.best_distance_history.append(best_distance)  # Save the best distance for each iteration
            self.best_path_history.append(best_path)
        return best_path, best_distance
  
    # Update the pheromones on the paths after all ants have completed their trips
    def update_pheromones(self, ants):
        self.graph.pheromones *= self.decay  # Reduce pheromones on all paths (evaporation)
        # For each ant, increase pheromones on the paths they took, based on how good their path was
        for ant in ants:
            for group in ant.path:
                pheromones = ant.solution_score
                combinations = [(int(x),int(y)) for x,y in itertools.combinations(group,2)]
                for x, y in combinations:
                    self.graph.pheromones[x, y] += (self.alpha * pheromones 
                    if self.mode == 'het' 
                    else self.alpha / pheromones )                    

    def evaluate_solution(self, ant):
        g =  self.calculate_total_goodness(ant)
        cv = self.coefficient_of_var(g)
        distance = ant.total_distance
        return self.w_g * g.sum() + self.w_cv * cv + self.w_ed * distance
                                    
    def calculate_score_deviation(self, mid, scores):
        result = 0
        for val in scores: 
            result += abs(mid - val)
        return result 
  
    def calcualte_group_goodness(self, group, ant):
        scores = ant.df.loc[group, 'Student_Score'].copy()
        mid = (scores.min() + scores.max()) /2 
        min_index = scores.idxmin()
        max_index = scores.idxmax()
        scores_without_min_max = scores.drop([min_index, max_index])
        summe = self.calculate_score_deviation(mid, scores_without_min_max)
        goodness = (scores.max() - scores.min())/(1+summe)
        return goodness

    def calculate_total_goodness(self,ant):
        goodness_solution = np.zeros(10)
        for i, group in enumerate(ant.path):
            goodness_solution[i] = self.calcualte_group_goodness(group, ant)
        return goodness_solution

    def coefficient_of_var(self, goodness_solution):
        if goodness_solution.sum() == 0: 
            return 0 
        else:
            return np.std(goodness_solution, ddof=1) / np.mean(goodness_solution)
       


    # Assign groups from best solution to DataFrame after algorithm runs 
    def add_groups(self, path, df):
        df['Group'] = None
        # Iterate through each group and update the 'Group' column in the Dat3aFrame
        for group_number, group in enumerate(path):
            for index in group:
                df.at[index, 'Group'] = group_number
        return df      

    # set parameters according to the mode in which the algorithm should run: homogeneous or heterogeneous
    def set_value(self):
        return np.inf if self.mode =='hom' else 0            