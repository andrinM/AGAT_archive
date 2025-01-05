import numpy as np
import itertools
import importlib

importlib.reload(itertools)

# Ant class represents an individual ant that travels across the graph
class Ant:
    def __init__(self, graph, group_size, df, mode):
        self.df = self.calculate_student_score(df) 
        self.graph = graph
        self.group_size = group_size
        self.mode = mode
        self.group_num = graph.num_nodes // group_size
        self.current_node = None
        # Initialize a path to store nodes grouped into "group_num" groups, each with "group_size" members 
        self.path = [[None] * group_size for i in range(self.group_num)]
        self.total_distance = 0   #Start with zero distance traveled
        self.group_distance = np.zeros(self.group_num)
        self.unvisited_nodes = set(range(graph.num_nodes)) # keep track of unvisited nodes 
        self.solution_score = self.set_init_score()

    def set_init_score(self): 
        return np.inf if self.mode == 'hom' else 0 
    # function to form groups of nodes (members) 
    def group(self): 
        for group in range(self.group_num):
            # Randomly choose first group member  
            self.current_node = int((np.random.choice(list(self.unvisited_nodes))))
            self.path[group][0] = self.current_node
            self.unvisited_nodes.remove(self.current_node)
            for node in range(1,self.group_size):
                next_node = self.select_next_node(self.path[group])
                self.path[group][node] = next_node
                self.current_node = next_node
            # calculate group_distances for each group 
        self.calculate_group_distance()               
        # Sum up all the group distances to get the total distance, at the end this should be minimalized through iterations 
        self.total_distance = self.group_distance.sum()
        return self.path

    # Select next node based on pheromone levels and distance
    def select_next_node(self, current_group):
        # If only one node's left, choose it 
        if len(self.unvisited_nodes) == 1: 
            return self.unvisited_nodes.pop()
        else: 
            # Array to store selection probabilities 
            probabilities = np.zeros(self.graph.num_nodes)
            for node in self.unvisited_nodes:
                loc_information = self.get_loc_information(current_group, node)
                probabilities[node] = (self.graph.pheromones[self.current_node][node]/loc_information 
                       if self.mode == 'hom' 
                       else loc_information * self.graph.pheromones[self.current_node][node])
            if probabilities.sum() != 0:                                  
                probabilities /= probabilities.sum()  # Normalize the probabilities to sum to 1
            else:
                # Fallback auf gleichmäßige Verteilung
                probabilities[list(self.unvisited_nodes)] = 1 / len(self.unvisited_nodes)
            # Choose the next node based on the calculated probabilities
            next_node = int(np.random.choice(range(self.graph.num_nodes), p=probabilities))
            self.unvisited_nodes.remove(next_node)
            return next_node 


    # Benefit of adding a student to a group: 
    # 1. regarding euclidean distance between all group members that are already added to group 
    # 2. regarding the student score: one high score, one low score and one average score if grouping is 3 people         
    def get_loc_information(self,current_group,node): 
        loc_information = 0.001
        for member in current_group:
            if member is not None:
                loc_information += self.graph.euclidean_dist[member][node] + abs(self.df.loc[node, 'Student_Score'] - self.df.loc[member, 'Student_Score']) 
        return loc_information                        
                                 

    def calculate_group_distance(self):
        for i, group in enumerate(self.path): 
            combinations = [(int(x),int(y)) for x,y in itertools.combinations(group,2)]
            for x,y in combinations:
                self.group_distance[i] += self.graph.euclidean_dist[x][y]
           

    def calculate_student_score(self,df):
        if not 'Student_Score' in df:
            df['Student_Score'] = df.iloc[:, 1:].sum(axis=1)
        return df
