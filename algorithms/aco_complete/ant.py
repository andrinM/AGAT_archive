import numpy as np
import itertools
import importlib

importlib.reload(itertools)

# Ant class represents an individual ant that travels across the graph
class Ant:
    def __init__(self, graph, group_size, must_links = None):
        self.graph = graph
        self.group_size = group_size
        self.group_num = graph.num_nodes // group_size
        self.must_links = must_links
        self.current_node = None
        # Initialize a path to store nodes grouped into "group_num" groups, each with "group_size" members 
        self.path = self.set_path()
        self.distance = 0   #Start with zero distance traveled
        self.group_distance = np.zeros(self.group_num)
        self.unvisited_nodes = self.set_unvisited_nodes()  # keep track of unvisited nodes 
        self.solution_score = np.inf

    def set_path(self):
        path = [[None] * self.group_size for i in range(self.group_num)]
        if self.must_links is not None: 
            for i, member in enumerate(self.must_links):
                path[i][:len(member)] = member
        return path 

    def set_unvisited_nodes(self):
        unvisited_nodes = set(range(self.graph.num_nodes))
        if self.must_links is not None: 
            linked_nodes = {node for link in self.must_links for node in link}
            unvisited_nodes -= linked_nodes
        return unvisited_nodes              

    # function to form groups of nodes (members) 
    def group(self): 
        for group in range(self.group_num):
            if all(value is not None for value in self.path[group]): 
                continue
            if self.path[group][0] is None:
                # Randomly choose first group member  
                self.current_node = int((np.random.choice(list(self.unvisited_nodes))))
                self.path[group][0] = self.current_node
                self.unvisited_nodes.remove(self.current_node)
            else: 
                self.current_node = self.path[group][self.path[group].index(None) - 1]     
            while None in self.path[group]:
                next_node = self.select_next_node(self.path[group])
                index = self.path[group].index(None)
                self.path[group][index] = next_node
                self.current_node = next_node
        self.calculate_group_distance()        
        self.set_distances()             
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
                if loc_information == 0: 
                    loc_information = 0.00001
                probabilities[node] = self.graph.pheromones[self.current_node][node] / loc_information
                assert probabilities[node] != np.inf     
            #if probabilities.sum() != 0:
            assert probabilities.sum() != 0                                   
            probabilities /= probabilities.sum()  # Normalize the probabilities to sum to 1
            #else:
                # Fallback auf gleichmäßige Verteilung
                #probabilities[list(self.unvisited_nodes)] = 1 / len(self.unvisited_nodes)
            # Choose the next node based on the calculated probabilities

            next_node = int(np.random.choice(range(self.graph.num_nodes), p=probabilities))
            self.unvisited_nodes.remove(next_node)
            return next_node 


    # Benefit of adding a student to a group: 
    # 1. regarding euclidean distance between all group members that are already added to group 
    # 2. regarding the student score: one high score, one low score and one average score if grouping is 3 people         
    def get_loc_information(self,current_group,node): 
        loc_information = 0
        for member in current_group:
            if member is not None:
                loc_information += self.graph.distance_matrix[member][node]                                    
        return loc_information                        

    def calculate_group_distance(self):
        self.calculate_distance()

    def set_distances(self):
        # Sum up all the group distances to get the total distance, at the end this should be minimalized through iterations 
        self.distance = self.group_distance.sum()
  


    def calculate_distance(self): 
        for i, group in enumerate(self.path): 
            combinations = [(int(x),int(y)) for x,y in itertools.combinations(group,2)]
            for x,y in combinations:
                self.group_distance[i] += self.graph.distance_matrix[x][y]

            
