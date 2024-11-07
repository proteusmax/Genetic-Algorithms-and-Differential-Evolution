import random
import numpy as np
from utils import *

class Chromosome:
    """
    Class for representing a real-encoded solution in the population.
    """
    def __init__(self, objective_function, penalty = True, genes = None):
        self.__obj_func_singleton = objective_function
        self.genes = genes if genes is not None else self.initialize_genes()
        self.apply_bounds()
        self.penalty = penalty
        self.n = objective_function.get_nvar()
        self.calculate_fitness()

    def initialize_genes(self):
        xmin = self.__obj_func_singleton.get_xmin()
        xmax = self.__obj_func_singleton.get_xmax()
        return np.array([np.random.uniform(low, high) for low, high in zip(xmin, xmax)])
    
    def calculate_fitness(self):
        weighted_penalty, unweighted_penalty, _ = self.__obj_func_singleton.evaluate_penalty(self.genes)
        if self.penalty: # penalty is included in fitness
            self.fitness = self.__obj_func_singleton.evaluate(self.genes) + weighted_penalty
        else: # fitness does not include penalty (for stochastic ranking)
            self.fitness = self.__obj_func_singleton.evaluate(self.genes)
            self.constraint_violation = weighted_penalty

    def return_fitness(self):
        fitness = self.__obj_func_singleton.evaluate(self.genes)
        weighted_penalty, unweighted_penalty, num_violations = self.__obj_func_singleton.evaluate_penalty(self.genes)

        return fitness, weighted_penalty, unweighted_penalty, num_violations

    def apply_bounds(self):
        xmin = self.__obj_func_singleton.get_xmin()
        xmax = self.__obj_func_singleton.get_xmax()
        self.genes = np.clip(self.genes, xmin, xmax)

    def binomial_crossover(self, trial, binary = False, crossover_rate=0.8): # Binomial or Binary crossover
        J = functions.set_J(self.n, binary, crossover_rate)
        offspring = Chromosome(self.__obj_func_singleton, penalty = self.penalty, genes=self.genes.copy())
        for j in J:
            offspring.genes[j] = trial.genes[j]
        offspring.calculate_fitness()
        
        return offspring

    def sbx(self, other, u = None, nc = 2):
        b = functions.spread_factor(u, nc)
        parent1 = self.genes
        parent2 = other.genes

        child1 = 0.5 * ((parent1 + parent2) - b * (parent2 - parent1)) 
        child2 = 0.5 * ((parent1 + parent2) + b * (parent2 - parent1))

        return Chromosome(self.__obj_func_singleton, genes=child1), Chromosome(self.__obj_func_singleton, genes=child2) 

    def parameter_based_mutation(self, t=1):
        y = self.genes
        y_l = self.__obj_func_singleton.get_xmin()
        y_u = self.__obj_func_singleton.get_xmax()
        eta_m = 100 + t
        delta_max = y_u - y_l
        delta = np.minimum(y - y_l, y_u - y) / delta_max
        beta_q = functions.beta_q_factor(delta=delta, eta_m=eta_m)
        self.genes = y + beta_q * delta_max
        self.calculate_fitness()