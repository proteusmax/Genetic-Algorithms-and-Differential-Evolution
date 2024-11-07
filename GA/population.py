import random
import numpy as np
import pandas as pd
from utils import *
from Problems import *
from GA import *
import configparser


class Population:
    """
    Class representing a population of Chromosomes in a genetic algorithm.

    Parameters:
    - n_var (int): number of variables in the problem
    """
    def __init__(self, objective_function, penalty = True, n_var=None, t= None, config_file=None, crossover_rate=0.8, mutation_factor=0.6, num_difference_vectors = 1, chromosomes = None):
        self.generation_t = t if t is not None else 0
        self.penalty = penalty
        self.generation_statistics = {}
        
        if isinstance(objective_function, str):
            self.objective_function = problems.FunctionFactory.select_function(objective_function, n_var)
        else:
            raise ValueError("objective_function should be a string whose name is defined in Problems/problems.py")
        
        self.load_config(config_file)
        self.update_dynamic_factors()
    
        self.chromosomes = chromosomes if chromosomes is not None else np.array([Chromosome(self.objective_function, penalty = self.penalty) for _ in range(self.pop_size)])
        
        self.update_best()

    def load_config(self, config_file):
        config = configparser.ConfigParser()
        config.read(config_file)
        # Population Settings
        self.pop_size = int(config['PopulationSettings']['pop_size'])
        self.max_generations = int(config['PopulationSettings']['max_generations'])
        self.crossover_rate = float(config['PopulationSettings']['crossover_rate']) 
        self.mutation_factor = float(config['PopulationSettings']['mutation_factor']) 
        self.num_difference_vectors = int(config['PopulationSettings']['num_difference_vectors'])
        # Dynamic Penalty
        self.max_penalty_exp = float(config['PenaltySettings']['max_penalty_exp'])
        self.min_penalty_exp = float(config['PenaltySettings']['min_penalty_exp'])
        self.max_penalty_factor = float(config['PenaltySettings']['max_penalty_factor'])
        self.min_penalty_factor = float(config['PenaltySettings']['min_penalty_factor'])
        self.max_tolerance_factor = float(config['PenaltySettings']['max_tolerance_factor'])
        self.min_tolerance_factor = float(config['PenaltySettings']['min_tolerance_factor'])

    def update_dynamic_factors(self):
        progress = self.generation_t / self.max_generations
        penalty_factor = self.min_penalty_factor + progress * (self.max_penalty_factor - self.min_penalty_factor)
        tolerance_factor = self.max_tolerance_factor - progress * (self.max_tolerance_factor - self.min_tolerance_factor)
        penalty_exp = self.min_penalty_exp + progress * (self.max_penalty_exp - self.min_penalty_exp)
        
        self.objective_function.set_penalty_factors(penalty_factor, tolerance_factor, penalty_exp)

    def update_best(self):
        self.best_chromosome = min(self.chromosomes, key=lambda chromo: chromo.fitness)

    def pass_next_generation(self):
        """
        Calculates fitness, penalties and number of violations from best_chromosome, saves it into a dictionary and updates
        generation count that is used for eta_m in pbm and as a stop criteria for evolve method
        """
        self.update_best()
        best_fitness, weighted_penalty, unweighted_penalty, num_violations =  self.best_chromosome.return_fitness()
        
        self.generation_t += 1
        self.update_dynamic_factors()

        self.generation_statistics[self.generation_t] = {
            "best_fitness": best_fitness, # Does not consider penalty
            "best_fitness_with_penalty": self.best_chromosome.fitness,
            "weighted_penalty": weighted_penalty,
            "unweighted_penalty": unweighted_penalty,
            "num_violations": num_violations
        }

    def get_population_statistics(self):
        """
        Returns dictionary with information about best individual across generations
        """
        return self.generation_statistics

    def random_selection(self, with_replacement=False):
        """
        Randomly selects individuals from the population.
        """
        selection_size = self.pop_size

        selected_indices = np.random.choice(
            self.pop_size,
            size=self.pop_size,
            replace=with_replacement
        )

        selected_chromosomes = [self.chromosomes[idx] for idx in selected_indices]
        self.chromosomes = np.array(selected_chromosomes)

    def tournament_selection(self, q=2, p=0):
        """
        Parameters:
        - q (int): number of individuals that participate in a match.
        - p (float): probability of flipping the decision (for probabilistic approach 0.5 < p <= 1, for deterministic p = 0)
        """
        winners = []

        while len(winners) < self.pop_size:
            shuffled_indices = np.random.permutation(self.pop_size)
            reshaped_indices = shuffled_indices[:len(shuffled_indices) - len(shuffled_indices) % q].reshape(-1, q)

            group_fitness = np.array([[self.chromosomes[idx].fitness for idx in group] for group in reshaped_indices])
            best_in_group = np.argmin(group_fitness, axis=1)

            alternative_indices = np.random.randint(0, q, size=best_in_group.shape)
            flip_decision = np.random.rand(best_in_group.size) < p
            selected_indices = np.where(flip_decision, alternative_indices, best_in_group)

            selected_chromosomes = [self.chromosomes[reshaped_indices[i, selected_indices[i]]] for i in range(len(selected_indices))]

            winners.extend(selected_chromosomes)

        self.chromosomes = np.array(winners[:self.pop_size])

    def parent_vs_child_selection(self, offspring_population):
        parent_fitness = np.array([parent.fitness for parent in self.chromosomes])
        child_fitness = np.array([child.fitness for child in offspring_population])
        mask = parent_fitness <= child_fitness
        best_population = np.where(mask, self.chromosomes, offspring_population)
        self.chromosomes = np.array(best_population)

    def differential_evolution(self, num_difference_vectors):
        """
        Applies differential evolution to the population which generates a trial population
        """
        all_genes = np.array([chromo.genes for chromo in self.chromosomes])
        trial_genes = all_genes.copy() 

        for _ in range(num_difference_vectors):
            idxs = np.arange(self.pop_size)
            i2 = np.random.choice(idxs, self.pop_size, replace=True)
            i3 = np.random.choice(idxs, self.pop_size, replace=True)

            valid = (i2 != idxs) & (i3 != idxs) & (i2 != i3)
            while not np.all(valid):
                i2 = np.where(valid, i2, np.random.choice(idxs, self.pop_size, replace=True))
                i3 = np.where(valid, i3, np.random.choice(idxs, self.pop_size, replace=True))
                valid = (i2 != idxs) & (i3 != idxs) & (i2 != i3)

            x2_genes = all_genes[i2]
            x3_genes = all_genes[i3]

            trial_genes += self.mutation_factor * (x2_genes - x3_genes)

        X_lower = np.array(self.objective_function.get_xmin())
        X_upper = np.array(self.objective_function.get_xmax())
        trial_genes = np.clip(trial_genes, X_lower, X_upper)

        trial_chromosomes = [Chromosome(self.objective_function, genes=trial_genes[i]) for i in range(self.pop_size)]

        return np.array(trial_chromosomes)

    def binomial_crossover(self, binary=False):
        """
        Apply binomial or binary crossover to the population.
        """
        trial_population = self.differential_evolution(self.num_difference_vectors)
        offspring_population = []

        for chromosome, trial in zip(self.chromosomes, trial_population):
            offspring = chromosome.binomial_crossover(trial, binary, self.crossover_rate)
            offspring_population.append(offspring)
    
        return np.array(offspring_population)

    def stochastic_ranking(self, binary=False, Pf=0.45):
        if self.penalty:
            raise ValueError("Cannot perform Stochastic Ranking if penalty is included in fitness")
        
        offspring_population = self.binomial_crossover(binary)

        combined_population = np.concatenate((self.chromosomes, offspring_population))

        ranked_population = functions.stochastic_ranking(combined_population, Pf)

        self.chromosomes = np.array(ranked_population[:self.pop_size])

    def sbx_and_pbm(self):
        """
        Apply SBX (Simulated Binary Crossover) and PBM (Parameter-Based Mutation) across the population.
        """
        offspring_population = []
        np.random.shuffle(self.chromosomes)
        for i in range(0, self.pop_size, 2):
            parent1 = self.chromosomes[i]
            parent2 = self.chromosomes[(i + 1) % self.pop_size]

            if np.random.rand() < self.crossover_rate:
                child1, child2 = parent1.sbx(parent2)
            else:
                child1, child2 = Chromosome(self.objective_function, genes=parent1.genes.copy()), Chromosome(self.objective_function, genes=parent2.genes.copy())

            child1.parameter_based_mutation(self.generation_t)
            child2.parameter_based_mutation(self.generation_t)

            child1.apply_bounds()
            child2.apply_bounds()
            offspring_population.extend([child1, child2])

        self.chromosomes = np.array(offspring_population[:self.pop_size])

    def roulette_wheel_selection(self, replace=True):
        """
        Selects individuals from the population using roulette wheel selection.
        """
        fitness_values = np.array([chromo.fitness for chromo in self.chromosomes])
        
        if np.min(fitness_values) < 0:
            fitness_values = fitness_values + np.abs(np.min(fitness_values))
        
        inverted_fitness_values = np.max(fitness_values) - fitness_values
        total_fitness = np.sum(inverted_fitness_values)

        if total_fitness == 0:
            probabilities = np.ones(self.pop_size) / self.pop_size
        else:
            probabilities = inverted_fitness_values / total_fitness

        selected_indices = np.random.choice(
            self.pop_size, size=self.pop_size, replace=replace, p=probabilities
        )
        selected_chromosomes = [self.chromosomes[idx] for idx in selected_indices]
        self.chromosomes = np.array(selected_chromosomes)

    def evolve(self, algorithm_selection):
        """
        Evolve the population with either GA (random selection, sbx_and_pbm) or DE (dif. evolution, binomial crossover and selection)
        """
        while self.generation_t < self.max_generations:
            if algorithm_selection == "GA":
                self.roulette_wheel_selection()
                self.sbx_and_pbm()
            elif algorithm_selection == "DE+SR":
                self.stochastic_ranking(binary=True, Pf=0.35)
            elif algorithm_selection == "DE":
                offspring_population = self.binomial_crossover(binary=True)
                self.parent_vs_child_selection(offspring_population)

            self.pass_next_generation()
                
