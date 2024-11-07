import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from scipy.stats import ranksums


def spread_factor(u=None, nc=2):
    """
    Computes the spread factor for Simulated Binary Crossover (SBX) given the u value

    Parameters:
    - u (float): random u value from 0 to 1
    - nc (int): n_c value, n=0 uniform distribution, 2<n<5 matches closely the simulation for single-point crossover

    Returns:
    - beta (float)
    """
    if u is None:
        u = random.random()

    if u <= 0.5:
        beta = (2 * u) ** (1/(nc + 1))
    else:
        beta = (1/(2 * (1 - u))) ** (1/(nc + 1))
    return beta

def beta_q_factor(delta, eta_m, u=None):
    """
    Computes the beta_q factor for parameter based mutation (PM)

    Parameters:
    - delta (float): value calculated with y parent solution, y upper and lower limits.
    - eta_m (float): 100 + generation number aka η_m
    
    Returns:
    - Returns:
    - beta_q (float)
    """    
    if u is None:
        u = np.random.rand(*delta.shape)
    
    delta = np.clip(delta, 0, 1)
    delta_q = np.where(
        u <= 0.5,
        (2 * u + (1 - 2 * u) * (1 - delta) ** (eta_m + 1)) ** (1 / (eta_m + 1)) - 1,
        1 - (2 * (1 - u) + 2 * (u - 0.5) * (1 - delta) ** (eta_m + 1)) ** (1 / (eta_m + 1))
    )

    return delta_q

def set_J(N, pc=0.8, binary=False):
    """
    Generate crossover points for binomial or binary crossover.
    
    Parameters:
    - N (int): Length of the chromosome.
    - pc (float): Crossover probability for binomial crossover.
    - binary (bool): If True, use binary (single-point) crossover. 
                     If False, use binomial crossover.
    
    Returns:
    - list: Indices selected for crossover.
    """
    if binary: # Single-point crossover
        j_star = np.random.randint(1, N)
        J = set(range(j_star, N))  # Include all points from j_star to the end
    else: # Binomial crossover
        j_star = np.random.randint(0, N)
        J = {j_star}
        for j in range(N):
            if j != j_star:
                u = np.random.random()
                if u < pc:
                    J.add(j)
    
    return list(J)

def dict_to_dataframe(stats_dict):
    """
    Convert a dictionary of statistics to a pandas DataFrame.

    Parameters:
    - stats_dict (dict): Dictionary where keys are generation numbers and values are dictionaries with fitness statistics.

    Returns:
    - pd.DataFrame: A DataFrame with 'Generation' as the index and columns for each statistic.
    """
    df = pd.DataFrame.from_dict(stats_dict, orient='index')
    
    df.index.name = 'Generation'
    
    return df

def compare_algorithms(results, p_value_threshold):
    """
    Perform the Wilcoxon rank-sum test to compare the performance of GA, DE,
    and an optional third algorithm for each test problem.
    
    Parameters:
    - results (dict): A dictionary containing the performance metrics for each test problem.
    
    Returns:
    - comparison_results (dict): A dictionary with the p-value and conclusion for each comparison in each test problem.
    """
    comparison_results = {}
    
    for problem, data in results.items():
        algorithms = list(data.keys())
        
        if len(algorithms) == 2:
            alg1, alg2 = algorithms
            performance1 = data[alg1]
            performance2 = data[alg2]
            
            stat, p_value = ranksums(performance1, performance2)
            conclusion = "No significant difference"
            if p_value < p_value_threshold:
                conclusion = f"{alg1} outperforms {alg2}" if stat < 0 else f"{alg2} outperforms {alg1}"
            
            comparison_results[problem] = {
                f"{alg1} vs {alg2}": {
                    "statistic": stat,
                    "p_value": p_value,
                    "conclusion": conclusion
                }
            }
        
        elif len(algorithms) == 3:
            alg1, alg2, alg3 = algorithms
            comparisons = [(alg1, alg2), (alg1, alg3), (alg2, alg3)]
            comparison_results[problem] = {}

            for algA, algB in comparisons:
                performanceA = data[algA]
                performanceB = data[algB]
                
                stat, p_value = ranksums(performanceA, performanceB)
                conclusion = "No significant difference"
                if p_value < p_value_threshold:
                    conclusion = f"{algA} outperforms {algB}" if stat < 0 else f"{algB} outperforms {algA}"
                
                comparison_results[problem][f"{algA} vs {algB}"] = {
                    "statistic": stat,
                    "p_value": p_value,
                    "conclusion": conclusion
                }
    
    return comparison_results

def stochastic_ranking(population, Pf=0.45):
    """
    Perform stochastic ranking on a population of Chromosome objects.
    
    Parameters:
    - population: List of Chromosome objects to be ranked. !! NOT CLASS POPULATION !!
    - Pf: Probability of comparing based on fitness.
    
    Returns:
    - sorted_population: List of Chromosome objects sorted by stochastic ranking.
    """
    fitness_values = np.array([ind.fitness for ind in population])
    constraint_violations = np.array([ind.constraint_violation for ind in population])
    
    use_fitness = np.random.rand(len(population)) < Pf
    
    combined_score = np.where(use_fitness, fitness_values, constraint_violations)
    
    sorted_indices = np.argsort(combined_score)
    
    sorted_population = [population[i] for i in sorted_indices]
    
    return sorted_population


def plot_columns(df):
    """
    Generate a line plot for each column in the dataframe.
    """
    num_columns = len(df.columns)
    fig, axes = plt.subplots(num_columns, 1, figsize=(10, 5 * num_columns), sharex=True)

    for i, column in enumerate(df.columns):
        ax = axes[i] if num_columns > 1 else axes
        ax.plot(df.index, df[column], marker='o', linestyle='-')
        ax.set_title(f"{column} Over Generations")
        ax.set_xlabel("Generation")
        ax.set_ylabel(column)
        ax.grid(True)

    plt.tight_layout()
    plt.show()