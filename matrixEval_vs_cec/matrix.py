import numpy as np
import matplotlib.pyplot as plt
from opfunu.cec_based import cec2014, cec2017, cec2020, cec2022
import pandas as pd
from tqdm import tqdm
import matplotlib
matplotlib.use('TkAgg')

def reflective_boundaries(position, lb, ub):
    position = np.where(position < lb, 2 * lb - position, position)
    position = np.where(position > ub, 2 * ub - position, position)
    return np.clip(position, lb, ub)

def distance(x, y):
    return np.linalg.norm(x - y)

def aquila_optimizer(fitness_function, bounds, dim, max_evals, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    population_size = 50
    max_iterations = max_evals // population_size
    lb, ub = bounds
    alpha = 0.1
    delta = 0.1
    beta = 1.8

    positions = np.random.uniform(lb, ub, (population_size, dim))
    fitness = np.zeros(population_size)
    num_evals = 0

    for i in range(population_size):
        fitness[i] = fitness_function(positions[i])
        num_evals += 1

    best_idx = np.argmin(fitness)
    X_best = positions[best_idx].copy()
    best_fitness = fitness[best_idx]
    convergence_curve = [best_fitness]

    for iter in range(max_iterations):
        a = 2 * (1 - iter / max_iterations)  # Exploration-exploitation balance

        # Weighted mean based on fitness
        weights = 1 / (fitness + np.finfo(float).eps)
        weighted_mean = np.sum(positions * weights[:, np.newaxis], axis=0) / np.sum(weights)

        for i in range(population_size):
            if iter <= (2/3) * max_iterations:
                if np.random.rand() < 0.5:
                    # High soar with vertical stoop
                    positions[i] = X_best * (1 - iter / max_iterations) + (weighted_mean - X_best) * np.random.rand()
                else:
                    # Contour flight with Lévy glide
                    r = np.random.rand()
                    theta = np.random.rand() * 2 * np.pi
                    x = r * np.sin(theta)
                    y = r * np.cos(theta)

                    sigma = (np.math.gamma(1+beta)*np.sin(np.pi*beta/2)/(np.math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
                    u = np.random.randn(dim) * sigma
                    v = np.random.randn(dim)
                    levy = 0.01 * u / np.abs(v)**(1/beta)

                    positions[i] = X_best * levy + positions[np.random.randint(population_size)] + (y - x) * np.random.rand()
            else:
                if np.random.rand() < 0.5:
                    # Low flight with gradual descent
                    positions[i] = (X_best * weighted_mean) * alpha - np.random.rand() + ((ub - lb) * np.random.rand() + lb) * delta
                else:
                    # Walk and grab prey
                    QF = iter**((2*np.random.rand()-1)/(1 - max_iterations)**2)
                    G1 = 2 * np.random.rand() - 1
                    G2 = 2 * (1 - iter / max_iterations)
                    levy = np.random.rand(dim)

                    positions[i] = QF * X_best - G1 * positions[i] * np.random.rand() - G2 * levy + np.random.rand(dim)

            # Apply reflective boundaries
            positions[i] = reflective_boundaries(positions[i], lb, ub)

            # Evaluate and update best
            fitness[i] = fitness_function(positions[i])
            num_evals += 1

            if fitness[i] < best_fitness:
                best_fitness = fitness[i]
                X_best = positions[i].copy()

            if num_evals >= max_evals:
                break

        convergence_curve.append(best_fitness)

        if num_evals >= max_evals:
            break

    return X_best, best_fitness, convergence_curve, num_evals

import os

def run_experiments(dimension=10, max_evals=60000, num_runs=50):
    benchmark_functions = []
    for i in range(1, 14):
        func_class = getattr(cec2014, f"F{i}2014")
        benchmark_functions.append(func_class(ndim=dimension))
    for i in range(1, 10):
        func_class = getattr(cec2017, f"F{i}2017")
        benchmark_functions.append(func_class(ndim=dimension))
    for i in range(1, 5):
        func_class = getattr(cec2020, f"F{i}2020")
        benchmark_functions.append(func_class(ndim=dimension))
    for i in range(1, 5):
        func_class = getattr(cec2022, f"F{i}2022")
        benchmark_functions.append(func_class(ndim=dimension))

    results_df_full = pd.DataFrame()
    convergence_data = {}
    stats_path = 'ao_statistics.csv'
    results_path = 'ao_results.csv'

    # Clear old results if they exist
    if os.path.exists(stats_path):
        os.remove(stats_path)
    if os.path.exists(results_path):
        os.remove(results_path)

    for func_idx, func in enumerate(benchmark_functions):
        func_name = func.__class__.__name__
        print(f"\nRunning on function {func_idx+1}/{len(benchmark_functions)}: {func_name}")
        convergence_data[func_name] = []
        results = []

        for run in tqdm(range(num_runs)):
            seed = run + 1000
            _, best_fitness, convergence, _ = aquila_optimizer(
                func.evaluate,
                [func.lb, func.ub],
                func.ndim,
                max_evals,
                seed=seed
            )
            results.append(best_fitness)
            if run < 5:
                convergence_data[func_name].append(convergence)

        # Store results for this function
        func_df = pd.DataFrame({func_name: results})
        results_df_full = pd.concat([results_df_full, func_df], axis=1)

        # Append to CSV immediately
        if not os.path.exists(results_path):
            func_df.to_csv(results_path, index=False)
        else:
            existing = pd.read_csv(results_path)
            combined = pd.concat([existing, func_df], axis=1)
            combined.to_csv(results_path, index=False)

        # Append stats to stats CSV
        stats_df = pd.DataFrame({
            'Mean': [np.mean(results)],
            'Std': [np.std(results)],
            'Min': [np.min(results)],
            'Median': [np.median(results)],
            'Max': [np.max(results)]
        }, index=[func_name])

        if not os.path.exists(stats_path):
            stats_df.to_csv(stats_path)
        else:
            stats_df.to_csv(stats_path, mode='a', header=False)

    return results_df_full, convergence_data


def generate_statistics(results_df):
    stats = {
        'Mean': results_df.mean(),
        'Std': results_df.std(),
        'Min': results_df.min(),
        'Median': results_df.median(),
        'Max': results_df.max()
    }
    stats_df = pd.DataFrame(stats).T
    return stats_df

def plot_convergence(convergence_data, functions_to_plot=None):
    if functions_to_plot is None:
        functions_to_plot = ['F12014', 'F12017', 'F12020', 'F12022']
    plt.figure(figsize=(12, 8))
    for func_name in functions_to_plot:
        avg_convergence = np.mean(convergence_data[func_name], axis=0)
        iterations = np.arange(len(avg_convergence))
        if '2014' in func_name:
            plt.plot(iterations, avg_convergence, 'c-', label=f'CEC2014 - {func_name}')
        elif '2017' in func_name:
            plt.plot(iterations, avg_convergence, 'orange', label=f'CEC2017 - {func_name}')
        elif '2020' in func_name:
            plt.plot(iterations, avg_convergence, 'r-', label=f'CEC2020 - {func_name}')
        elif '2022' in func_name:
            plt.plot(iterations, avg_convergence, 'g-', label=f'CEC2022 - {func_name}')
    plt.xlabel('Iterations')
    plt.ylabel('Fitness Value')
    plt.title('Convergence Curve of Aquila Optimizer')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    dimension = 10
    max_evals = 60000
    num_runs = 50
    print(f"Running Aquila Optimizer on 30 benchmark functions")
    print(f"Dimension: {dimension}, Max Evaluations: {max_evals}, Runs per function: {num_runs}")
    results_df, convergence_data = run_experiments(dimension, max_evals, num_runs)
    stats_df = generate_statistics(results_df)
    print("\nStatistics:")
    print(stats_df)
    results_df.to_csv('ao_results.csv')
    stats_df.to_csv('ao_statistics.csv')
    plot_convergence(convergence_data)
