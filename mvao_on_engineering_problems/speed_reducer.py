import numpy as np
import matplotlib.pyplot as plt
from time import time
import matplotlib
matplotlib.use('TkAgg')
# MVAO implementation with reflective boundaries
def reflective_boundaries(position, lb, ub):
    """Apply reflective boundary handling."""
    position = np.where(position < lb, 2 * lb - position, position)
    position = np.where(position > ub, 2 * ub - position, position)
    return np.clip(position, lb, ub)

def speed_reducer_objective(x):
    """Calculate the weight of the speed reducer (objective function)."""
    x1, x2, x3, x4, x5, x6, x7 = x
    
    # Handle integer constraint for x3 (number of teeth)
    x3 = round(x3)
    
    # Calculate weight
    weight = 0.7854 * x1 * x2**2 * (3.3333 * x3**2 + 14.9334 * x3 - 43.0934)
    weight += 7.4777 * (x6**3 + x7**3)
    weight += 0.7854 * (x4 * x6**2 + x5 * x7**2)
    
    return weight

def speed_reducer_constraints(x):
    """Calculate constraint violations for the speed reducer problem."""
    x1, x2, x3, x4, x5, x6, x7 = x
    x3 = round(x3)  # Ensure x3 is integer
    
    # Constraints
    g = np.zeros(11)
    
    g[0] = 27 / (x1 * x2**2 * x3) - 1
    g[1] = 397.5 / (x1 * x2**2 * x3**2) - 1
    g[2] = 1.93 * x4**3 / (x2 * x3 * x6**4) - 1
    g[3] = 1.93 * x5**3 / (x2 * x3 * x7**4) - 1
    g[4] = np.sqrt((745 * x4 / (x2 * x3))**2 + 16.9e6) / (110 * x6**3) - 1
    g[5] = np.sqrt((745 * x5 / (x2 * x3))**2 + 157.5e6) / (85 * x7**3) - 1
    g[6] = x2 * x3 / 40 - 1
    g[7] = 5 * x2 / x1 - 1
    g[8] = x1 / (12 * x2) - 1
    g[9] = (1.5 * x6 + 1.9) / x4 - 1
    g[10] = (1.1 * x7 + 1.9) / x5 - 1
    
    return g

def fitness_function(x):
    """Combined objective and constraint handling using penalty method."""
    # Get objective value
    obj = speed_reducer_objective(x)
    
    # Check constraints
    constraints = speed_reducer_constraints(x)
    violation = sum(max(0, g) for g in constraints)
    
    # Apply death penalty for constraint violations
    if violation > 0:
        return obj + 1e6 * violation
    
    return obj

def mvao_speed_reducer(max_iter=500, population_size=50, verbose=True,random_state=None):
    """
    Modified Aquila Optimizer (MVAO) for the Speed Reducer problem
    """
    if random_state is not None:
        np.random.seed(random_state)#ahh..... reproducibility
    # Problem dimensions and bounds
    dim = 7
    lb = np.array([2.6, 0.7, 17, 7.3, 7.8, 2.9, 5.0])
    ub = np.array([3.6, 0.8, 28, 8.3, 8.3, 3.9, 5.5])
    
    # MVAO Parameters
    alpha = 0.1
    delta = 0.1
    beta = 1.8  # Modified beta for Lévy flights
    
    # Initialize population
    positions = np.zeros((population_size, dim))
    for i in range(dim):
        positions[:, i] = np.random.uniform(lb[i], ub[i], population_size)
    
    # Round x3 to integers (number of teeth constraint)
    positions[:, 2] = np.round(positions[:, 2])
    
    # Evaluate initial population
    fitness = np.array([fitness_function(p) for p in positions])
    
    # Find initial best solution
    best_idx = np.argmin(fitness)
    X_best = positions[best_idx].copy()
    best_fitness = fitness[best_idx]
    
    # Initialize convergence curve
    convergence_curve = np.zeros(max_iter)
    convergence_curve[0] = best_fitness
    
    # Start optimization loop
    start_time = time()
    for iter in range(max_iter):
        # Calculate weighted mean based on fitness
        weights = 1 / (fitness + np.finfo(float).eps)
        weighted_mean = np.sum(positions * weights[:, np.newaxis], axis=0) / np.sum(weights)
        
        for i in range(population_size):
            if iter <= (2/3) * max_iter:  # Exploration phase
                if np.random.rand() < 0.5:
                    # High soar with vertical stoop (expanded exploration)
                    positions[i] = X_best * (1 - iter / max_iter) + (weighted_mean - X_best) * np.random.rand()
                else:
                    # Contour flight with Lévy glide (narrowed exploration)
                    r = np.random.rand()
                    theta = np.random.rand() * 2 * np.pi
                    x = r * np.sin(theta)
                    y = r * np.cos(theta)
                    
                    # Lévy flight calculation with modified beta
                    sigma = (np.math.gamma(1+beta)*np.sin(np.pi*beta/2)/(np.math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
                    u = np.random.randn(dim) * sigma
                    v = np.random.randn(dim)
                    levy = 0.01 * u / np.abs(v)**(1/beta)
                    
                    positions[i] = X_best * levy + positions[np.random.randint(population_size)] + (y - x) * np.random.rand()
            else:  # Exploitation phase
                if np.random.rand() < 0.5:
                    # Low flight with gradual descent (expanded exploitation)
                    positions[i] = (X_best * weighted_mean) * alpha - np.random.rand() + ((ub - lb) * np.random.rand() + lb) * delta
                else:
                    # Walk and grab prey (narrowed exploitation)
                    QF = iter**((2*np.random.rand()-1)/(1 - max_iter)**2)
                    G1 = 2 * np.random.rand() - 1
                    G2 = 2 * (1 - iter / max_iter)
                    levy = np.random.rand(dim)  # Simplified Lévy for vectorization
                    
                    positions[i] = QF * X_best - G1 * positions[i] * np.random.rand() - G2 * levy + np.random.rand(dim)
            
            # Apply reflective boundaries instead of simple clipping
            positions[i] = reflective_boundaries(positions[i], lb, ub)
            
            # Ensure x3 is integer (number of teeth constraint)
            positions[i, 2] = round(positions[i, 2])
            
            # Evaluate fitness
            fitness[i] = fitness_function(positions[i])
            
            # Update best solution
            if fitness[i] < best_fitness:
                best_fitness = fitness[i]
                X_best = positions[i].copy()
        
        # Record best fitness
        convergence_curve[iter] = best_fitness
        
        # Display progress
        if verbose and (iter % 10 == 0 or iter == max_iter - 1):
            print(f"Iteration {iter+1}/{max_iter}, Best fitness: {best_fitness:.6f}")
    
    # Ensure x3 is integer in final solution
    X_best[2] = round(X_best[2])
    
    # Calculate final objective and constraint violations
    final_obj = speed_reducer_objective(X_best)
    final_constraints = speed_reducer_constraints(X_best)
    
    # Check if all constraints are satisfied
    constraints_satisfied = all(g <= 0 for g in final_constraints)
    
    # Print results
    if verbose:
        print("\n" + "="*50)
        print("MVAO Optimization Complete")
        print("="*50)
        print(f"Execution Time: {time() - start_time:.2f} seconds")
        print(f"Best Solution: {X_best}")
        print(f"Objective Value (Weight): {final_obj:.6f}")
        print(f"Constraints Satisfied: {constraints_satisfied}")
        print("Constraint Values:")
        for i, g in enumerate(final_constraints):
            print(f"  g{i+1}: {g:.6f} {'✓' if g <= 0 else '✗'}")
    
    # Plot convergence curve
    plt.figure(figsize=(10, 6))
    plt.semilogy(range(1, max_iter+1), convergence_curve)
    plt.xlabel('Iterations')
    plt.ylabel('Fitness (log scale)')
    plt.title('MVAO Convergence Curve for Speed Reducer Problem')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    return X_best, final_obj, convergence_curve

if __name__ == "__main__":
    # Run the optimizer
    best_solution, best_fitness, convergence = mvao_speed_reducer(max_iter=500, population_size=50,random_state=42)
    
    # Display final optimal design variables
    print("\nOptimal Design Variables:")
    print(f"x1 (face width): {best_solution[0]:.6f}")
    print(f"x2 (module of teeth): {best_solution[1]:.6f}")
    print(f"x3 (number of teeth): {int(best_solution[2])}")
    print(f"x4 (length of shaft 1): {best_solution[3]:.6f}")
    print(f"x5 (length of shaft 2): {best_solution[4]:.6f}")
    print(f"x6 (diameter of shaft 1): {best_solution[5]:.6f}")
    print(f"x7 (diameter of shaft 2): {best_solution[6]:.6f}")
    print(f"\nMinimum Weight: {best_fitness:.6f}")
