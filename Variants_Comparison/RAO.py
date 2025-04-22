#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import math
from matplotlib.backend_bases import MouseButton
from scipy.stats import levy
matplotlib.use('TkAgg')

def distance(position, xprey, yprey):
    return np.sqrt((position[0] - xprey)**2 + (position[1] - yprey)**2)

def tent_map(n, x0=0.7):
    """Tent chaos map for initialization and chaotic local search."""
    x = np.zeros(n)
    x[0] = x0
    for i in range(n - 1):
        if x[i] < 0.5:
            x[i+1] = 2 * x[i]
        else:
            x[i+1] = 2 * (1 - x[i])
        x[i+1] = (1 - 1e-6) * x[i+1] + 1e-6 * np.random.rand()
    return x

def opposition_based_learning(population, fitness, lb, ub):
    """Opposition-Based Learning to enhance population diversity."""
    new_population = np.zeros_like(population)
    for i in range(len(population)):
        opposition = lb + ub - population[i]
        opposition = np.clip(opposition, lb, ub)
        opposition_fitness = distance(opposition, prey_pos[0], prey_pos[1])
        if opposition_fitness < fitness[i]:
            new_population[i] = opposition
        else:
            new_population[i] = population[i]
    return new_population

plt.figure(figsize=(8, 8))
plt.grid(True)
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.title('Click to place prey (first) and Predators (next clicks)\nRight click when done')

positions = []
prey_pos = None

def onclick(event):
    global prey_pos
    if event.button is MouseButton.RIGHT:
        plt.close()
        return

    if event.button is MouseButton.LEFT:
        if prey_pos is None:
            prey_pos = np.array([event.xdata, event.ydata])
            plt.plot(prey_pos[0], prey_pos[1], 'ks', markersize=12, markerfacecolor='magenta')
        else:
            positions.append([event.xdata, event.ydata])
            plt.plot(event.xdata, event.ydata, 'ro', markersize=8, markerfacecolor='blue')
        plt.draw()

plt.connect('button_press_event', onclick)
plt.show()

if len(positions) == 0 or prey_pos is None:
    print("No positions selected. Exiting.")
    exit()

positions = np.array(positions)
num_predators = len(positions)
max_iter = 500
dim = 2
lb = 0
ub = 100

# MPA Parameters
FADs = 0.2
P = 0.5
restart_interval = 100  # Restart every 100 iterations

# Initialize population using Tent Chaotic Mapping
initial_positions = np.zeros((num_predators, dim))
for d in range(dim):
    chaos_sequence = tent_map(num_predators, np.random.rand())
    initial_positions[:, d] = lb + (ub - lb) * chaos_sequence
positions = initial_positions.copy()

# Initialize best solution
X_best = positions[0].copy()
best_fitness = float('inf')
convergence_curve = np.zeros(max_iter)

# Evaluate initial population
fitness = np.array([distance(pos, prey_pos[0], prey_pos[1]) for pos in positions])
best_index = np.argmin(fitness)
X_best = positions[best_index].copy()
best_fitness = fitness[best_index]

plt.figure(figsize=(8, 8))
for iter in range(max_iter):
    plt.clf()
    plt.grid(True)
    plt.xlim(-10, 110)
    plt.ylim(-10, 110)

    plt.plot(prey_pos[0], prey_pos[1], 'ks', markersize=12, markerfacecolor='magenta')
    plt.plot(positions[:, 0], positions[:, 1], 'ro', markersize=8, markerfacecolor='blue')
    plt.plot(X_best[0], X_best[1], 'g*', markersize=15, markerfacecolor='lime')

    # Restart Strategy
    if iter % restart_interval == 0 and iter > 0:
        print(f"Restarting search at iteration {iter}")
        for i in range(num_predators):
            positions[i] = lb + np.random.rand(dim) * (ub - lb)
        fitness = np.array([distance(pos, prey_pos[0], prey_pos[1]) for pos in positions])
        best_index = np.argmin(fitness)
        X_best = positions[best_index].copy()
        best_fitness = fitness[best_index]

    # Sort predators based on fitness
    sorted_indices = np.argsort(fitness)
    sorted_positions = positions[sorted_indices]

    CF = (1 - iter / max_iter)**(2 * iter / max_iter)
    t = iter + 1
    T = max_iter

    for i in range(num_predators):
        if iter < max_iter / 3:  # Phase 1: High exploration
            step_size = np.random.rand(dim) * (sorted_positions[i] - prey_pos) * CF
            positions[i] = positions[i] + step_size
        elif iter < 2 * max_iter / 3:  # Phase 2: Transition phase
            if np.random.rand() < P:
                step_size = np.random.randn(dim) * (sorted_positions[i] - prey_pos)
                positions[i] = positions[i] + step_size
            else:
                step_size = levy.rvs(1.5, size=dim) * (sorted_positions[i] - prey_pos) * CF
                positions[i] = positions[i] + step_size
        else:  # Phase 3: High exploitation
            phi = 2 * np.random.rand(dim) - 1
            decay_factor = (1 - t / T)**(2 * t / T)

            if np.random.rand() < 0.5:  # Expanded Exploitation
                positions[i] = X_best + decay_factor * (X_best - positions[i]) * phi
            else:  # Narrowed Exploitation (Lévy flight)
                LF = levy.rvs(1.5, size=dim)
                positions[i] = X_best + LF * (X_best - positions[i]) * np.random.rand(dim)

            # Chaotic Local Search (applied with a probability in exploitation)
            if np.random.rand() < 0.2:
                chaos_dim = np.random.randint(dim)
                original_value = positions[i, chaos_dim]
                chaos_sequence = tent_map(5, np.random.rand())  # Generate a short chaotic sequence
                for chaos_val in chaos_sequence:
                    perturbed_value = X_best[chaos_dim] + (ub - lb) * chaos_val * 0.1 # Small perturbation
                    perturbed_position = positions[i].copy()
                    perturbed_position[chaos_dim] = np.clip(perturbed_value, lb, ub)
                    perturbed_fitness = distance(perturbed_position, prey_pos[0], prey_pos[1])
                    if perturbed_fitness < fitness[i]:
                        positions[i] = perturbed_position
                        fitness[i] = perturbed_fitness
                        if perturbed_fitness < best_fitness:
                            best_fitness = perturbed_fitness
                            X_best = perturbed_position.copy()
                            break # Move to the next predator after improvement

        # Boundary check
        positions[i] = np.clip(positions[i], lb, ub)

    # Opposition-Based Learning
    positions = opposition_based_learning(positions, fitness, lb, ub)
    fitness = np.array([distance(pos, prey_pos[0], prey_pos[1]) for pos in positions])
    best_index = np.argmin(fitness)
    if fitness[best_index] < best_fitness:
        best_fitness = fitness[best_index]
        X_best = positions[best_index].copy()

    convergence_curve[iter] = best_fitness

    if iter % 10 == 0:
        print(f'Iteration: {iter}, Best Distance: {best_fitness:.4f}')

    plt.title(f'MPA with Restart, OBL, & Chaotic LS - Iteration {iter+1}')
    plt.pause(0.01)

plt.close()

# Convergence plot
plt.figure(figsize=(10, 6))
plt.semilogy(convergence_curve, 'b', linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Best Distance to Prey')
plt.title('MPA with Restart, OBL, & Chaotic LS Convergence Curve')
plt.grid(True)
plt.show()


# In[ ]:




