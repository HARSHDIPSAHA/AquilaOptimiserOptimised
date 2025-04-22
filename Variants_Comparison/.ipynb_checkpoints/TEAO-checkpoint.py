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
    """Tent chaos map for initialization."""
    x = np.zeros(n)
    x[0] = x0
    for i in range(n - 1):
        if x[i] < 0.5:
            x[i+1] = 2 * x[i]
        else:
            x[i+1] = 2 * (1 - x[i])
        # Introduce a small perturbation to avoid fixed points and cycles
        x[i+1] = (1 - 1e-6) * x[i+1] + 1e-6 * np.random.rand()
    return x

def random_opposition(position, lb, ub):
    """Random Opposition-Based Learning."""
    dim = len(position)
    opposition = np.zeros_like(position)
    for i in range(dim):
        rand = np.random.rand()
        opposition[i] = lb + ub - position[i] + rand * (position[i] - (lb + ub) / 2)
    return np.clip(opposition, lb, ub)

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
FADs = 0.2  # Fish Aggregating Devices effect
P = 0.5     # Probability of switching from top predator to Lévy

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

    # Sort predators based on fitness
    sorted_indices = np.argsort(fitness)
    sorted_positions = positions[sorted_indices]

    CF = (1 - iter / max_iter)**(2 * iter / max_iter)
    t = iter + 1
    T = max_iter

    for i in range(num_predators):
        if iter < max_iter / 3:  # Phase 1: High exploration (no change)
            step_size = np.random.rand(dim) * (sorted_positions[i] - prey_pos) * CF
            positions[i] = positions[i] + step_size
        elif iter < 2 * max_iter / 3:  # Phase 2: Transition phase (no change)
            if np.random.rand() < P:
                step_size = np.random.randn(dim) * (sorted_positions[i] - prey_pos)
                positions[i] = positions[i] + step_size
            else:
                step_size = levy.rvs(1.5, size=dim) * (sorted_positions[i] - prey_pos) * CF
                positions[i] = positions[i] + step_size
        else:  # Phase 3: High exploitation (new update rules)
            phi = 2 * np.random.rand(dim) - 1  # Random parameter in (-1, 1)
            decay_factor = (1 - t / T)**(2 * t / T)

            if np.random.rand() < 0.5:  # Expanded Exploitation
                positions[i] = X_best + decay_factor * (X_best - positions[i]) * phi
            else:  # Narrowed Exploitation (Lévy flight)
                LF = levy.rvs(1.5, size=dim)
                positions[i] = X_best + LF * (X_best - positions[i]) * np.random.rand(dim)

        # Boundary check
        positions[i] = np.clip(positions[i], lb, ub)

        # Apply Random Opposition-Based Learning with a probability
        if np.random.rand() < 0.1:
            opposition = random_opposition(positions[i], lb, ub)
            opposition_fitness = distance(opposition, prey_pos[0], prey_pos[1])
            if opposition_fitness < fitness[i]:
                positions[i] = opposition
                fitness[i] = opposition_fitness
                if opposition_fitness < best_fitness:
                    best_fitness = opposition_fitness
                    X_best = opposition.copy()

    # Evaluate current population
    fitness = np.array([distance(pos, prey_pos[0], prey_pos[1]) for pos in positions])
    current_best_index = np.argmin(fitness)
    if fitness[current_best_index] < best_fitness:
        best_fitness = fitness[current_best_index]
        X_best = positions[current_best_index].copy()

    convergence_curve[iter] = best_fitness

    if iter % 10 == 0:
        print(f'Iteration: {iter}, Best Distance: {best_fitness:.4f}')

    plt.title(f'MPA with Tent Initialization & New Updates - Iteration {iter+1}')
    plt.pause(0.01)

plt.close()

# Convergence plot
plt.figure(figsize=(10, 6))
plt.semilogy(convergence_curve, 'b', linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Best Distance to Prey')
plt.title('MPA with Tent Initialization & New Updates Convergence Curve')
plt.grid(True)
plt.show()


# In[ ]:




