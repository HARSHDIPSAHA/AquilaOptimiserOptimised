#!/usr/bin/env python
# coding: utf-8

# In[20]:


import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib
from matplotlib.backend_bases import MouseButton
from scipy.stats import t
matplotlib.use('TkAgg')

def distance(position, xprey, yprey):
    return np.sqrt((position[0] - xprey)**2 + (position[1] - yprey)**2)

def tent_map(n, x0=0.7):
    """Improved tent chaos map."""
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

plt.figure(figsize=(8, 8))
plt.grid(True)
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.title('Click to place prey (first) and Aquilas (next clicks)\nRight click when done')

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
num_aquilas = len(positions)
max_iter = 500
dim = 2
lb = 0
ub = 100

# AO Parameters
alpha = 0.1
delta = 0.1
beta = 1.5  # For Lévy flight

X_best = positions[0].copy()
best_fitness = float('inf')
convergence_curve = np.zeros(max_iter)

# Initialize population using improved tent chaos map
initial_positions = np.zeros((num_aquilas, dim))
for d in range(dim):
    chaos_sequence = tent_map(num_aquilas, np.random.rand())
    initial_positions[:, d] = lb + (ub - lb) * chaos_sequence
positions = initial_positions.copy()

# Initialize best solution
for i in range(num_aquilas):
    current_fitness = distance(positions[i], prey_pos[0], prey_pos[1])
    if current_fitness < best_fitness:
        best_fitness = current_fitness
        X_best = positions[i].copy()

plt.figure(figsize=(8, 8))
for iter in range(max_iter):
    plt.clf()
    plt.grid(True)
    plt.xlim(-10, 110)
    plt.ylim(-10, 110)

    plt.plot(prey_pos[0], prey_pos[1], 'ks', markersize=12, markerfacecolor='magenta')
    plt.plot(positions[:, 0], positions[:, 1], 'ro', markersize=8, markerfacecolor='blue')
    plt.plot(X_best[0], X_best[1], 'g*', markersize=15, markerfacecolor='lime')

    a = 2 * (1 - iter/max_iter)  # Exploration-exploitation balance

    # Adaptive t-distribution parameter
    df = 5 + 95 * (iter / max_iter)  # Degrees of freedom, increases over time

    # Calculate mean position
    X_M = np.mean(positions, axis=0)

    for i in range(num_aquilas):
        if iter <= (2/3)*max_iter:  # Exploration phase
            if np.random.rand() < 0.5:
                # Expanded exploration (High soar with vertical stoop)
                positions[i] = X_best * (1 - iter/max_iter) + (X_M - X_best) * t.rvs(df, size=dim)
            else:
                # Narrowed exploration (Contour flight with short glide)
                r = np.random.rand()
                theta = np.random.rand() * 2 * np.pi
                x = r * np.sin(theta)
                y = r * np.cos(theta)

                # Lévy flight calculation
                sigma = (math.gamma(1+beta)*math.sin(math.pi*beta/2)/(math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
                u = t.rvs(df, size=dim) * sigma # Using t-distribution for Lévy-like steps
                v = t.rvs(df, size=dim)
                levy = 0.01 * u / np.abs(v)**(1/beta)

                positions[i] = X_best * levy + positions[np.random.randint(num_aquilas)] + (y - x) * np.random.rand()
        else:  # Exploitation phase
            if np.random.rand() < 0.5:
                # Expanded exploitation (Low flight with gradual descent)
                positions[i] = (X_best - X_M) * alpha - np.random.rand() + ((ub - lb) * np.random.rand() + lb) * delta
            else:
                # Narrowed exploitation (Walk and grab prey)
                QF = iter**((2*np.random.rand()-1)/(1 - max_iter)**2)
                G1 = 2 * np.random.rand() - 1
                G2 = 2 * (1 - iter/max_iter)
                levy = np.random.randn(dim) * 0.1 # Gaussian-like steps for exploitation

                positions[i] = QF * X_best - G1 * positions[i] * np.random.rand() - G2 * levy + np.random.rand()

        # Boundary check
        positions[i] = np.clip(positions[i], lb, ub)

        # Update best solution
        current_fitness = distance(positions[i], prey_pos[0], prey_pos[1])
        if current_fitness < best_fitness:
            best_fitness = current_fitness
            X_best = positions[i].copy()

    convergence_curve[iter] = best_fitness

    if iter % 10 == 0:
        print(f'Iteration: {iter}, Best Distance: {best_fitness:.4f}')

    plt.title(f'Aquila Optimizer - Iteration {iter+1}')
    plt.pause(0.01)

plt.close()

# Convergence plot
plt.figure(figsize=(10, 6))
plt.semilogy(convergence_curve, 'b', linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Best Distance to Prey')
plt.title('Aquila Optimizer Convergence Curve')
plt.grid(True)
plt.show()


# In[ ]:





# In[ ]:




