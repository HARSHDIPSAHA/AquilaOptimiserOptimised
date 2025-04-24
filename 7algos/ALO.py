# Code for ALO (Ant Lion Optimizer)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backend_bases import MouseButton
matplotlib.use('TkAgg')

def distance(position, xprey, yprey):
    return np.sqrt((position[0] - xprey)**2 + (position[1] - yprey)**2)

plt.figure(figsize=(8, 8))
plt.grid(True)
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.title('Click to place prey (first) and Ants (next clicks)\nRight click when done')

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
n = len(positions)
max_iter = 500
dim = 2
lb = 0
ub = 100

# Initialize antlions (same as ants at start)
antlions = positions.copy()
X_best = positions[0].copy()
best_fitness = float('inf')
convergence_curve = np.zeros(max_iter)

for i in range(n):
    current_fitness = distance(positions[i], prey_pos[0], prey_pos[1])
    if current_fitness < best_fitness:
        best_fitness = current_fitness
        X_best = positions[i].copy()

def reflective_boundaries(position, lb, ub):
    position = np.where(position < lb, 2 * lb - position, position)
    position = np.where(position > ub, 2 * ub - position, position)
    return position

def random_walk(dim, max_iter):
    walk = np.zeros((max_iter, dim))
    for d in range(dim):
        s = np.random.choice([-1, 1], size=max_iter)
        walk[:, d] = np.cumsum(s)
    # Normalize
    min_walk = walk.min(axis=0)
    max_walk = walk.max(axis=0)
    walk = (walk - min_walk) / (max_walk - min_walk + 1e-12)
    return walk

plt.figure(figsize=(8, 8))
for iter in range(max_iter):
    plt.clf()
    plt.grid(True)
    plt.xlim(-10, 110)
    plt.ylim(-10, 110)

    plt.plot(prey_pos[0], prey_pos[1], 'ks', markersize=12, markerfacecolor='magenta')
    plt.plot(positions[:, 0], positions[:, 1], 'ro', markersize=8, markerfacecolor='blue')
    plt.plot(X_best[0], X_best[1], 'g*', markersize=15, markerfacecolor='lime')

    # Sort antlions (lower distance is better)
    fitness = np.array([distance(antlions[i], prey_pos[0], prey_pos[1]) for i in range(n)])
    sorted_idx = np.argsort(fitness)
    elite_antlion = antlions[sorted_idx[0]].copy()

    # Adaptive boundaries
    I = 1 + 4 * iter / max_iter
    lb_iter = lb / I
    ub_iter = ub / I

    for i in range(n):
        # Roulette wheel selection
        probs = 1.0 / (fitness + 1e-12)
        probs /= probs.sum()
        selected_idx = np.random.choice(n, p=probs)
        selected_antlion = antlions[selected_idx]

        # Random walks
        RW_selected = random_walk(dim, max_iter)
        RW_elite = random_walk(dim, max_iter)

        # Map walks to search space
        pos_selected = lb_iter + RW_selected[iter] * (ub_iter - lb_iter)
        pos_elite = lb_iter + RW_elite[iter] * (ub_iter - lb_iter)

        # Update ant position
        positions[i] = (pos_selected + pos_elite) / 2.0
        positions[i] = reflective_boundaries(positions[i], lb, ub)

        current_fitness = distance(positions[i], prey_pos[0], prey_pos[1])
        if current_fitness < best_fitness:
            best_fitness = current_fitness
            X_best = positions[i].copy()

    # Update antlions if ants are better
    for i in range(n):
        if distance(positions[i], prey_pos[0], prey_pos[1]) < distance(antlions[i], prey_pos[0], prey_pos[1]):
            antlions[i] = positions[i].copy()

    convergence_curve[iter] = best_fitness

    if iter % 10 == 0:
        print(f'Iteration: {iter}, Best Distance: {best_fitness:.4f}')
        plt.title(f'Ant Lion Optimizer - Iteration {iter+1}')
        plt.pause(0.01)

plt.close()

plt.figure(figsize=(10, 6))
plt.semilogy(convergence_curve, 'b', linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Best Distance to Prey')
plt.title('ALO Convergence Curve')
plt.grid(True)
plt.show()
