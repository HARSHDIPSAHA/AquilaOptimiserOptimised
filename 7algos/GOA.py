# Code for GOA (Grasshopper Optimization Algorithm)
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
plt.title('Click to place prey (first) and Grasshoppers (next clicks)\nRight click when done')

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

plt.figure(figsize=(8, 8))
cMax = 1.0
cMin = 0.00001

for iter in range(max_iter):
    plt.clf()
    plt.grid(True)
    plt.xlim(-10, 110)
    plt.ylim(-10, 110)

    plt.plot(prey_pos[0], prey_pos[1], 'ks', markersize=12, markerfacecolor='magenta')
    plt.plot(positions[:, 0], positions[:, 1], 'ro', markersize=8, markerfacecolor='blue')
    plt.plot(X_best[0], X_best[1], 'g*', markersize=15, markerfacecolor='lime')

    c = cMax - iter*(cMax - cMin)/max_iter
    
    for i in range(n):
        new_pos = np.zeros(dim)
        total_force = np.zeros(dim)
        
        for j in range(n):
            if i != j:
                dist = np.linalg.norm(positions[i] - positions[j])
                r_ij_vec = (positions[j] - positions[i]) / (dist + 1e-20)
                social_force = 0.5 * np.exp(-dist/1.5) - np.exp(-dist)
                total_force += social_force * r_ij_vec
                
        gravity_force = 0.5 * (X_best - positions[i])
        wind_direction = 0.4 * (X_best - positions[i])
        
        new_pos = c*(total_force) + gravity_force + wind_direction
        new_pos = positions[i] + c*new_pos
        new_pos = reflective_boundaries(new_pos, lb, ub)
        
        current_fitness = distance(new_pos, prey_pos[0], prey_pos[1])
        if current_fitness < best_fitness:
            best_fitness = current_fitness
            X_best = new_pos.copy()
            positions[i] = new_pos.copy()
        else:
            positions[i] = new_pos.copy()

    convergence_curve[iter] = best_fitness

    if iter % 10 == 0:
        print(f'Iteration: {iter}, Best Distance: {best_fitness:.4f}')
        plt.title(f'Grasshopper Optimization - Iteration {iter+1}')
        plt.pause(0.01)

plt.close()

plt.figure(figsize=(10, 6))
plt.semilogy(convergence_curve, 'b', linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Best Distance to Prey')
plt.title('GOA Convergence Curve')
plt.grid(True)
plt.show()
