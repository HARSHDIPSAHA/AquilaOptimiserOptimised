# Code for PSO (Particle Swarm Optimization)
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
plt.title('Click to place prey (first) and Particles (next clicks)\nRight click when done')

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

# PSO Parameters
w = 0.7
c1 = 1.5
c2 = 1.5

velocities = np.random.uniform(-1, 1, (n, dim))
pbest_positions = positions.copy()
pbest_fitness = np.array([distance(pos, prey_pos[0], prey_pos[1]) for pos in positions])
gbest_index = np.argmin(pbest_fitness)
gbest_position = pbest_positions[gbest_index].copy()
gbest_fitness = pbest_fitness[gbest_index]
convergence_curve = np.zeros(max_iter)

def reflective_boundaries(position, lb, ub):
    position = np.where(position < lb, 2 * lb - position, position)
    position = np.where(position > ub, 2 * ub - position, position)
    return position

plt.figure(figsize=(8, 8))
for iter in range(max_iter):
    plt.clf()
    plt.grid(True)
    plt.xlim(-10, 110)
    plt.ylim(-10, 110)

    plt.plot(prey_pos[0], prey_pos[1], 'ks', markersize=12, markerfacecolor='magenta')
    plt.plot(positions[:, 0], positions[:, 1], 'ro', markersize=8, markerfacecolor='blue')
    plt.plot(gbest_position[0], gbest_position[1], 'g*', markersize=15, markerfacecolor='lime')

    for i in range(n):
        r1 = np.random.rand(dim)
        r2 = np.random.rand(dim)
        velocities[i] = (w * velocities[i] +
                         c1 * r1 * (pbest_positions[i] - positions[i]) +
                         c2 * r2 * (gbest_position - positions[i]))
        positions[i] = positions[i] + velocities[i]
        positions[i] = reflective_boundaries(positions[i], lb, ub)

        current_fitness = distance(positions[i], prey_pos[0], prey_pos[1])
        if current_fitness < pbest_fitness[i]:
            pbest_fitness[i] = current_fitness
            pbest_positions[i] = positions[i].copy()

        if current_fitness < gbest_fitness:
            gbest_fitness = current_fitness
            gbest_position = positions[i].copy()

    convergence_curve[iter] = gbest_fitness

    if iter % 10 == 0:
        print(f'Iteration: {iter}, Best Distance: {gbest_fitness:.4f}')
        plt.title(f'Particle Swarm Optimization - Iteration {iter+1}')
        plt.pause(0.01)

plt.close()

plt.figure(figsize=(10, 6))
plt.semilogy(convergence_curve, 'b', linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Best Distance to Prey')
plt.title('PSO Convergence Curve')
plt.grid(True)
plt.show()
