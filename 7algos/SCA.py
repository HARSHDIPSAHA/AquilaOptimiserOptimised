# Code for GWO (Grey Wolf Optimizer)
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
plt.title('Click to place prey (first) and Wolves (next clicks)\nRight click when done')

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

# Initialize alpha, beta, delta wolves
fitness = np.array([distance(pos, prey_pos[0], prey_pos[1]) for pos in positions])
idx = np.argsort(fitness)
alpha_pos = positions[idx[0]].copy()
alpha_score = fitness[idx[0]]
beta_pos = positions[idx[1]].copy()
beta_score = fitness[idx[1]]
delta_pos = positions[idx[2]].copy()
delta_score = fitness[idx[2]]

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
    plt.plot(alpha_pos[0], alpha_pos[1], 'g*', markersize=15, markerfacecolor='lime')

    a = 2 - iter * (2 / max_iter)  # linearly decreases from 2 to 0

    for i in range(n):
        for l in range(dim):
            r1 = np.random.rand()
            r2 = np.random.rand()

            A1 = 2 * a * r1 - a
            C1 = 2 * r2
            D_alpha = abs(C1 * alpha_pos[l] - positions[i, l])
            X1 = alpha_pos[l] - A1 * D_alpha

            r1 = np.random.rand()
            r2 = np.random.rand()
            A2 = 2 * a * r1 - a
            C2 = 2 * r2
            D_beta = abs(C2 * beta_pos[l] - positions[i, l])
            X2 = beta_pos[l] - A2 * D_beta

            r1 = np.random.rand()
            r2 = np.random.rand()
            A3 = 2 * a * r1 - a
            C3 = 2 * r2
            D_delta = abs(C3 * delta_pos[l] - positions[i, l])
            X3 = delta_pos[l] - A3 * D_delta

            positions[i, l] = (X1 + X2 + X3) / 3

        positions[i] = reflective_boundaries(positions[i], lb, ub)

    # Update alpha, beta, delta
    for i in range(n):
        fit = distance(positions[i], prey_pos[0], prey_pos[1])
        if fit < alpha_score:
            alpha_score = fit
            alpha_pos = positions[i].copy()
        elif fit < beta_score and fit > alpha_score:
            beta_score = fit
            beta_pos = positions[i].copy()
        elif fit < delta_score and fit > beta_score and fit > alpha_score:
            delta_score = fit
            delta_pos = positions[i].copy()

    convergence_curve[iter] = alpha_score

    if iter % 10 == 0:
        print(f'Iteration: {iter}, Best Distance: {alpha_score:.4f}')
        plt.title(f'Grey Wolf Optimizer - Iteration {iter+1}')
        plt.pause(0.01)

plt.close()

plt.figure(figsize=(10, 6))
plt.semilogy(convergence_curve, 'b', linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Best Distance to Prey')
plt.title('GWO Convergence Curve')
plt.grid(True)
plt.show()
