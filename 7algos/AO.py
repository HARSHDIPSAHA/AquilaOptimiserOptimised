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
    
    # Calculate mean position
    X_M = np.mean(positions, axis=0)
    
    for i in range(num_aquilas):
        if iter <= (2/3)*max_iter:  # Exploration phase
            if np.random.rand() < 0.5:
                # Expanded exploration (High soar with vertical stoop)
                positions[i] = X_best * (1 - iter/max_iter) + (X_M - X_best) * np.random.rand()
            else:
                # Narrowed exploration (Contour flight with short glide)
                r = np.random.rand()
                theta = np.random.rand() * 2 * np.pi
                x = r * np.sin(theta)
                y = r * np.cos(theta)
                
                # Lévy flight calculation
                sigma = (np.math.gamma(1+beta)*np.sin(np.pi*beta/2)/(np.math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
                u = np.random.randn(dim) * sigma
                v = np.random.randn(dim)
                levy = 0.01 * u / np.abs(v)**(1/beta)
                
                positions[i] = X_best * levy + positions[np.random.randint(num_aquilas)] + (y - x) * np.random.rand()
        else:  # Exploitation phase
            if np.random.rand() < 0.5:
                # Expanded exploitation (Low flight with gradual descent)
                positions[i] = (X_best * X_M) * alpha - np.random.rand() + ((ub - lb) * np.random.rand() + lb) * delta
            else:
                # Narrowed exploitation (Walk and grab prey)
                QF = iter**((2*np.random.rand()-1)/(1 - max_iter)**2)
                G1 = 2 * np.random.rand() - 1
                G2 = 2 * (1 - iter/max_iter)
                levy = np.random.rand()  # Simplified Lévy for visualization
                
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