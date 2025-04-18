import numpy as np

def reflective_boundaries(position, lb, ub):
    position = np.where(position < lb, 2 * lb - position, position)
    position = np.where(position > ub, 2 * ub - position, position)
    return np.clip(position, lb, ub)

def three_bar_truss_objective(x):
    A1 = x[0]
    A2 = x[1]
    H = 100
    volume = (2 * A1 * np.sqrt(2) * H) + (A2 * H)
    return volume

def three_bar_truss_constraints(x):
    A1 = x[0]
    A2 = x[1]
    P = 2
    sigma = 2
    g1 = (P / (sigma * A1)) * (np.sqrt(2) + A2/A1) - 1
    g2 = (P / (sigma * A2)) * (2 + np.sqrt(2) * A1/A2) - 1
    g3 = (P / (sigma * A1)) * (np.sqrt(2) + 1) - 1
    return np.array([g1, g2, g3])

def fitness_function(x):
    obj = three_bar_truss_objective(x)
    constraints = three_bar_truss_constraints(x)
    violation = sum(max(0, g) for g in constraints)
    if violation > 0:
        return obj + 1e6 * violation
    return obj

def mvao_three_bar_truss(max_iter=500, population_size=50,random_state=None):
    
    if random_state is not None:
        np.random.seed(random_state)#ahh..... reproducibility
    dim = 2
    lb = np.array([0.0, 0.0])
    ub = np.array([1.0, 1.0])
    alpha = 0.1
    delta = 0.1
    beta = 1.8
    positions = np.zeros((population_size, dim))
    for i in range(dim):
        positions[:, i] = np.random.uniform(lb[i], ub[i], population_size)
    fitness = np.array([fitness_function(p) for p in positions])
    best_idx = np.argmin(fitness)
    X_best = positions[best_idx].copy()
    best_fitness = fitness[best_idx]
    for iter in range(max_iter):
        weights = 1 / (fitness + np.finfo(float).eps)
        weighted_mean = np.sum(positions * weights[:, np.newaxis], axis=0) / np.sum(weights)
        for i in range(population_size):
            if iter <= (2/3) * max_iter:
                if np.random.rand() < 0.5:
                    positions[i] = X_best * (1 - iter / max_iter) + (weighted_mean - X_best) * np.random.rand()
                else:
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
                    positions[i] = (X_best * weighted_mean) * alpha - np.random.rand() + ((ub - lb) * np.random.rand() + lb) * delta
                else:
                    QF = iter**((2*np.random.rand()-1)/(1 - max_iter)**2)
                    G1 = 2 * np.random.rand() - 1
                    G2 = 2 * (1 - iter / max_iter)
                    levy = np.random.rand(dim)
                    positions[i] = QF * X_best - G1 * positions[i] * np.random.rand() - G2 * levy + np.random.rand(dim)
            positions[i] = reflective_boundaries(positions[i], lb, ub)
            fitness[i] = fitness_function(positions[i])
            if fitness[i] < best_fitness:
                best_fitness = fitness[i]
                X_best = positions[i].copy()
    final_obj = three_bar_truss_objective(X_best)
    final_constraints = three_bar_truss_constraints(X_best)
    constraints_satisfied = all(g <= 0 for g in final_constraints)
    return X_best, final_obj, constraints_satisfied

best_solution, best_fitness, constraints_satisfied = mvao_three_bar_truss(random_state=42)
print("\nThree-Bar Truss Design Results:")
print(f"Cross-sectional area A1 = A3: {best_solution[0]:.6f}")
print(f"Cross-sectional area A2: {best_solution[1]:.6f}")
print(f"Minimum volume/weight: {best_fitness:.6f}")
print(f"All constraints satisfied: {constraints_satisfied}")
