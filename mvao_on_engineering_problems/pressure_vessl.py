import numpy as np

def reflective_boundaries(position, lb, ub):
    position = np.where(position < lb, 2 * lb - position, position)
    position = np.where(position > ub, 2 * ub - position, position)
    return np.clip(position, lb, ub)

def pressure_vessel_objective(x):
    Ts = x[0]
    Th = x[1]
    R = x[2]
    L = x[3]
    cost = 0.6224 * Ts * R * L + 1.7781 * Th * R**2 + 3.1661 * Ts**2 * L + 19.84 * Ts**2 * R
    return cost

def pressure_vessel_constraints(x):
    Ts = x[0]
    Th = x[1]
    R = x[2]
    L = x[3]
    g1 = -Ts + 0.0193 * R
    g2 = -Th + 0.00954 * R
    g3 = -np.pi * R**2 * L - (4/3) * np.pi * R**3 + 1296000
    g4 = L - 240
    return np.array([g1, g2, g3, g4])

def fitness_function(x):
    obj = pressure_vessel_objective(x)
    constraints = pressure_vessel_constraints(x)
    violation = sum(max(0, g) for g in constraints)
    if violation > 0:
        return obj + 1e6 * violation
    return obj

def mvao_pressure_vessel(max_iter=500, population_size=50):
    dim = 4
    lb = np.array([0.0625, 0.0625, 10.0, 10.0])
    ub = np.array([99.0, 99.0, 200.0, 200.0])
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
    final_obj = pressure_vessel_objective(X_best)
    final_constraints = pressure_vessel_constraints(X_best)
    constraints_satisfied = all(g <= 0 for g in final_constraints)
    return X_best, final_obj, constraints_satisfied

best_solution, best_fitness, constraints_satisfied = mvao_pressure_vessel()
print("\nPressure Vessel Design Results:")
print(f"Shell thickness (Ts): {best_solution[0]:.6f}")
print(f"Head thickness (Th): {best_solution[1]:.6f}")
print(f"Inner radius (R): {best_solution[2]:.6f}")
print(f"Cylinder length (L): {best_solution[3]:.6f}")
print(f"Minimum cost: {best_fitness:.6f}")
print(f"All constraints satisfied: {constraints_satisfied}")
