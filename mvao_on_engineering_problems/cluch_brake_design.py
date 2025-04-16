import numpy as np

def reflective_boundaries(position, lb, ub):
    position = np.where(position < lb, 2 * lb - position, position)
    position = np.where(position > ub, 2 * ub - position, position)
    return np.clip(position, lb, ub)

def multiple_disc_clutch_brake_objective(x):
    ri = x[0]
    ro = x[1]
    t = x[2]
    F = x[3]
    Z = int(round(x[4]))
    weight = np.pi * t * (ro**2 - ri**2) * Z
    return weight

def multiple_disc_clutch_brake_constraints(x):
    ri = x[0]
    ro = x[1]
    t = x[2]
    F = x[3]
    Z = int(round(x[4]))
    Mf = 3
    Ms = 40
    Iz = 55
    n = 250
    mu = 0.5
    s = 1.5
    Tmax = 15
    pmax = 1
    Vsrmax = 10
    rho = 0.0000078
    deltat = 0.5
    c = 400
    omega = 2 * np.pi * n / 60
    Mh = 2/3 * mu * F * Z * (ro**3 - ri**3) / (ro**2 - ri**2)
    pmax_calc = F / (np.pi * (ro**2 - ri**2))
    Vsr = 2/3 * np.pi * n * (ro**3 - ri**3) / (30 * (ro**2 - ri**2))
    T = Iz * omega**2 / (2 * Mh)
    g1 = ro - ri - 20
    g2 = 30 - Z * t
    g3 = pmax_calc - pmax
    g4 = Vsrmax - Vsr
    g5 = Mh - s * Ms
    g6 = T - Tmax
    g7 = Mf - Mh
    g8 = 0.00001 - t
    return np.array([g1, g2, g3, g4, g5, g6, g7, g8])

def fitness_function(x):
    x_eval = x.copy()
    x_eval[4] = int(round(x_eval[4]))
    obj = multiple_disc_clutch_brake_objective(x_eval)
    constraints = multiple_disc_clutch_brake_constraints(x_eval)
    violation = sum(max(0, -g) for g in constraints)
    if violation > 0:
        return obj + 1e6 * violation
    return obj

def mvao_multiple_disc_clutch_brake(max_iter=500, population_size=50):
    dim = 5
    lb = np.array([60.0, 90.0, 1.0, 600.0, 2.0])
    ub = np.array([80.0, 110.0, 3.0, 1000.0, 9.0])
    alpha = 0.1
    delta = 0.1
    beta = 1.8
    positions = np.zeros((population_size, dim))
    for i in range(dim):
        positions[:, i] = np.random.uniform(lb[i], ub[i], population_size)
    positions[:, 4] = np.round(positions[:, 4])
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
            positions[i, 4] = np.round(positions[i, 4])
            fitness[i] = fitness_function(positions[i])
            if fitness[i] < best_fitness:
                best_fitness = fitness[i]
                X_best = positions[i].copy()
    X_best[4] = int(round(X_best[4]))
    final_obj = multiple_disc_clutch_brake_objective(X_best)
    final_constraints = multiple_disc_clutch_brake_constraints(X_best)
    constraints_satisfied = all(g >= 0 for g in final_constraints)
    return X_best, final_obj, constraints_satisfied

best_solution, best_fitness, constraints_satisfied = mvao_multiple_disc_clutch_brake()

print("\nMultiple Disc Clutch Brake Design Results:")
print(f"Inner radius (ri): {best_solution[0]:.6f}")
print(f"Outer radius (ro): {best_solution[1]:.6f}")
print(f"Thickness (t): {best_solution[2]:.6f}")
print(f"Actuating force (F): {best_solution[3]:.6f}")
print(f"Number of friction surfaces (Z): {int(round(best_solution[4]))}")
print(f"Minimum weight: {best_fitness:.6f}")
print(f"All constraints satisfied: {constraints_satisfied}")
