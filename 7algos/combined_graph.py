import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
import math
import matplotlib
matplotlib.use("TkAgg")

# CEC Benchmark Functions
class CECFunction(Enum):
    F1 = 1   # Sphere
    F2 = 2   # Elliptic
    F3 = 3   # Bent Cigar
    F4 = 4   # Discus
    F5 = 5   # Rosenbrock
    F6 = 6   # Ackley
    F7 = 7   # Rastrigin
    F8 = 8   # Schaffer F7
    F9 = 9   # Griewank
    F10 = 10  # Weierstrass
    F11 = 11  # Katsuura
    F12 = 12  # Lunacek bi-Rastrigin
    F13 = 13  # Step
    F14 = 14  # HappyCat
    F15 = 15  # HGBat
    F16 = 16  # Expanded Griewank plus Rosenbrock
    F17 = 17  # Expanded Scaffer F6
    # Composition Functions:
    F18 = 18
    F19 = 19
    F20 = 20
    F21 = 21
    F22 = 22
    F23 = 23
    F24 = 24
    F25 = 25
    F26 = 26
    F27 = 27
    F28 = 28
    F29 = 29
    F30 = 30

# Example approximations or placeholders
def cec_function(x, func: CECFunction):
    x = np.asarray(x)
    if func == CECFunction.F1:
        return np.sum(x**2)
    elif func == CECFunction.F2:
        return np.sum(10**6**(np.arange(len(x))/(len(x)-1)) * x**2)
    elif func == CECFunction.F3:
        return x[0]**2 + 10**6 * np.sum(x[1:]**2)
    elif func == CECFunction.F4:
        return 10**6 * x[0]**2 + np.sum(x[1:]**2)
    elif func == CECFunction.F5:
        return np.sum(100*(x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2)
    elif func == CECFunction.F6:
        return -20*np.exp(-0.2*np.sqrt(np.mean(x**2))) - \
               np.exp(np.mean(np.cos(2*np.pi*x))) + 20 + np.e
    elif func == CECFunction.F7:
        return 10*len(x) + np.sum(x**2 - 10*np.cos(2*np.pi*x))
    elif func == CECFunction.F8:
        return np.sum(np.power(np.sum(x[:i+1]**2), 0.25) *
                      (1 + np.sin(50 * np.power(np.sum(x[:i+1]**2), 0.1))**2)
                      for i in range(len(x)-1))
    elif func == CECFunction.F9:
        return 1 + (1/4000)*np.sum(x**2) - np.prod(np.cos(x/np.sqrt(np.arange(1, len(x)+1))))
    elif func == CECFunction.F10:
        a = 0.5
        b = 3
        kmax = 20
        return np.sum([np.sum([a**k * np.cos(2*np.pi*b**k*(xi+0.5)) for k in range(kmax)])
                      - len(x) * np.sum([a**k * np.cos(np.pi*b**k) for k in range(kmax)]) for xi in x])
    elif func == CECFunction.F11:
        return np.prod((1 + (np.arange(1, len(x)+1)) * (np.abs(x)**(0.5 + np.arange(1, len(x)+1)/100)))) - 1
    elif func == CECFunction.F12:
        return np.sum((x - 1)**2) + 10 * (len(x) - np.sum(np.cos(2*np.pi*(x - 1))))
    elif func == CECFunction.F13:
        return np.sum(np.floor(x + 0.5)**2)
    elif func == CECFunction.F14:
        alpha = 1/8
        return np.sum((np.sum(x**2) - len(x))**2)**alpha + (0.5 * np.sum(x**2) + np.sum(x)) / len(x)
    elif func == CECFunction.F15:
        return (np.abs(np.sum(x**2)**2 - np.sum(x))**0.5) + (0.5 * np.sum(x**2) + np.sum(x)) / len(x)
    elif func == CECFunction.F16:
        # Simplified: combine Griewank and Rosenbrock
        part1 = np.sum(100*(x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2)
        part2 = 1 + (1/4000)*np.sum(x**2) - np.prod(np.cos(x/np.sqrt(np.arange(1, len(x)+1))))
        return part1 + part2
    elif func == CECFunction.F17:
        return np.sum(0.5 + (np.sin(np.sqrt(x[:-1]**2 + x[1:]**2))**2 - 0.5) /
                      ((1 + 0.001*(x[:-1]**2 + x[1:]**2))**2))
    elif CECFunction.F18.value <= func.value <= CECFunction.F30.value:
        return composition_function(x, func.value)
    else:
        raise ValueError("Unsupported CEC function")

def composition_function(x, fid):
    return np.sum(np.sin(x)) + fid * 100  # placeholder

class Metaheuristic:
    def __init__(self, name, **params):
        self.name = name
        self.params = params
        
    def optimize(self, func, dim, lb, ub, max_iter, pop_size):
        raise NotImplementedError

# ======================
# Algorithm Implementations 
# ======================
class PSO(Metaheuristic):
    def optimize(self, func, dim, lb, ub, max_iter, pop_size):
        positions = np.random.uniform(lb, ub, (pop_size, dim))
        velocities = np.zeros((pop_size, dim))
        pbest_pos = positions.copy()
        pbest_val = np.array([func(x) for x in positions])
        gbest_pos = pbest_pos[np.argmin(pbest_val)]
        gbest_val = np.min(pbest_val)
        
        convergence = np.zeros(max_iter)
        
        for iter in range(max_iter):
            for i in range(pop_size):
                # Update velocity and position
                r1, r2 = np.random.rand(2)
                velocities[i] = (self.params.get('w', 0.7) * velocities[i] +
                                self.params.get('c1', 1.5) * r1 * (pbest_pos[i] - positions[i]) +
                                self.params.get('c2', 1.5) * r2 * (gbest_pos - positions[i]))
                
                positions[i] = np.clip(positions[i] + velocities[i], lb, ub)
                
                # Update personal best
                current_val = func(positions[i])
                if current_val < pbest_val[i]:
                    pbest_val[i] = current_val
                    pbest_pos[i] = positions[i].copy()
                    
                    # Update global best
                    if current_val < gbest_val:
                        gbest_val = current_val
                        gbest_pos = positions[i].copy()
            
            convergence[iter] = gbest_val
        
        return convergence

class GWO(Metaheuristic):
    def optimize(self, func, dim, lb, ub, max_iter, pop_size):
        positions = np.random.uniform(lb, ub, (pop_size, dim))
        convergence = np.zeros(max_iter)
        
        alpha_pos = positions[0].copy()
        alpha_score = float('inf')
        
        for iter in range(max_iter):
            a = 2 - iter * (2 / max_iter)  # Linearly decreases from 2 to 0
            
            # Update alpha, beta, delta
            fitness = np.array([func(x) for x in positions])
            sorted_indices = np.argsort(fitness)
            alpha, beta, delta = positions[sorted_indices[:3]]
            
            for i in range(pop_size):
                for d in range(dim):
                    # Update positions
                    r1, r2 = np.random.rand(2)
                    A1 = 2*a*r1 - a
                    C1 = 2*r2
                    D_alpha = abs(C1*alpha[d] - positions[i,d])
                    X1 = alpha[d] - A1*D_alpha
                    
                    r1, r2 = np.random.rand(2)
                    A2 = 2*a*r1 - a
                    C2 = 2*r2
                    D_beta = abs(C2*beta[d] - positions[i,d])
                    X2 = beta[d] - A2*D_beta
                    
                    r1, r2 = np.random.rand(2)
                    A3 = 2*a*r1 - a
                    C3 = 2*r2
                    D_delta = abs(C3*delta[d] - positions[i,d])
                    X3 = delta[d] - A3*D_delta
                    
                    positions[i,d] = np.clip((X1 + X2 + X3)/3, lb, ub)
                
                # Update alpha
                current_val = func(positions[i])
                if current_val < alpha_score:
                    alpha_score = current_val
                    alpha_pos = positions[i].copy()
            
            convergence[iter] = alpha_score
        
        return convergence

# Add implementations for other algorithms following the same pattern

# ======================
# Experimental Setup
# ======================
class GOA(Metaheuristic):
    def __init__(self):
        super().__init__("GOA", cMax=1.0, cMin=0.00001)
        
    def optimize(self, func, dim, lb, ub, max_iter, pop_size):
        # Initialize population
        positions = np.random.uniform(lb, ub, (pop_size, dim))
        velocities = np.zeros((pop_size, dim))
        
        # Initialize best solution
        best_pos = np.zeros(dim)
        best_score = float('inf')
        convergence = np.zeros(max_iter)
        
        # Find initial best
        for i in range(pop_size):
            current_score = func(positions[i])
            if current_score < best_score:
                best_score = current_score
                best_pos = positions[i].copy()

        for iter in range(max_iter):
            c = self.params['cMax'] - iter*(self.params['cMax']-self.params['cMin'])/max_iter
            
            for i in range(pop_size):
                # Calculate social forces
                total_force = np.zeros(dim)
                for j in range(pop_size):
                    if i != j:
                        # Ensure vector operations maintain 1D shape
                        diff = positions[j] - positions[i]
                        dist = np.linalg.norm(diff) + 1e-20
                        r_ij_vec = diff / dist
                        social_force = 0.5*np.exp(-dist/1.5) - np.exp(-dist)
                        total_force += social_force * r_ij_vec
                
                # Calculate new position with explicit dimension handling
                cognitive = 0.5 * (best_pos - positions[i])
                wind_effect = 0.4 * (best_pos - positions[i])
                new_pos = positions[i] + c*(total_force + cognitive + wind_effect)
                
                # Apply boundary constraints
                new_pos = np.clip(new_pos, lb, ub)
                
                # Update position and best solution
                positions[i] = new_pos
                current_score = func(new_pos)
                
                if current_score < best_score:
                    best_score = current_score
                    best_pos = new_pos.copy()

            convergence[iter] = best_score

        return convergence
class MVAO(Metaheuristic):
    def __init__(self):
        super().__init__("MVAO", alpha=0.1, delta=0.1, beta=1.8)
        
    def optimize(self, func, dim, lb, ub, max_iter, pop_size):
        positions = np.random.uniform(lb, ub, (pop_size, dim))
        best_pos = positions[0].copy()
        best_score = float('inf')
        convergence = np.zeros(max_iter)
        
        # Initialize best solution
        for i in range(pop_size):
            current_score = func(positions[i])
            if current_score < best_score:
                best_score = current_score
                best_pos = positions[i].copy()

        def reflective_boundaries(position):
            position = np.where(position < lb, 2*lb - position, position)
            position = np.where(position > ub, 2*ub - position, position)
            return position

        for iter in range(max_iter):
            # Calculate weighted mean position
            fitness_values = np.array([func(pos) for pos in positions])
            weights = 1 / (fitness_values + np.finfo(float).eps)
            weighted_mean = np.sum(positions * weights[:, np.newaxis], axis=0) / np.sum(weights)
            
            a = 2 * (1 - iter/max_iter)  # Exploration-exploitation balance

            for i in range(pop_size):
                if iter <= (2/3)*max_iter:  # Exploration phase
                    if np.random.rand() < 0.5:
                        # Expanded exploration with weighted mean
                        new_pos = best_pos*(1 - iter/max_iter) + (weighted_mean - best_pos)*np.random.rand()
                    else:
                        # Narrowed exploration with Lévy flight
                        # Generate random direction in N-dim space
                        direction = np.random.randn(dim)
                        direction /= np.linalg.norm(direction) + 1e-8
                        
                        # Lévy flight calculation
                        beta = self.params['beta']
                        sigma = (math.gamma(1+beta)*math.sin(math.pi*beta/2) /
                                (math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
                        u = np.random.randn(dim) * sigma
                        v = np.random.randn(dim)
                        levy = 0.01 * u / np.abs(v)**(1/beta)
                        
                        new_pos = best_pos*levy + positions[np.random.randint(pop_size)] + direction*np.random.rand()
                else:  # Exploitation phase
                    if np.random.rand() < 0.5:
                        # Expanded exploitation with weighted mean
                        new_pos = (best_pos*weighted_mean)*self.params['alpha'] - np.random.rand() + \
                                  ((ub - lb)*np.random.rand() + lb)*self.params['delta']
                    else:
                        # Narrowed exploitation
                        QF = iter**((2*np.random.rand()-1)/(1 - max_iter)**2)
                        G1 = 2*np.random.rand() - 1
                        G2 = 2*(1 - iter/max_iter)
                        new_pos = QF*best_pos - G1*positions[i]*np.random.rand() - G2*np.random.rand() + np.random.rand()
                
                # Apply reflective boundaries
                new_pos = reflective_boundaries(new_pos)
                new_pos = np.clip(new_pos, lb, ub)
                
                # Update position and best solution
                current_score = func(new_pos)
                if current_score < best_score:
                    best_score = current_score
                    best_pos = new_pos.copy()
                    positions[i] = new_pos.copy()
                else:
                    positions[i] = new_pos.copy()

            convergence[iter] = best_score

        return convergence
class AO(Metaheuristic):
    def __init__(self):
        super().__init__("AO", alpha=0.1, delta=0.1, beta=1.5)
        
    def optimize(self, func, dim, lb, ub, max_iter, pop_size):
        positions = np.random.uniform(lb, ub, (pop_size, dim))
        best_pos = positions[0].copy()
        best_score = float('inf')
        convergence = np.zeros(max_iter)
        
        # Initialize best solution
        for i in range(pop_size):
            current_score = func(positions[i])
            if current_score < best_score:
                best_score = current_score
                best_pos = positions[i].copy()
        
        for iter in range(max_iter):
            a = 2 * (1 - iter/max_iter)  # Exploration-exploitation balance
            X_mean = np.mean(positions, axis=0)
            
            for i in range(pop_size):
                if iter <= (2/3)*max_iter:  # Exploration phase
                    if np.random.rand() < 0.5:
                        # Expanded exploration
                        new_pos = best_pos*(1-iter/max_iter) + (X_mean-best_pos)*np.random.rand()
                    else:
                        # Narrowed exploration with Lévy flight
                        r = np.random.rand()
                        # Generate random direction vector for any dimension
                        direction = np.random.randn(dim)
                        direction /= np.linalg.norm(direction) + 1e-8
                        
                        # Lévy flight calculation
                        beta = self.params['beta']
                        sigma = (np.math.gamma(1+beta)*np.sin(np.pi*beta/2) /
                                (np.math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
                        u = np.random.randn(dim) * sigma
                        v = np.random.randn(dim)
                        levy = 0.01 * u / np.abs(v)**(1/beta)
                        
                        new_pos = best_pos*levy + positions[np.random.randint(pop_size)] + direction*r
                else:  # Exploitation phase
                    if np.random.rand() < 0.5:
                        # Expanded exploitation
                        new_pos = (best_pos*X_mean)*self.params['alpha'] - np.random.rand() + \
                                  ((ub-lb)*np.random.rand() + lb)*self.params['delta']
                    else:
                        # Narrowed exploitation
                        QF = iter**((2*np.random.rand()-1)/(1-max_iter)**2)
                        G1 = 2*np.random.rand() - 1
                        G2 = 2*(1-iter/max_iter)
                        new_pos = QF*best_pos - G1*positions[i]*np.random.rand() - G2*np.random.rand() + np.random.rand()
                
                # Apply boundaries
                new_pos = np.clip(new_pos, lb, ub)
                current_score = func(new_pos)
                
                # Update position and best solution
                if current_score < best_score:
                    best_score = current_score
                    best_pos = new_pos.copy()
                    positions[i] = new_pos.copy()
                else:
                    positions[i] = new_pos.copy()
            
            convergence[iter] = best_score
        
        return convergence

class EO(Metaheuristic):
    def __init__(self):
        super().__init__("EO", a1=2, a2=1, GP=0.5)
        
    def optimize(self, func, dim, lb, ub, max_iter, pop_size):
        # Adapted from search result [3]
        positions = np.random.uniform(lb, ub, (pop_size, dim))
        best_pos = positions.copy()
        best_score = float('inf')
        convergence = np.zeros(max_iter)

        for iter in range(max_iter):
            fitness = np.array([func(pos) for pos in positions])
            idx = np.argsort(fitness)
            Ceq = positions[idx[:4]]
            Ceq_ave = np.mean(Ceq, axis=0)

            for i in range(pop_size):
                lambda1 = np.random.rand(dim)
                r = np.random.rand(dim)
                F = self.params['a1'] * np.sign(r-0.5) * (np.exp(-lambda1*iter/max_iter) - 1)
                GCP = 0.5 * np.random.rand(dim) * (np.random.rand(dim) < self.params['GP'])
                Ceq_i = Ceq[np.random.randint(0,4)]
                positions[i] = Ceq_i + (positions[i]-Ceq_i)*F + (Ceq_ave-positions[i])*GCP
                positions[i] = np.clip(positions[i], lb, ub)

                current_score = func(positions[i])
                if current_score < best_score:
                    best_score = current_score
                    best_pos = positions[i].copy()

            convergence[iter] = best_score
        return convergence

# Similarly implement SCA, ALO, AO, MPA using their respective search result codes
# [Add implementations for SCA (4), ALO (5), AO (6), MPA (7) following same pattern]
class SCA(Metaheuristic):
    def __init__(self):
        super().__init__("SCA")
    def optimize(self, func, dim, lb, ub, max_iter, pop_size):
        positions = np.random.uniform(lb, ub, (pop_size, dim))
        best_pos = positions[0].copy()
        best_score = float('inf')
        convergence = np.zeros(max_iter)
        for i in range(pop_size):
            score = func(positions[i])
            if score < best_score:
                best_score = score
                best_pos = positions[i].copy()
        for iter in range(max_iter):
            r1 = 2 - iter * (2.0 / max_iter)
            for i in range(pop_size):
                for d in range(dim):
                    r2 = 2 * np.pi * np.random.rand()
                    r3 = 2 * np.random.rand()
                    r4 = np.random.rand()
                    if r4 < 0.5:
                        positions[i, d] = positions[i, d] + r1 * np.sin(r2) * abs(r3 * best_pos[d] - positions[i, d])
                    else:
                        positions[i, d] = positions[i, d] + r1 * np.cos(r2) * abs(r3 * best_pos[d] - positions[i, d])
                positions[i] = np.clip(positions[i], lb, ub)
                score = func(positions[i])
                if score < best_score:
                    best_score = score
                    best_pos = positions[i].copy()
            convergence[iter] = best_score
        return convergence

class ALO(Metaheuristic):
    def __init__(self):
        super().__init__("ALO")
    def optimize(self, func, dim, lb, ub, max_iter, pop_size):
        positions = np.random.uniform(lb, ub, (pop_size, dim))
        antlions = positions.copy()
        best_score = float('inf')
        best_pos = positions[0].copy()
        convergence = np.zeros(max_iter)
        for i in range(pop_size):
            score = func(positions[i])
            if score < best_score:
                best_score = score
                best_pos = positions[i].copy()
        def random_walk(dim, max_iter):
            walk = np.zeros((max_iter, dim))
            for d in range(dim):
                s = np.random.choice([-1, 1], size=max_iter)
                walk[:, d] = np.cumsum(s)
            min_walk = walk.min(axis=0)
            max_walk = walk.max(axis=0)
            walk = (walk - min_walk) / (max_walk - min_walk + 1e-12)
            return walk
        for iter in range(max_iter):
            fitness = np.array([func(antlions[i]) for i in range(pop_size)])
            sorted_idx = np.argsort(fitness)
            elite_antlion = antlions[sorted_idx[0]].copy()
            I = 1 + 4 * iter / max_iter
            lb_iter = lb / I
            ub_iter = ub / I
            for i in range(pop_size):
                probs = 1.0 / (fitness + 1e-12)
                probs /= probs.sum()
                selected_idx = np.random.choice(pop_size, p=probs)
                selected_antlion = antlions[selected_idx]
                RW_selected = random_walk(dim, max_iter)
                RW_elite = random_walk(dim, max_iter)
                pos_selected = lb_iter + RW_selected[iter] * (ub_iter - lb_iter)
                pos_elite = lb_iter + RW_elite[iter] * (ub_iter - lb_iter)
                positions[i] = (pos_selected + pos_elite) / 2.0
                positions[i] = np.clip(positions[i], lb, ub)
                score = func(positions[i])
                if score < best_score:
                    best_score = score
                    best_pos = positions[i].copy()
            for i in range(pop_size):
                if func(positions[i]) < func(antlions[i]):
                    antlions[i] = positions[i].copy()
            convergence[iter] = best_score
        return convergence

class AO(Metaheuristic):
    def __init__(self):
        super().__init__("AO", alpha=0.1, delta=0.1, beta=1.8)
    def optimize(self, func, dim, lb, ub, max_iter, pop_size):
        positions = np.random.uniform(lb, ub, (pop_size, dim))
        best_pos = positions[0].copy()
        best_score = float('inf')
        convergence = np.zeros(max_iter)
        for i in range(pop_size):
            score = func(positions[i])
            if score < best_score:
                best_score = score
                best_pos = positions[i].copy()
        for iter in range(max_iter):
            a = 2 * (1 - iter/max_iter)
            fitness_values = np.array([func(pos) for pos in positions])
            weights = 1 / (fitness_values + np.finfo(float).eps)
            weighted_mean = np.sum(positions * weights[:, np.newaxis], axis=0) / np.sum(weights)
            for i in range(pop_size):
                if iter <= (2/3)*max_iter:
                    if np.random.rand() < 0.5:
                        positions[i] = best_pos * (1 - iter/max_iter) + (weighted_mean - best_pos) * np.random.rand()
                    else:
                        r = np.random.rand()
                        theta = np.random.rand() * 2 * np.pi
                        x = r * np.sin(theta)
                        y = r * np.cos(theta)
                        beta = self.params['beta']
                        sigma = (np.math.gamma(1+beta)*np.sin(np.pi*beta/2)/(np.math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
                        u = np.random.randn(dim) * sigma
                        v = np.random.randn(dim)
                        levy = 0.01 * u / np.abs(v)**(1/beta)
                        positions[i] = best_pos * levy + positions[np.random.randint(pop_size)] + (y - x) * np.random.rand()
                else:
                    if np.random.rand() < 0.5:
                        positions[i] = (best_pos * weighted_mean) * self.params['alpha'] - np.random.rand() + ((ub - lb) * np.random.rand() + lb) * self.params['delta']
                    else:
                        QF = iter**((2*np.random.rand()-1)/(1 - max_iter)**2)
                        G1 = 2 * np.random.rand() - 1
                        G2 = 2 * (1 - iter/max_iter)
                        levy = np.random.rand()
                        positions[i] = QF * best_pos - G1 * positions[i] * np.random.rand() - G2 * levy + np.random.rand()
                positions[i] = np.clip(positions[i], lb, ub)
                score = func(positions[i])
                if score < best_score:
                    best_score = score
                    best_pos = positions[i].copy()
            convergence[iter] = best_score
        return convergence

class MPA(Metaheuristic):
    def __init__(self):
        super().__init__("MPA")
    def optimize(self, func, dim, lb, ub, max_iter, pop_size):
        positions = np.random.uniform(lb, ub, (pop_size, dim))
        best_pos = positions[0].copy()
        best_score = float('inf')
        convergence = np.zeros(max_iter)
        def levy(dim):
            beta = 1.5
            sigma = (np.math.gamma(1+beta)*np.sin(np.pi*beta/2)/(np.math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
            u = np.random.randn(dim) * sigma
            v = np.random.randn(dim)
            return u / (np.abs(v)**(1/beta))
        for i in range(pop_size):
            score = func(positions[i])
            if score < best_score:
                best_score = score
                best_pos = positions[i].copy()
        for iter in range(max_iter):
            Elite = best_pos.copy()
            FADs = 0.2
            for i in range(pop_size):
                if iter < max_iter/3:
                    RB = np.random.randn(dim)
                    positions[i] = positions[i] + np.random.rand(dim)*(Elite - np.random.rand(dim)*positions[i]) + RB*np.random.rand(dim)
                elif iter < 2*max_iter/3:
                    if i < pop_size/2:
                        RB = np.random.randn(dim)
                        positions[i] = Elite + np.random.rand(dim)*np.abs(Elite - positions[i]) + RB*np.random.rand(dim)
                    else:
                        LF = levy(dim)
                        positions[i] = positions[i] + LF*np.random.rand(dim)*(Elite - np.random.rand(dim)*positions[i])
                else:
                    LF = levy(dim)
                    positions[i] = Elite + LF*np.random.rand(dim)*np.abs(Elite - positions[i])
                if np.random.rand() < FADs:
                    positions[i] = positions[i] + FADs*(lb + np.random.rand(dim)*(ub-lb))
                positions[i] = np.clip(positions[i], lb, ub)
                score = func(positions[i])
                if score < best_score:
                    best_score = score
                    best_pos = positions[i].copy()
            convergence[iter] = best_score
        return convergence

# First install required package
# pip install opfunu
import numpy as np
import matplotlib.pyplot as plt
from opfunu.cec_based import cec2014, cec2017, cec2020, cec2022



# ======================
# CEC Function Loader
# ======================

# ======================
# CEC Function Loader (Updated)
# ======================

def get_cec_functions(dim):
    """Get all CEC functions for different years using official Opfunu API"""
    funcs = []
    
    # CEC 2014 (F1-F30)
    for fid in range(1, 31):
        func_class = getattr(cec2014, f"F{fid}2014")
        funcs.append(func_class(ndim=dim))
    
    # CEC 2017 (F1-F30)
    for fid in range(1, 10):
        func_class = getattr(cec2017, f"F{fid}2017")
        funcs.append(func_class(ndim=dim))
    
    # CEC 2020 (F1-F10)
    for fid in range(1, 11):
        func_class = getattr(cec2020, f"F{fid}2020")
        funcs.append(func_class(ndim=dim))
    
    # CEC 2022 (F1-F12)
    for fid in range(1, 13):
        func_class = getattr(cec2022, f"F{fid}2022")
        funcs.append(func_class(ndim=dim))
    
    return funcs

# ======================
# Modified Experimental Setup
# ======================
import matplotlib.pyplot as plt
def run_cec_comparison(dim=10, runs=10):
    algorithms = [
        PSO("PSO", w=0.7, c1=1.5, c2=1.5),
        GWO("GWO"),
        GOA(),
        EO(),
        SCA(),
        ALO(),
        MVAO(),
        AO(),
        MPA(),
    ]
    
    # Load all CEC functions
    cec_funcs = get_cec_functions(dim)
    
    for func in cec_funcs:
        print(f"\n=== Testing on {func.name} ===")
        results = {algo.name: [] for algo in algorithms}
        
        for run in range(runs):
            for algo in algorithms:
                convergence = algo.optimize(
                    func=func.evaluate,
                    dim=dim,
                    lb=func.lb[0],
                    ub=func.ub[0],
                    max_iter=500,
                    pop_size=30
                )
                results[algo.name].append(convergence)

        # Plotting inside the loop
        plt.figure(figsize=(12, 8))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']
        markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*']
        
        for idx, algo in enumerate(algorithms):
            avg_curve = np.mean(results[algo.name], axis=0)
            plt.semilogy(avg_curve, 
                        label=algo.name,
                        color=colors[idx % len(colors)],
                        marker=markers[idx % len(markers)],
                        markevery=50,
                        linewidth=2,
                        alpha=0.8)
        
        plt.title(f'Algorithm Comparison on {func.name} Function', fontsize=14, pad=20)
        plt.xlabel('Iteration', fontsize=12, labelpad=10)
        plt.ylabel('Best Fitness (log scale)', fontsize=12, labelpad=10)
        plt.grid(True, which='both', linestyle='--', alpha=0.6)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.xlim(left=0)
        plt.xticks(np.arange(0, 501, 50))
        plt.ylim(bottom=1e-10)
        plt.tight_layout()
        plt.savefig(f'{func.name}_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()




# def plot_results(results, func_name):
#     plt.figure(figsize=(12, 8))
#     colors = plt.cm.tab20(np.linspace(0, 1, len(results)))
    
#     for (algo_name, curves), color in zip(results.items(), colors):
#         avg_curve = np.mean(curves, axis=0)
#         plt.semilogy(avg_curve, label=algo_name, color=color, linewidth=2)
    
#     plt.title(f'Algorithm Comparison on {func_name}', fontsize=14)
#     plt.xlabel('Iteration')
#     plt.ylabel('Fitness (log scale)')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig(f'{func_name}_comparison.png', dpi=300)
#     plt.close()
#     plt.show()

if __name__ == "__main__":
    run_cec_comparison(dim=10, runs=3)


