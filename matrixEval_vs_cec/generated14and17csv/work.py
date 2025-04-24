import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from opfunu.cec_based import cec2014, cec2017, cec2020, cec2022
from mealpy import BBO, EO, GWO, SCA, ALO, PSO, MPA
from mealpy.utils.problem import Problem
from mealpy.utils.space import FloatVar
import logging

logging.basicConfig(level=logging.INFO)

class CECProblem(Problem):
    def __init__(self, bounds, obj_func, **kwargs):
        super().__init__(bounds, n_objs=1, obj_weights=[-1], **kwargs)
        self.obj_func = obj_func

    def fit_func(self, solution):
        return [self.obj_func(solution)]

def run_algorithm(algorithm_class, problem, max_evals=10000, n_runs=3):
    results = []
    for _ in range(n_runs):
        try:
            model = algorithm_class(epoch=1000, pop_size=10)
            g_best = model.solve(problem)
            results.append(g_best.target.fitness)
        except Exception as e:
            logging.error(f"Error running {algorithm_class.__name__}: {str(e)}")
            results.append(np.nan)
    return np.nanmean(results), np.nanstd(results)

def generate_comparison_table(dim=10, n_runs=3):
    algorithms = [
        ('AO', BBO.OriginalBBO),
        ('GWO', GWO.OriginalGWO),
        ('PSO', PSO.OriginalPSO),
        ('MPA', MPA.OriginalMPA),
        ('SCA', SCA.OriginalSCA)
    ]
    
    results = []
    cec_modules = {
        2014: (cec2014, 30),
        2017: (cec2017, 30),
        2020: (cec2020, 12),
        2022: (cec2022, 12)
    }

    for year, (cec_module, num_funcs) in cec_modules.items():
        for func_num in range(1, num_funcs + 1):
            func_name = f"F{func_num}{year}"
            try:
                func_class = getattr(cec_module, f"F{func_num}{year}", None)
                if not func_class:
                    continue
                
                func = func_class(ndim=dim)
                lb = np.array(func.lb)
                ub = np.array(func.ub)
                
                # Convert bounds to Mealpy format
                bounds = {}
                for i in range(dim):
                    bounds[f"x{i+1}"] = FloatVar(lb[i], ub[i])
                
                problem = CECProblem(
                    bounds=bounds,
                    obj_func=func.evaluate,
                    name=func_name,
                    minmax="min"
                )

                algo_results = {'Function': func_name}
                for algo_name, algo_class in algorithms:
                    mean, std = run_algorithm(algo_class, problem, n_runs=n_runs)
                    algo_results[f'{algo_name} (Mean)'] = mean
                    algo_results[f'{alao_name} (STD)'] = std
                
                results.append(algo_results)
                
            except Exception as e:
                logging.warning(f"Skipping {func_name}: {str(e)}")
                continue

    # Create DataFrame and save
    df = pd.DataFrame(results)
    df.to_csv("optimization_comparison.csv", index=False)
    
    # Generate comparison plot
    plt.figure(figsize=(15, 8))
    for algo_name, _ in algorithms:
        means = df[f'{algo_name} (Mean)'].dropna()
        plt.plot(means, label=algo_name, marker='o')
    
    plt.title('Algorithm Performance Comparison on CEC Benchmarks')
    plt.xlabel('Function Number')
    plt.ylabel('Mean Fitness (log scale)')
    plt.yscale('log')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('algorithm_comparison.png')
    plt.show()

    return df

if __name__ == "__main__":
    df = generate_comparison_table(dim=10, n_runs=3)
    print("Results saved to optimization_comparison.csv")
    print("Comparison plot saved to algorithm_comparison.png")
