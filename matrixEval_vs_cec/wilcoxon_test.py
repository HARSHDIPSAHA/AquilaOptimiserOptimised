import pandas as pd 
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from itertools import combinations

results = pd.read_csv(r"H:\academics\SEM4\ai\project\gitRepo\AquilaOptimiserOptimised\matrixEval_vs_cec\ao_results.csv")
stats = pd.read_csv(r"H:\academics\SEM4\ai\project\gitRepo\AquilaOptimiserOptimised\matrixEval_vs_cec\ao_statistics.csv", index_col=0)

results = results.drop(columns=['Unnamed: 0'], errors='ignore')

functions = results.columns.tolist()
wilcoxon_results = []

for func1, func2 in combinations(functions, 2):
    try:
        stat, p = wilcoxon(results[func1], results[func2])
        is_significant = "Yes" if p < 0.05 else "No"
        wilcoxon_results.append((func1, func2, stat, p, is_significant))
    except ValueError as e:
        wilcoxon_results.append((func1, func2, None, f'Error: {str(e)}', "No"))

wilcoxon_df = pd.DataFrame(wilcoxon_results, columns=["Function A", "Function B", "Statistic", "p-value", "IsSignificant"])

output_path = r"H:\academics\SEM4\ai\project\gitRepo\AquilaOptimiserOptimised\matrixEval_vs_cec\wilcoxon_results_all_pairs.csv"
wilcoxon_df.to_csv(output_path, index=False)

print("Wilcoxon test completed for all pairs.")
print(wilcoxon_df.head())
