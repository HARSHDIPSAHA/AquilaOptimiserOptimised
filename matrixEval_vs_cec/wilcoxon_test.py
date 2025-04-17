import pandas as pd
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

results_df = pd.read_csv(r"H:\academics\SEM4\ai\project\gitRepo\AquilaOptimiserOptimised\matrixEval_vs_cec\ao_results.csv", index_col=0)
stats_df = pd.read_csv(r"H:\academics\SEM4\ai\project\gitRepo\AquilaOptimiserOptimised\matrixEval_vs_cec\ao_statistics.csv", index_col=0)

def categorize_cec_function(func_name):
    """Classify functions into CEC suites"""
    if '2014' in func_name: return 'CEC2014'
    if '2017' in func_name: return 'CEC2017'
    if '2020' in func_name: return 'CEC2020'
    if '2022' in func_name: return 'CEC2022'
    return 'Other'

# Preprocess data
cec_analysis = pd.DataFrame({
    'Function': stats_df.index,
    'Suite': [categorize_cec_function(f) for f in stats_df.index],
    'Mean': stats_df['Mean'],
    'Std': stats_df['Std'],
    'Min': stats_df['Min'],
    'Median': stats_df['Median'],
    'Max': stats_df['Max']
})

# 1. Statistical Analysis
def perform_statistical_analysis():
    """Calculate suite-wise statistics and Wilcoxon tests"""
    suite_stats = cec_analysis.groupby('Suite').agg({
        'Mean': ['mean', 'std'],
        'Std': 'mean',
        'Median': 'mean'
    }).reset_index()

    # Wilcoxon signed-rank test between suites
    p_values = {}
    suites = ['CEC2014', 'CEC2017', 'CEC2020', 'CEC2022']
    for i in range(len(suites)):
        for j in range(i+1, len(suites)):
            s1 = cec_analysis[cec_analysis.Suite == suites[i]]['Mean']
            s2 = cec_analysis[cec_analysis.Suite == suites[j]]['Mean']
            stat, p = wilcoxon(s1, s2)
            p_values[f"{suites[i]} vs {suites[j]}"] = p
    
    return suite_stats, pd.DataFrame(p_values.items(), columns=['Comparison', 'p-value'])

suite_stats, wilcoxon_results = perform_statistical_analysis()

# 2. Visualization
plt.figure(figsize=(15, 10))

# Boxplot of function performance by suite
plt.subplot(2,2,1)
sns.boxplot(x='Suite', y='Mean', data=cec_analysis)
plt.title('Distribution of Mean Fitness Across CEC Suites')
plt.yscale('log')

# Convergence plot for each suite
plt.subplot(2,2,2)
for suite in cec_analysis.Suite.unique():
    if suite == 'Other': continue
    suite_means = results_df[cec_analysis[cec_analysis.Suite == suite]['Function']].mean(axis=1)
    plt.plot(suite_means, label=suite)
plt.xlabel('Run #')
plt.ylabel('Average Fitness')
plt.title('Convergence Across CEC Suites')
plt.legend()
plt.yscale('log')

# Performance comparison table
plt.subplot(2,2,3)
cell_text = suite_stats.round(2).values
plt.table(cellText=cell_text,
          colLabels=suite_stats.columns,
          loc='center',
          cellLoc='center')
plt.axis('off')
plt.title('Suite-wise Statistical Summary')

# Significant comparisons
plt.subplot(2,2,4)
plt.table(cellText=wilcoxon_results.round(4).values,
          colLabels=wilcoxon_results.columns,
          loc='center',
          cellLoc='center')
plt.axis('off')
plt.title('Wilcoxon Signed-Rank Test Results')

plt.tight_layout()
plt.savefig('cec_analysis_report.png', dpi=300)
plt.show()

# 3. Detailed Function Analysis
def analyze_top_functions(n=5):
    """Identify best/worst performing functions"""
    best_functions = cec_analysis.nsmallest(n, 'Mean')
    worst_functions = cec_analysis.nlargest(n, 'Mean')
    
    print(f"\nTop {n} Best Performing Functions:")
    print(best_functions)
    
    print(f"\nTop {n} Worst Performing Functions:")
    print(worst_functions)

analyze_top_functions()

# 4. Save Analysis Report
report = f"""
CEC Benchmark Analysis Report
=============================
Total Functions Analyzed: {len(cec_analysis)}
Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

Key Findings:
1. {suite_stats.iloc[0]['Suite']} showed the lowest average mean fitness ({suite_stats.iloc[0]['Mean']['mean']:.2e})
2. {suite_stats.iloc[-1]['Suite']} had the highest variance (std: {suite_stats.iloc[-1]['Mean']['std']:.2e})
3. Most significant difference: {wilcoxon_results.loc[wilcoxon_results['p-value'].idxmin(), 'Comparison']} (p={wilcoxon_results['p-value'].min():.4e})
"""

with open('cec_analysis_report.txt', 'w') as f:
    f.write(report)