import pandas as pd
from scipy.stats import wilcoxon

ao_df = pd.read_csv("ao_results.csv", index_col=0)
o_df = pd.read_csv("ao_statistics.csv", index_col=0)

assert ao_df.shape == o_df.shape, "The two dataframes must have the same shape!"
assert list(ao_df.index) == list(o_df.index), "The row indices (e.g., function names) must match!"

results = []

for func in ao_df.index:
    stat, p_value = wilcoxon(ao_df.loc[func], o_df.loc[func])
    results.append({
        "Function": func,
        "Wilcoxon_Statistic": stat,
        "P_Value": p_value,
        "Significant": "Yes" if p_value < 0.05 else "No"
    })

wilcoxon_results_df = pd.DataFrame(results)

wilcoxon_results_df.to_csv("wilcoxon_significance_results.csv", index=False)

print(wilcoxon_results_df)
