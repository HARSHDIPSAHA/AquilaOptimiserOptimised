import numpy as np
import pandas as pd
file_14 = r"H:\academics\SEM4\ai\project\gitRepo\AquilaOptimiserOptimised\matrixEval_vs_cec\14.csv"
file_17 = r"H:\academics\SEM4\ai\project\gitRepo\AquilaOptimiserOptimised\matrixEval_vs_cec\17.csv"
df_14 = pd.read_csv(file_14)
df_17 = pd.read_csv(file_17)

# Helper function to extract mean from "Mean ± STD" strings
def extract_mean(value):
    try:
        return float(value.split('±')[0].strip())
    except:
        return np.nan

# Apply extraction to all relevant columns in df_14
df_means_14 = df_14.copy()
algo_columns = df_14.columns[1:]

for col in algo_columns:
    df_means_14[col] = df_14[col].apply(extract_mean)

# Calculate rankings (lower mean is better, so use rank with method='min')
df_means_14["Ranking"] = df_means_14[algo_columns].rank(axis=1, method='min')[ "Modified Aquila (Mean ± STD)"]

# Merge the new ranking into the original df
df_14["Ranking"] = df_means_14["Ranking"]
df_means_17 = df_17.copy()
algo_columns_17 = df_17.columns[1:]

for col in algo_columns_17:
    df_means_17[col] = df_17[col].apply(extract_mean)

# Calculate rankings (lower mean is better, so use rank with method='min')
df_means_17["Ranking"] = df_means_17[algo_columns_17].rank(axis=1, method='min')["Modified Aquila (Mean ± STD)"]

# Merge the new ranking into the original df
df_17["Ranking"] = df_means_17["Ranking"]


print(df_17)
