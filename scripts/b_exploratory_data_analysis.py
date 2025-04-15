import matplotlib.pyplot as plt
import pandas as pd

from scripts.a_data_loading_cleaning import (
    df_controlled_cleaned,
)

# --- 1. Histogram of Energy per Token (kWh) ---
plt.figure(figsize=(10, 6))
plt.hist(df_controlled_cleaned['energy_per_token_kwh'].dropna(), bins=90, edgecolor='black')
plt.title('Histogram of Energy per Token (kWh)')
plt.xlabel('Energy per Token (kWh)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# --- 2. Boxplot for Energy per Token (kWh) to Visualize Outliers ---
plt.figure(figsize=(10, 2))
plt.boxplot(df_controlled_cleaned['energy_per_token_kwh'].dropna(), vert=False)
plt.title('Boxplot of Energy per Token (kWh)')
plt.xlabel('Energy per Token (kWh)')
plt.grid(True)
plt.show()

# --- 3. Scatter Plot: Energy per Token (kWh) vs. FLOPs per Token ---
plt.figure(figsize=(10, 6))
plt.scatter(df_controlled_cleaned['flops_per_token'], df_controlled_cleaned['energy_per_token_kwh'], alpha=0.6)
plt.title('Energy per Token (kWh) vs. FLOPs per Token')
plt.xlabel('FLOPs per Token')
plt.ylabel('Energy per Token (kWh)')
plt.grid(True)
plt.show()

# --- 4. Scatter Plot: Divergence Energy (Energy per Token / FLOPs per Token) vs Batch Size ---
# Attempt to convert batch_size___fixed_batching to numeric values.
df_controlled_cleaned['batch_size_numeric'] = pd.to_numeric(df_controlled_cleaned['batch_size___fixed_batching'], errors='coerce')

plt.figure(figsize=(10, 6))
plt.scatter(df_controlled_cleaned['batch_size_numeric'], df_controlled_cleaned['divergence_energy_flops'], alpha=0.6)
plt.title('Divergence Energy vs. Batch Size (Fixed Batching)')
plt.xlabel('Batch Size (Fixed Batching)')
plt.ylabel('Divergence (Energy per Token / FLOPs per Token)')
plt.grid(True)
plt.show()

# --- 5. Correlation Matrix for Key Metrics ---
# Select key columns for correlation. Adjust the list if you add more metrics.
cols_for_corr = ['flops', 'total_generated_tokens', 'flops_per_token', 'energy_per_token_kwh', 'divergence_energy_flops']
corr_matrix = df_controlled_cleaned[cols_for_corr].corr()
print("Correlation Matrix:")
print(corr_matrix)
