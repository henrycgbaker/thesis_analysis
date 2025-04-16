import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ---------------------------
# 1. Histogram
# ---------------------------
def plot_histogram(data, column='energy_per_token_kwh', bins=90):
    if column not in data:
        print(f"Column '{column}' not found in DataFrame.")
        return
    plt.figure(figsize=(10, 6))
    plt.hist(data[column].dropna(), bins=bins, edgecolor='black')
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

# ---------------------------
# 2. Boxplot
# ---------------------------
def plot_boxplot(data, column='energy_per_token_kwh'):
    if column not in data:
        print(f"Column '{column}' not found in DataFrame.")
        return
    plt.figure(figsize=(10, 2))
    plt.boxplot(data[column].dropna(), vert=False)
    plt.title(f'Boxplot of {column}')
    plt.xlabel(column)
    plt.grid(True)
    plt.show()

# ---------------------------
# 3. Scatter Plot
# ---------------------------
def plot_scatter(data, x_column='flops_per_token', y_column='energy_per_token_kwh'):
    if x_column not in data or y_column not in data:
        print(f"Columns '{x_column}' or '{y_column}' not found in DataFrame.")
        return
    plt.figure(figsize=(10, 6))
    plt.scatter(data[x_column], data[y_column], alpha=0.6)
    plt.title(f'{y_column} vs. {x_column}')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.grid(True)
    plt.show()

# ---------------------------
# 4. Scatter: Divergence vs. Batch Size
# ---------------------------
def plot_divergence(data, x_column='batch_size___fixed_batching', y_column='divergence_energy_flops'):
    if x_column not in data or y_column not in data:
        print(f"Columns '{x_column}' or '{y_column}' not found in DataFrame.")
        return
    plt.figure(figsize=(10, 6))
    plt.scatter(data[x_column], data[y_column], alpha=0.6)
    plt.title(f'{y_column} vs. {x_column}')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.grid(True)
    plt.show()

# ---------------------------
# 5. Correlation Matrix
# ---------------------------
def plot_correlation_matrix(data, columns=None):
    if columns is None:
        columns = ['energy_per_token_kwh', 'flops_per_token', 'divergence_energy_flops']
    
    missing = [col for col in columns if col not in data]
    if missing:
        print(f"Missing columns for correlation matrix: {missing}")
        return
    
    corr = data[columns].corr()
    plt.figure(figsize=(8, 6))
    plt.matshow(corr, cmap='coolwarm', fignum=1)
    plt.xticks(range(len(columns)), columns, rotation=45)
    plt.yticks(range(len(columns)), columns)
    plt.colorbar()
    plt.title('Correlation Matrix', pad=20)
    plt.show()

# ---------------------------
# Plot All Diagnostics
# ---------------------------
def plot_all_diagnostics(df):
    """
    Runs the full diagnostics suite on a cleaned dataset.
    """
    print("📊 Plotting histogram...")
    plot_histogram(df)

    print("📦 Plotting boxplot...")
    plot_boxplot(df)

    print("🔬 Scatter: Energy vs FLOPs...")
    plot_scatter(df)

    print("📈 Scatter: Divergence vs Batch Size...")
    plot_divergence(df)

    print("🔗 Correlation matrix...")
    plot_correlation_matrix(df)

# ---------------------------
# Main Entry Point
# ---------------------------

if __name__ == "__main__":

    from scripts.a_data_loading_cleaning import run_load_clean_diagnose_data

    csv_path = "results/controlled_results.csv"
    df = run_load_clean_diagnose_data(csv_path)

    plot_histogram(df, 'energy_per_token_kwh')
    plot_boxplot(df, 'energy_per_token_kwh')
    plot_scatter(df, 'flops_per_token', 'energy_per_token_kwh')
    plot_divergence(df, 'batch_size___fixed_batching', 'divergence_energy_flops')
    columns_to_correlate = ['energy_per_token_kwh', 'flops_per_token', 'divergence_energy_flops']
    plot_correlation_matrix(df, columns_to_correlate)
