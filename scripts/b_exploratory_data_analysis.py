import matplotlib.pyplot as plt
import pandas as pd

# --- 1. Histogram of Energy per Token (kWh) ---
def plot_histogram(data, column, bins=90):
    plt.figure(figsize=(10, 6))
    plt.hist(data[column].dropna(), bins=bins, edgecolor='black')
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

# --- 2. Boxplot for Energy per Token (kWh) to Visualize Outliers ---
def plot_boxplot(data, column):
    plt.figure(figsize=(10, 2))
    plt.boxplot(data[column].dropna(), vert=False)
    plt.title(f'Boxplot of {column}')
    plt.xlabel(column)
    plt.grid(True)
    plt.show()
    
# --- 3. Scatter Plot: Energy per Token (kWh) vs. FLOPs per Token ---
def plot_scatter(data, x_column, y_column):
    plt.figure(figsize=(10, 6))
    plt.scatter(data[x_column], data[y_column], alpha=0.6)
    plt.title(f'{y_column} vs. {x_column}')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.grid(True)
    plt.show()
    
# --- 4. Scatter Plot: Divergence Energy (Energy per Token / FLOPs per Token) vs Batch Size ---
# Attempt to convert batch_size___fixed_batching to numeric values.

def plot_divergence(data, x_column, y_column):
    plt.figure(figsize=(10, 6))
    plt.scatter(data[x_column], data[y_column], alpha=0.6)
    plt.title(f'{y_column} vs. {x_column}')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.grid(True)
    plt.show()
    
# --- 5. Correlation Matrix for Key Metrics ---
# Select key columns for correlation. Adjust the list if you add more metrics.
def plot_correlation_matrix(data, columns):
    plt.figure(figsize=(10, 6))
    corr = data[columns].corr()
    plt.matshow(corr, cmap='coolwarm', fignum=1)
    plt.xticks(range(len(columns)), columns, rotation=45)
    plt.yticks(range(len(columns)), columns)
    plt.colorbar()
    plt.title('Correlation Matrix')
    plt.show()

if __name__ == "__main__":

    from scripts.a_data_loading_cleaning import run_load_clean_diagnose_data

    csv_path = "results/controlled_results.csv"
    df = run_load_clean_diagnose_data(csv_path)

    # Plot Histogram
    plot_histogram(df, 'energy_per_token_kwh')
    
    # Plot Boxplot
    plot_boxplot(df, 'energy_per_token_kwh')
    
    # Plot Scatter: Energy per Token vs FLOPs per Token
    plot_scatter(df, 'flops_per_token', 'energy_per_token_kwh')
    
    # Plot Divergence: Energy vs Batch Size
    #df_controlled_cleaned['batch_size_numeric'] = pd.to_numeric(df_controlled_cleaned['batch_size___fixed_batching'], errors='coerce')
    plot_divergence(df, 'batch_size___fixed_batching', 'divergence_energy_flops')
    
    # Plot Correlation Matrix
    columns_to_correlate = ['energy_per_token_kwh', 'flops_per_token', 'divergence_energy_flops']
    plot_correlation_matrix(df, columns_to_correlate)
