#!/usr/bin/env python
"""
Dynamic Plotting Script for Performance Metrics

This script defines a series of helper functions that dynamically generate plots
for various configurations:
  - Number of Processes
  - Batching (Batch Size)
  - Precision
  - Decoding (Temperature, top_k / top_p)
  - Latency

Each function adjusts automatically to the underlying data (e.g. dynamic x-ticks,
groupings, and color assignments) so that changes in the configuration values are
handled without code modifications.

Usage:
  • In your notebook, import these functions and call them individually, for example:
      plot_num_processes(dfs)
  • Or, run the script directly to generate all figures using the plot_all(dfs) function.
  
Ensure you have your master DataFrame “df” loaded and then create the dfs dictionary as above.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings


# ---------------------------
# Plot for Number of Processes
# ---------------------------
def plot_num_processes(dfs):
    """
    Plots Energy- and FLOPs-per-Token vs Number of Processes (from dfs['num_processes']).
    """
    if 'num_processes' not in dfs:
        print("num_processes DataFrame not found in dfs.")
        return
    
    num_proc_df = dfs['num_processes'].copy()
    num_proc_df['num_processes'] = num_proc_df['num_processes'].astype(int)

    fig, ax1 = plt.subplots(figsize=(8, 6))
    color_energy = 'tab:blue'
    ax1.set_xlabel('Number of Processes')
    ax1.set_ylabel('Energy-per-Token (kWh)', color=color_energy)
    ax1.plot(num_proc_df['num_processes'], num_proc_df['energy_per_token_kwh'],
             marker='o', linestyle='-', color=color_energy, label='Energy-per-Token (kWh)')
    ax1.set_ylim(bottom=0)
    ax1.tick_params(axis='y', labelcolor=color_energy)
    ax1.set_xticks(sorted(num_proc_df['num_processes'].unique()))

    ax2 = ax1.twinx()
    color_flops = 'tab:red'
    ax2.set_ylabel('FLOPs-per-Token', color=color_flops)
    ax2.plot(num_proc_df['num_processes'], num_proc_df['flops_per_token'],
             marker='s', linestyle='--', color=color_flops, label='FLOPs-per-Token')
    ax2.tick_params(axis='y', labelcolor=color_flops)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

    plt.title('Energy- & FLOPs-per-Token vs Number of Processes')
    fig.tight_layout()
    plt.show()


# ---------------------------
# Plot for Batching (Batch Size)
# ---------------------------
def plot_batching(dfs):
    """
    Plots Energy- and FLOPs-per-Token vs Batch Size (from dfs['batching']).
    Dynamically sets the x-ticks based on the minimum and maximum batch sizes.
    """
    if 'batching' not in dfs:
        print("batching DataFrame not found in dfs.")
        return

    batching_df = dfs['batching'].copy()
    batching_df['batch_size___fixed_batching'] = batching_df['batch_size___fixed_batching'].astype(int)

    fig, ax1 = plt.subplots(figsize=(8, 6))
    color_energy = 'tab:blue'
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Energy-per-Token (kWh)', color=color_energy)
    ax1.plot(batching_df['batch_size___fixed_batching'], batching_df['energy_per_token_kwh'],
             marker='o', linestyle='-', color=color_energy, label='Energy-per-Token (kWh)')
    ax1.set_ylim(bottom=0)
    ax1.tick_params(axis='y', labelcolor=color_energy)
    min_bs = batching_df['batch_size___fixed_batching'].min()
    max_bs = batching_df['batch_size___fixed_batching'].max()
    ax1.set_xticks(np.arange(min_bs, max_bs + 1, 8))

    ax2 = ax1.twinx()
    color_flops = 'tab:red'
    ax2.set_ylabel('FLOPs-per-Token', color=color_flops)
    ax2.plot(batching_df['batch_size___fixed_batching'], batching_df['flops_per_token'],
             marker='s', linestyle='--', color=color_flops, label='FLOPs-per-Token')
    ax2.tick_params(axis='y', labelcolor=color_flops)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

    plt.title('Energy- & FLOPs-per-Token vs Batch Size')
    fig.tight_layout()
    plt.show()


# ---------------------------
# Plot for Precision
# ---------------------------
def plot_precision(dfs):
    """
    Plots Energy- and FLOPs-per-Token vs Precision (from dfs['precis']).
    Dynamically categorizes precision into FP32, FP16, INT8, and INT4.
    """
    if 'precis' not in dfs:
        print("precis DataFrame not found in dfs.")
        return
    
    precision_df = dfs['precis'].copy()

    def determine_precision(row):
        if row.get('load_in_4bit', False):
            return 'INT4'
        elif row.get('load_in_8bit', False):
            return 'INT8'
        elif row.get('fp_precision') == 'torch.float16':
            return 'FP16'
        else:
            return 'FP32'

    precision_df['precision_mode'] = precision_df.apply(determine_precision, axis=1)
    precision_order = ['FP32', 'FP16', 'INT8', 'INT4']
    precision_df['precision_mode'] = pd.Categorical(precision_df['precision_mode'], categories=precision_order, ordered=True)
    precision_df = precision_df.sort_values('precision_mode')

    fig, ax1 = plt.subplots(figsize=(8, 6))
    color_energy = 'tab:blue'
    ax1.set_xlabel('Precision')
    ax1.set_ylabel('Energy-per-Token (kWh)', color=color_energy)
    ax1.plot(precision_df['precision_mode'], precision_df['energy_per_token_kwh'],
             marker='o', linestyle='-', color=color_energy, label='Energy-per-Token (kWh)')
    ax1.set_ylim(bottom=0)
    ax1.tick_params(axis='y', labelcolor=color_energy)
    ax1.grid(True)

    ax2 = ax1.twinx()
    color_flops = 'tab:red'
    ax2.set_ylabel('FLOPs-per-Token', color=color_flops)
    ax2.plot(precision_df['precision_mode'], precision_df['flops_per_token'],
             marker='s', linestyle='--', color=color_flops, label='FLOPs-per-Token')
    ax2.tick_params(axis='y', labelcolor=color_flops)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

    plt.title('Energy- and FLOPs-per-Token vs Precision')
    fig.tight_layout()
    plt.show()


# ---------------------------
# Plot for Decoder Temperature (Grouping by Method and Config Mode)
# ---------------------------
def plot_decoder_temperature(dfs):
    """
    Plots Energy- and FLOPs-per-Token vs Decoder Temperature (from dfs['decoding']).
    Groups the data by decoding method ('greedy', 'top_k', 'top_p') and their config parameters.
    """
    if 'decoding' not in dfs:
        print("decoding DataFrame not found in dfs.")
        return

    decoder_df = dfs['decoding'].copy()
    decoder_df = decoder_df[decoder_df['decoder_config_decoding_mode'].notna()].copy()
    decoder_df['method'] = decoder_df['decoder_config_decoding_mode']
    decoder_df['temperature'] = decoder_df['decoder_temperature']

    def get_config_mode(row):
        if row['method'] == 'top_k':
            return row['decoder_top_k']
        elif row['method'] == 'top_p':
            return row['decoder_top_p']
        else:
            return 'greedy'

    decoder_df['config_mode'] = decoder_df.apply(get_config_mode, axis=1)
    decoder_df = decoder_df[decoder_df['method'].isin(['greedy', 'top_k', 'top_p'])].copy()
    groups = decoder_df.groupby(['method', 'config_mode'])

    colors = {'greedy': 'blue', 'top_k': 'green', 'top_p': 'red'}

    fig, ax1 = plt.subplots(figsize=(14, 8))
    ax2 = ax1.twinx()

    for (method, config_mode), subdf in groups:
        label_str = f"{method} ({config_mode})"
        ax1.plot(subdf['temperature'], subdf['energy_per_token_kwh'],
                 marker='o', linestyle='-', label="_no_legend", color=colors.get(method, 'black'))
        ax1.set_ylim(bottom=0)
        ax2.plot(subdf['temperature'], subdf['flops_per_token'],
                 marker='s', linestyle='--', label=label_str, color=colors.get(method, 'black'))

    ax1.set_xlabel('Decoder Temperature')
    ax1.set_ylabel('Energy-per-Token (kWh)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax2.set_ylabel('FLOPs-per-Token', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax1.set_title('Energy- and FLOPs-per-Token vs Decoder Temperature')
    ax1.grid(True)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right')

    fig.tight_layout()
    plt.show()


# ---------------------------
# Plot for Decoder top_k and top_p
# ---------------------------
def plot_decoder_topk_top_p(dfs):
    """
    Creates two side-by-side subplots for Energy-per-Token vs Top-k and Top-p values
    using the 'decoding' DataFrame. Colors are determined by decoder temperature.
    """
    if 'decoding' not in dfs:
        print("decoding DataFrame not found in dfs.")
        return

    decoder_df = dfs['decoding'].copy()
    # Separate data for top_k and top_p.
    top_k_df = decoder_df[decoder_df['decoder_config_decoding_mode'] == 'top_k'].copy()
    top_p_df = decoder_df[decoder_df['decoder_config_decoding_mode'] == 'top_p'].copy()

    unique_temps_top_k = sorted(top_k_df['decoder_temperature'].unique())
    unique_temps_top_p = sorted(top_p_df['decoder_temperature'].unique())

    colormap = plt.cm.viridis
    colors_top_k = {temp: colormap(i/len(unique_temps_top_k)) for i, temp in enumerate(unique_temps_top_k)}
    colors_top_p = {temp: colormap(i/len(unique_temps_top_p)) for i, temp in enumerate(unique_temps_top_p)}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot for Top-k.
    for temp in unique_temps_top_k:
        subdf = top_k_df[top_k_df['decoder_temperature'] == temp]
        axes[0].plot(subdf['decoder_top_k'],
                     subdf['energy_per_token_kwh'],
                     marker='o', linestyle='-', label=f"Temp {temp}",
                     color=colors_top_k[temp])
    axes[0].set_xlabel('Top-k Value')
    axes[0].set_ylabel('Energy-per-Token (kWh)')
    axes[0].set_title('Energy-per-Token vs Top-k Value')
    axes[0].grid(True)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles[::-1], labels[::-1], title="Decoder Temperature", loc='best')

    # Plot for Top-p.
    for temp in unique_temps_top_p:
        subdf = top_p_df[top_p_df['decoder_temperature'] == temp]
        axes[1].plot(subdf['decoder_top_p'],
                     subdf['energy_per_token_kwh'],
                     marker='o', linestyle='-', label=f"Temp {temp}",
                     color=colors_top_p[temp])
    axes[1].set_xlabel('Top-p Value')
    axes[1].set_ylabel('Energy-per-Token (kWh)')
    axes[1].set_title('Energy-per-Token vs Top-p Value')
    axes[1].grid(True)
    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].legend(handles[::-1], labels[::-1], title="Decoder Temperature", loc='best')

    fig.tight_layout()
    plt.show()


# ---------------------------
# Plot for Latency (Fixed Order using Categories)
# ---------------------------
def plot_latency(dfs):
    """
    Plots Energy- and FLOPs-per-Token vs Latency Config using a fixed category order.
    """
    if 'latency' not in dfs:
        print("latency DataFrame not found in dfs.")
        return

    latency_df = dfs['latency'].copy()

    def determine_latency_label(row):
        if not row.get('latency_simulation_simulate', False):
            return "No\nsimulation"
        else:
            delay_min = row.get('latency_simulation_delay_min', None)
            delay_max = row.get('latency_simulation_delay_max', None)
            simulate_burst = row.get('latency_simulation_simulate_burst', False)
            if not simulate_burst:
                return f"Sim\n({delay_min}-{delay_max})"
            else:
                burst_size = row.get('latency_simulation_burst_size', None)
                burst_interval = row.get('latency_simulation_burst_interval', None)
                return f"Sim ({delay_min}-{delay_max})\nBurst ({burst_size},{burst_interval})"

    latency_df["latency_label"] = latency_df.apply(determine_latency_label, axis=1).astype(str)
    latency_order = [
        "No\nsimulation",
        "Sim\n(0.05-0.2)",
        "Sim\n(0.2-0.6)",
        "Sim (0.05-0.2)\nBurst (5,4.0)",
        "Sim (0.2-0.6)\nBurst (8,5.0)"
    ]
    latency_df['latency_label'] = pd.Categorical(latency_df['latency_label'],
                                                  categories=latency_order, ordered=True)
    latency_df = latency_df.sort_values('latency_label')

    fig, ax1 = plt.subplots(figsize=(8, 6))
    color_energy = 'tab:blue'
    ax1.set_xlabel('Latency Config')
    ax1.set_ylabel('Energy per Token (kWh)', color=color_energy)
    ax1.plot(latency_df['latency_label'].astype(str), latency_df['energy_per_token_kwh'],
             marker='o', linestyle='-', color=color_energy, label='Energy per Token (kWh)')
    ax1.tick_params(axis='y', labelcolor=color_energy)
    ax1.grid(True)

    ax2 = ax1.twinx()
    color_flops = 'tab:red'
    ax2.set_ylabel('FLOPs per Token', color=color_flops)
    ax2.plot(latency_df['latency_label'].astype(str), latency_df['flops_per_token'],
             marker='s', linestyle='--', color=color_flops, label='FLOPs per Token')
    ax2.tick_params(axis='y', labelcolor=color_flops)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

    plt.title('Energy- and FLOPs-per-Token vs Latency Config')
    fig.tight_layout()
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    plt.show()


# ---------------------------
# Plot for Latency (Dynamic x-axis using Numeric Scale)
# ---------------------------
def plot_latency_dynamic(dfs):
    """
    Plots Energy- and FLOPs-per-Token vs Latency on a dynamic numeric x-axis.
    For simulated rows, the x-axis value is the mean of delay_min and delay_max;
    non-simulation rows are placed at 0. Distinct burstiness classes are plotted as separate lines.
    """
    if 'latency' not in dfs:
        print("latency DataFrame not found in dfs.")
        return

    latency_df = dfs['latency'].copy()

    def compute_latency_category(row):
        if not row.get('latency_simulation_simulate', False):
            return "No simulation"
        else:
            delay_min = row.get('latency_simulation_delay_min', None)
            delay_max = row.get('latency_simulation_delay_max', None)
            return f"Sim ({delay_min}-{delay_max})"

    latency_df['latency_category'] = latency_df.apply(compute_latency_category, axis=1)

    def compute_burstiness(row):
        if not row.get('latency_simulation_simulate', False):
            return "Baseline"
        else:
            if not row.get('latency_simulation_simulate_burst', False):
                return "Non-burst"
            else:
                burst_size = row.get('latency_simulation_burst_size', None)
                burst_interval = row.get('latency_simulation_burst_interval', None)
                return f"Burst ({burst_size},{burst_interval})"

    latency_df['burstiness'] = latency_df.apply(compute_burstiness, axis=1)
    latency_df['latency_category'] = latency_df['latency_category'].astype(str)
    latency_df['burstiness'] = latency_df['burstiness'].astype(str)

    non_sim = "No simulation"
    sim_cats = sorted(latency_df.loc[latency_df['latency_simulation_simulate'] == True, 'latency_category'].unique())
    full_order = [non_sim] + sim_cats
    latency_df['latency_category'] = pd.Categorical(latency_df['latency_category'],
                                                     categories=full_order, ordered=True)

    def compute_latency_numeric(row):
        if not row.get('latency_simulation_simulate', False):
            return 0.0
        else:
            try:
                min_val = float(row.get('latency_simulation_delay_min', 0))
                max_val = float(row.get('latency_simulation_delay_max', 0))
                return (min_val + max_val) / 2.0
            except Exception:
                return np.nan

    latency_df['latency_numeric'] = latency_df.apply(compute_latency_numeric, axis=1)

    unique_x = latency_df[['latency_numeric', 'latency_category']].drop_duplicates().sort_values('latency_numeric')
    x_ticks = unique_x['latency_numeric'].values
    x_labels = unique_x['latency_category'].values

    energy_pivot = latency_df.pivot_table(
        index='latency_category',
        columns='burstiness',
        values='energy_per_token_kwh',
        aggfunc='mean'
    )
    flops_pivot = latency_df.pivot_table(
        index='latency_category',
        columns='burstiness',
        values='flops_per_token',
        aggfunc='mean'
    )

    x_positions = np.arange(len(full_order))

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    for burst_cat in energy_pivot.columns:
        y_energy = energy_pivot[burst_cat].reindex(full_order)
        ax1.plot(x_positions, y_energy, marker='o', linestyle='-', label=f"Energy: {burst_cat}")
    for burst_cat in flops_pivot.columns:
        y_flops = flops_pivot[burst_cat].reindex(full_order)
        ax2.plot(x_positions, y_flops, marker='s', linestyle='--', label=f"FLOPs: {burst_cat}")

    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(full_order, rotation=45, ha='right')
    ax1.set_xlabel("Latency Config")
    ax1.set_ylabel("Energy per Token (kWh)", color='tab:blue')
    ax2.set_ylabel("FLOPs per Token", color='tab:red')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    plt.title("Energy- and FLOPs-per-Token vs Latency Config (Dynamic)")
    fig.tight_layout()
    plt.show()


# ---------------------------
# Function to Plot All Figures
# ---------------------------
def plot_all(dfs):
    """
    Calls all individual plotting functions.
    """
    plot_num_processes(dfs)
    plot_batching(dfs)
    plot_precision(dfs)
    plot_decoder_temperature(dfs)
    plot_decoder_topk_top_p(dfs)
    plot_latency(dfs)
    plot_latency_dynamic(dfs)


# ---------------------------
# Main Section (Optional)
# ---------------------------
if __name__ == "__main__":
    csv_path = "results/controlled_results.csv"
    df = run_load_clean_diagnose_data(csv_path)
    
    # Then create the dfs dictionary as follows:
    configs = ['num_processes', 'batching', 'precis', 'decoding', 'latency']
    dfs = {config: df[df['config_name'].str.startswith(config)] for config in configs}
    
