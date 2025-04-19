import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def _plot_with_band(ax, raw_df, x_col, y_col, mean_df, mean_col, std_col,
                    color, raw_kwargs=None, band_alpha=0.2,
                    label_mean="Mean", label_band="±1 std", label_raw="Raw"):
    """
    Plot raw scatter and a smoothed mean ± std band on ax.
    Maps categorical x to numeric positions if needed.
    """
    raw_kwargs = raw_kwargs or {}

    # Determine positions for x-axis
    idx_str = mean_df.index.astype(str)
    try:
        positions = idx_str.astype(float)
    except ValueError:
        positions = np.arange(len(idx_str))
        ax.set_xticks(positions)
        ax.set_xticklabels(idx_str)
    # Map raw x to positions
    mapping = {str(v): p for v, p in zip(idx_str, positions)}
    raw_x = raw_df[x_col].astype(str).map(mapping)

    # Scatter raw points
    if label_raw:
        ax.scatter(raw_x, raw_df[y_col],
                   color=color, alpha=raw_kwargs.get('alpha',0.3),
                   marker=raw_kwargs.get('marker','o'),
                   label=label_raw)

    # Compute mean and std arrays
    mean_vals = mean_df[mean_col].values
    std_vals = mean_df[std_col].fillna(0).values
    lower = mean_vals - std_vals
    upper = mean_vals + std_vals

    if len(positions) > 1:
        # Smooth interpolation
        x_fine = np.linspace(positions.min(), positions.max(), 200)
        mean_fine = np.interp(x_fine, positions, mean_vals)
        low_fine = np.interp(x_fine, positions, lower)
        up_fine = np.interp(x_fine, positions, upper)
        # Plot mean line
        ax.plot(x_fine, mean_fine, linestyle='-', color=color, label=label_mean)
        # Plot band
        if label_band:
            ax.fill_between(x_fine, low_fine, up_fine, color=color, alpha=band_alpha, label=label_band)
    else:
        # Single point
        ax.plot(positions, mean_vals,
                marker=raw_kwargs.get('marker','o'), linestyle='-', color=color,
                label=label_mean)
        if label_band:
            ax.fill_between(positions, lower, upper, color=color, alpha=band_alpha, label=label_band)


# ---------------------------
# Plot: Number of Processes
# ---------------------------
def plot_num_processes(dfs):
    df = dfs.get('num_processes')
    if df is None:
        print("num_processes DataFrame not found in dfs.")
        return
    df = df.copy()
    df['num_processes'] = df['num_processes'].astype(int)

    # Stats
    energy_stats = df.groupby('num_processes').agg(
        energy_mean=('energy_per_token_kwh','mean'),
        energy_std =('energy_per_token_kwh','std')
    )
    flops_stats = df.groupby('num_processes').agg(
        flops_mean=('flops_per_token','mean'),
        flops_std =('flops_per_token','std')
    )

    fig, ax1 = plt.subplots(figsize=(8,6))
    ax2 = ax1.twinx()
    # Energy
    _plot_with_band(
        ax1, df, 'num_processes', 'energy_per_token_kwh',
        energy_stats, 'energy_mean','energy_std',
        color='tab:blue',
        raw_kwargs={'alpha':0.3,'marker':'o'},
        label_mean='Mean energy', label_band='±1 std', label_raw='Raw energy'
    )
    ax1.set_xlabel('Number of Processes')
    ax1.set_ylabel('Energy-per-Token (kWh)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # FLOPs
    _plot_with_band(
        ax2, df, 'num_processes', 'flops_per_token',
        flops_stats, 'flops_mean','flops_std',
        color='tab:red',
        raw_kwargs={'alpha':0.3,'marker':'s'},
        label_mean='Mean FLOPs', label_band=None, label_raw='Raw FLOPs'
    )
    ax2.set_ylabel('FLOPs-per-Token', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, labs1+labs2, loc='best')

    plt.title('Energy & FLOPs-per-Token vs Number of Processes')
    plt.tight_layout()
    plt.show()


# ---------------------------
# Plot: Batching
# ---------------------------
def plot_batching(dfs):
    df = dfs.get('batching')
    if df is None:
        print("batching DataFrame not found in dfs.")
        return
    df = df.copy()
    df['batch_size'] = df['batch_size___fixed_batching'].astype(int)

    energy_stats = df.groupby('batch_size').agg(
        energy_mean=('energy_per_token_kwh','mean'),
        energy_std =('energy_per_token_kwh','std')
    )
    flops_stats = df.groupby('batch_size').agg(
        flops_mean=('flops_per_token','mean'),
        flops_std =('flops_per_token','std')
    )

    fig, ax1 = plt.subplots(figsize=(8,6))
    ax2 = ax1.twinx()

    _plot_with_band(
        ax1, df, 'batch_size','energy_per_token_kwh',
        energy_stats,'energy_mean','energy_std',
        'tab:blue', raw_kwargs={'alpha':0.3,'marker':'o'},
        label_mean='Mean energy', label_band='±1 std', label_raw='Raw energy'
    )
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Energy-per-Token (kWh)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    _plot_with_band(
        ax2, df, 'batch_size','flops_per_token',
        flops_stats,'flops_mean','flops_std',
        'tab:red', raw_kwargs={'alpha':0.3,'marker':'s'},
        label_mean='Mean FLOPs', label_band=None, label_raw='Raw FLOPs'
    )
    ax2.set_ylabel('FLOPs-per-Token', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, labs1+labs2, loc='best')

    plt.title('Energy & FLOPs-per-Token vs Batch Size')
    plt.tight_layout()
    plt.show()


# ---------------------------
# Plot: Precision
# ---------------------------
def plot_precision(dfs):
    df = dfs.get('precis')
    if df is None:
        print("precis DataFrame not found in dfs.")
        return
    df = df.copy()
    def mode(r):
        if r.get('load_in_4bit'): return 'INT4'
        if r.get('load_in_8bit'): return 'INT8'
        if r.get('fp_precision')=='torch.float16': return 'FP16'
        return 'FP32'
    df['mode'] = pd.Categorical(df.apply(mode,axis=1),
                                 categories=['FP32','FP16','INT8','INT4'],ordered=True)

    energy_stats = df.groupby('mode').agg(
        energy_mean=('energy_per_token_kwh','mean'),
        energy_std =('energy_per_token_kwh','std')
    )
    flops_stats = df.groupby('mode').agg(
        flops_mean=('flops_per_token','mean'),
        flops_std =('flops_per_token','std')
    )

    fig, ax1 = plt.subplots(figsize=(8,6))
    ax2 = ax1.twinx()

    _plot_with_band(
        ax1, df, 'mode','energy_per_token_kwh',
        energy_stats,'energy_mean','energy_std',
        'tab:blue', raw_kwargs={'alpha':0.3,'marker':'o'},
        label_mean='Mean energy', label_band='±1 std', label_raw='Raw energy'
    )
    ax1.set_xlabel('Precision')
    ax1.set_ylabel('Energy-per-Token (kWh)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    _plot_with_band(
        ax2, df, 'mode','flops_per_token',
        flops_stats,'flops_mean','flops_std',
        'tab:red', raw_kwargs={'alpha':0.3,'marker':'s'},
        label_mean='Mean FLOPs', label_band=None, label_raw='Raw FLOPs'
    )
    ax2.set_ylabel('FLOPs-per-Token', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, labs1+labs2, loc='best')

    plt.title('Energy & FLOPs-per-Token vs Precision')
    plt.tight_layout()
    plt.show()


# ---------------------------
# Plot: Decoder Temperature by Method
# ---------------------------
def plot_decoder_temperature(dfs):
    df = dfs.get('decoding')
    if df is None:
        print("decoding DataFrame not found in dfs.")
        return
    df = df[df['decoder_config_decoding_mode'].notna()]
    df = df[df['decoder_config_decoding_mode'].isin(['greedy','top_k','top_p'])]
    df['method'] = df['decoder_config_decoding_mode']
    df['temperature'] = df['decoder_temperature']

    colors = {'greedy':'tab:blue','top_k':'tab:green','top_p':'tab:red'}
    fig, ax1 = plt.subplots(figsize=(14,8))
    ax2 = ax1.twinx()

    for method, sub in df.groupby('method'):
        color = colors[method]
        # raw scatter
        ax1.scatter(sub['temperature'], sub['energy_per_token_kwh'], color=color, alpha=0.3,
                    marker='o', label=f"Raw energy ({method})")
        ax2.scatter(sub['temperature'], sub['flops_per_token'], color=color, alpha=0.3,
                    marker='s', label=None)
        # stats
        stats = sub.groupby('temperature').agg(
            energy_mean=('energy_per_token_kwh','mean'),
            energy_std =('energy_per_token_kwh','std'),
            flops_mean =('flops_per_token','mean'),
            flops_std  =('flops_per_token','std'),
        )
        # band & mean
        _plot_with_band(ax1, sub, 'temperature','energy_per_token_kwh',
                        stats,'energy_mean','energy_std',
                        color, raw_kwargs={'alpha':0.0},
                        label_mean=f"Mean energy ({method})",
                        label_band="±1 std",
                        label_raw=None)
        _plot_with_band(ax2, sub, 'temperature','flops_per_token',
                        stats,'flops_mean','flops_std',
                        color, raw_kwargs={'alpha':0.0,'marker':'s'},
                        label_mean=f"Mean FLOPs ({method})",
                        label_band=None,
                        label_raw=None)

    ax1.set_xlabel('Decoder Temperature')
    ax1.set_ylabel('Energy-per-Token (kWh)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax2.set_ylabel('FLOPs-per-Token', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, labs1+labs2, loc='best')

    plt.title('Energy & FLOPs-per-Token vs Decoder Temperature')
    plt.tight_layout()
    plt.show()


def plot_decoder_topk_top_p(dfs):
    """
    Creates two side‑by‑side subplots for raw + mean±std Energy-per-Token vs Top-k and Top-p.
    Colours each series by decoder temperature, with smooth error bands.
    """
    if 'decoding' not in dfs:
        print("decoding DataFrame not found in dfs.")
        return

    df = dfs['decoding'].copy()
    df = df[df['decoder_config_decoding_mode'].isin(['top_k','top_p'])]
    df['temp'] = df['decoder_temperature']

    # Split out top_k vs top_p runs
    top_k_df = df[df['decoder_config_decoding_mode']=='top_k']
    top_p_df = df[df['decoder_config_decoding_mode']=='top_p']

    # Define colour map per unique temperature
    cmap   = plt.cm.viridis
    temps_k = sorted(top_k_df['temp'].unique())
    temps_p = sorted(top_p_df['temp'].unique())
    colors_k = {t: cmap(i/len(temps_k)) for i,t in enumerate(temps_k)}
    colors_p = {t: cmap(i/len(temps_p)) for i,t in enumerate(temps_p)}

    fig, (ax_k, ax_p) = plt.subplots(1, 2, figsize=(14,6))

    # --- Top‑k subplot ---
    for t in temps_k:
        sub = top_k_df[top_k_df['temp']==t]
        # 1) raw scatter
        ax_k.scatter(
            sub['decoder_top_k'], sub['energy_per_token_kwh'],
            color=colors_k[t], alpha=0.3, marker='o', s=40,
            label=f"Raw, Temp {t}"
        )
        # 2) compute stats
        stats = sub.groupby('decoder_top_k').agg(
            energy_mean=('energy_per_token_kwh','mean'),
            energy_std =('energy_per_token_kwh','std')
        )
        # 3) smooth band + mean
        _plot_with_band(
            ax_k, sub, 'decoder_top_k', 'energy_per_token_kwh',
            stats, 'energy_mean', 'energy_std',
            color=colors_k[t],
            raw_kwargs={'alpha':0.0},
            label_mean=f"Mean, Temp {t}",
            label_band="±1 std",
            label_raw=None
        )
    ax_k.set_xlabel("Top‑k Value")
    ax_k.set_ylabel("Energy‑per‑Token (kWh)")
    ax_k.set_title("Energy‑per‑Token vs Top‑k")
    ax_k.grid(True)

    # --- Top‑p subplot ---
    for t in temps_p:
        sub = top_p_df[top_p_df['temp']==t]
        ax_p.scatter(
            sub['decoder_top_p'], sub['energy_per_token_kwh'],
            color=colors_p[t], alpha=0.3, marker='o', s=40,
            label=f"Raw, Temp {t}"
        )
        stats = sub.groupby('decoder_top_p').agg(
            energy_mean=('energy_per_token_kwh','mean'),
            energy_std =('energy_per_token_kwh','std')
        )
        _plot_with_band(
            ax_p, sub, 'decoder_top_p', 'energy_per_token_kwh',
            stats, 'energy_mean', 'energy_std',
            color=colors_p[t],
            raw_kwargs={'alpha':0.0},
            label_mean=f"Mean, Temp {t}",
            label_band="±1 std",
            label_raw=None
        )
    ax_p.set_xlabel("Top‑p Value")
    ax_p.set_ylabel("Energy‑per‑Token (kWh)")
    ax_p.set_title("Energy‑per‑Token vs Top‑p")
    ax_p.grid(True)

    # Combine legends
    lines_k, labs_k = ax_k.get_legend_handles_labels()
    lines_p, labs_p = ax_p.get_legend_handles_labels()
    ax_k.legend(lines_k + lines_p, labs_k + labs_p, loc='best', title="Decoder Temp")

    plt.tight_layout()
    plt.show()



def plot_latency(dfs):

    if 'latency' not in dfs:
        print("latency DataFrame not found in dfs.")
        return

    df = dfs['latency'].copy()

    # 1) Numeric latency: 0 for no sim, else mean of min/max
    df['latency_numeric'] = df.apply(
        lambda r: 0.0
                  if not r.get('latency_simulation_simulate', False)
                  else (float(r['latency_simulation_delay_min']) +
                        float(r['latency_simulation_delay_max'])) / 2,
        axis=1
    )

    # 2) Base boolean flags (bursty=T, constant=F)
    bursty   = df['latency_simulation_simulate_burst'].fillna(False)
    constant = ~bursty

    # “No sim” rows belong to both classes
    no_sim = ~df['latency_simulation_simulate'].fillna(False)
    df['latency_bursty']   = bursty   | no_sim
    df['latency_constant'] = constant | no_sim

    # Print counts:
    print("Observations per class:")
    print(f"  No simulation total:     {no_sim.sum()}")
    print(f"  In Latency (constant):   {constant.sum()}")
    print(f"  In Latency (bursty):     {bursty.sum()}")

    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    ax1.set_xlabel("Mean Latency (ms)")
    ax1.set_ylabel("Energy-per-Token (kWh)")
    ax2.set_ylabel("FLOPs-per-Token")

    colors = {
        'Latency (constant)': 'tab:blue',
        'Latency (bursty)'   : 'tab:red'
    }
    
    # Energy curves per class (allowing overlap)
    mask_dict = {
        'Latency (constant)': df['latency_constant'],
        'Latency (bursty)' : df['latency_bursty']
    }
    
    # Build burst‐range categories:
    burst_ranges = (
        df[df['latency_simulation_simulate_burst']]
        [['latency_simulation_delay_min','latency_simulation_delay_max']]
        .drop_duplicates()
        .apply(tuple, axis=1)
        .tolist()
    )
    
    # 4) Create a categorical column “burst_range” that matches each row
    df['burst_range'] = list(zip(df['latency_simulation_delay_min'],
                                 df['latency_simulation_delay_max']))
    #    Anything outside your four defined ranges gets labeled “other”
    df['burst_range'] = df['burst_range'].where(
        df['burst_range'].isin(burst_ranges),
        other='other'
    )

    # 5) Choose a blue colormap ramp:
    cmap = cm.get_cmap('Blues', n_ranges + 1)  # +1 so “other” can be very light
    # Map each range → a color
    colors = {rng: cmap(i)
              for i, rng in enumerate(burst_ranges)}
    colors['other'] = cmap(0.2)  # pale blue for anything else
    
    for cls_name, mask in mask_dict.items():
        sub_df = df[mask]
        
        c = colors[cls_name]
        
        ax1.scatter(sub_df['latency_numeric'],
                    sub_df['energy_per_token_kwh'],
                    color=c, alpha=0.3, label=f"Raw energy ({cls})")

        stats = (sub_df.groupby('latency_numeric')
                    .energy_per_token_kwh
                    .agg(['mean','std'])
                    .rename(columns={'mean':'m','std':'s'}))
        
        # Patch in 0.0 manually if needed
        if 0.0 not in stats.index and not sub_df[sub_df['latency_numeric']==0].empty:
            z = sub_df[sub_df['latency_numeric']==0].energy_per_token_kwh
            stats.loc[0.0] = [z.mean(), z.std()]
        stats = stats.sort_index()

        _plot_with_band(
            ax1, sub_df, 'latency_numeric', 'energy_per_token_kwh',
            stats, 'm', 's', c,
            raw_kwargs={'alpha':0},
            label_mean=f"Mean energy ({cls})",
            label_band="±1 std"
        )

    # FLOPs curve (across all)
    fl = (df.groupby('latency_numeric')
            .flops_per_token
            .agg(['mean','std'])
            .rename(columns={'mean':'m','std':'s'}))
    fl = fl.sort_index()
    _plot_with_band(
        ax2, df, 'latency_numeric', 'flops_per_token',
        fl, 'm', 's', 'tab:purple',
        raw_kwargs={'alpha':0,'marker':'s'},
        label_mean="FLOPs (all latencies)"
    )

    # Tidy up axes, legend, title
    ax1.set_xlim(df['latency_numeric'].min(),
                 df['latency_numeric'].max())
    ax1.legend(loc='best')
    ax1.set_title("Energy & FLOPs-per-Token vs Latency (Dynamic)")
    plt.tight_layout()
    plt.show()

# ---------------------------
# Function to Plot All Figures
# ---------------------------
def plot_all_vizs(dfs):
    """
    Calls all individual plotting functions.
    """
    plot_num_processes(dfs)
    plot_batching(dfs)
    plot_precision(dfs)
    plot_decoder_temperature(dfs)
    plot_decoder_topk_top_p(dfs)
    plot_latency(dfs)

# ---------------------------
