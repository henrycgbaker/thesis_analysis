import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter
import pandas as pd
import numpy as np
from matplotlib import cm
    
def _plot_with_band(ax, raw_df, x_col, y_col, mean_df, mean_col, std_col,
                    color, raw_kwargs=None, band_alpha=0.2,
                    normalise_axes=None,
                    plot_mean=True, plot_band=True, plot_raw=True,
                    label_mean=None, label_band=None, label_raw=None):
    raw_kwargs = raw_kwargs or {}
    normalise_axes = normalise_axes or []

    # Determine x positions
    idx_str = mean_df.index.astype(str)
    try:
        positions = idx_str.astype(float)
    except ValueError:
        positions = np.arange(len(idx_str))
        ax.set_xticks(positions)
        ax.set_xticklabels(idx_str)
    mapping = {str(v): p for v, p in zip(idx_str, positions)}
    raw_x = raw_df[x_col].astype(str).map(mapping)

    # Compute mean/std
    mean_vals = mean_df[mean_col].values.copy()
    std_vals  = mean_df[std_col].fillna(0).values.copy()
    lower     = mean_vals - std_vals
    upper     = mean_vals + std_vals

    # Normalize if requested
    baseline = None
    if ax in normalise_axes:
        baseline    = mean_vals[0]
        mean_vals  /= baseline
        std_vals   /= baseline
        lower       = mean_vals - std_vals
        upper       = mean_vals + std_vals
        # Append "(normalised)" once
        old_label = ax.get_ylabel()
        if "(normalised)" not in old_label:
            color_lbl = ax.yaxis.label.get_color()
            ax.set_ylabel(f"{old_label} (normalised)", color=color_lbl)
        # Suffix every y-tick
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:g}x"))

    # Raw scatter (no legend entry unless label_raw)
    if plot_raw:
        raw_y = raw_df[y_col] if baseline is None else raw_df[y_col] / baseline
        scatter_kwargs = dict(color=color,
                              alpha=raw_kwargs.get('alpha', 0.3),
                              marker=raw_kwargs.get('marker','o'),
                              label=None)
        ax.scatter(raw_x, raw_y, **scatter_kwargs)

    # Plot mean ± band
    if len(positions) > 1:
        x_fine    = np.linspace(positions.min(), positions.max(), 200)
        mean_fine = np.interp(x_fine, positions, mean_vals)
        low_fine  = np.interp(x_fine, positions, lower)
        up_fine   = np.interp(x_fine, positions, upper)

        if plot_mean:
            pk = dict(linestyle='-', color=color)
            if label_mean:
                pk['label'] = label_mean
            ax.plot(x_fine, mean_fine, **pk)

        if plot_band:
            fk = dict(color=color, alpha=band_alpha, label=None)
            ax.fill_between(x_fine, low_fine, up_fine, **fk)
    else:
        if plot_mean:
            pk = dict(marker=raw_kwargs.get('marker','o'), linestyle='-', color=color)
            if label_mean:
                pk['label'] = label_mean
            ax.plot(positions, mean_vals, **pk)
        if plot_band:
            fk = dict(color=color, alpha=band_alpha, label=None)
            ax.fill_between(positions, lower, upper, **fk)

# ---------------------------
# Plot: Number of Processes
# ---------------------------
def plot_num_processes(dfs, normalise_axes=None):
    df = dfs.get('num_processes')
    if df is None:
        print("num_processes DataFrame not found in dfs.")
        return
    df = df.copy()
    df['num_processes'] = df['num_processes'].astype(int)

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

    ax1.grid(True, axis='y', linestyle='--', alpha=0.4)
    ax1.grid(True, axis='x', linestyle=':',  alpha=0.2)

    name_to_ax = {'ax1': ax1, 'ax2': ax2}
    if normalise_axes:
        normalise_axes = [name_to_ax[n] for n in normalise_axes if n in name_to_ax]
    else:
        normalise_axes = []

    ax1.set_xlabel('Number of Processes')
    ax1.set_ylabel('Energy-per-Token (kWh)', color='tab:blue')
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    _plot_with_band(
        ax1, df, 'num_processes', 'energy_per_token_kwh',
        energy_stats, 'energy_mean','energy_std',
        color='tab:blue', raw_kwargs={'alpha':0.3,'marker':'o'},
        normalise_axes=normalise_axes,
        plot_mean=True, plot_band=True, plot_raw=True,
        label_mean='Mean energy (across runs)', label_band=None, label_raw=None
    )
    # Baseline annotation
    if ax1 in normalise_axes:
        first = energy_stats.index[0]
        ax1.axhline(1, linestyle=':', color='red', linewidth=2)
        ax1.text(ax1.get_xlim()[1], 1, f"baseline = {first} process", ha='right', va='bottom', color='red')

    _plot_with_band(
        ax2, df, 'num_processes', 'flops_per_token',
        flops_stats, 'flops_mean','flops_std',
        color='tab:red', raw_kwargs={'alpha':0.3,'marker':'s'},
        normalise_axes=[],
        plot_mean=True, plot_band=True, plot_raw=True,
        label_mean='FLOPs (constant)', label_band=None, label_raw=None
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
def plot_batching(dfs, normalise_axes=None):
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

    name_to_ax = {'ax1': ax1, 'ax2': ax2}
    if normalise_axes:
        normalise_axes = [name_to_ax[n] for n in normalise_axes if n in name_to_ax]
    else:
        normalise_axes = []

    ax1.grid(True, axis='y', linestyle='--', alpha=0.4)
    ax1.grid(True, axis='x', linestyle=':',  alpha=0.2)

    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Energy-per-Token (kWh)', color='tab:blue')
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    _plot_with_band(
        ax1, df, 'batch_size', 'energy_per_token_kwh',
        energy_stats, 'energy_mean','energy_std',
        color='tab:blue', raw_kwargs={'alpha':0.3,'marker':'o'}, band_alpha=0.2,
        normalise_axes=normalise_axes,
        plot_mean=True, plot_band=True, plot_raw=True,
        label_mean='Mean energy (across runs)', label_band=None, label_raw=None
    )
    # Baseline annotation
    if ax1 in normalise_axes:
        first = energy_stats.index[0]
        ax1.axhline(1, linestyle=':', color='red', linewidth=2)
        ax1.text(ax1.get_xlim()[1], 1, f"baseline = batch size of {first}", ha='right', va='bottom', color='red')

    _plot_with_band(
        ax2, df, 'batch_size', 'flops_per_token',
        flops_stats, 'flops_mean','flops_std',
        color='tab:red', raw_kwargs={'alpha':0.3,'marker':'s'}, band_alpha=0.2,
        normalise_axes=[],
        plot_mean=True, plot_band=True, plot_raw=True,
        label_mean='FLOPs (constant)', label_band=None, label_raw=None
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
def plot_precision(dfs, normalise_axes=None):
    df = dfs.get('precis')
    if df is None:
        print("precis DataFrame not found in dfs.")
        return
    df = df.copy()
    def mode(r):
        if r.get('load_in_4bit'):      return 'INT4'
        if r.get('load_in_8bit'):      return 'INT8'
        if r.get('fp_precision')=='torch.float16': return 'FP16'
        return 'FP32'
    df['mode'] = pd.Categorical(
        df.apply(mode, axis=1),
        categories=['FP32','FP16','INT8','INT4'], ordered=True
    )

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

    name_to_ax = {'ax1': ax1, 'ax2': ax2}
    if normalise_axes:
        normalise_axes = [name_to_ax[n] for n in normalise_axes if n in name_to_ax]
    else:
        normalise_axes = []

    ax1.grid(True, axis='y', linestyle='--', alpha=0.4)
    ax1.grid(True, axis='x', linestyle=':',  alpha=0.2)

    ax1.set_xlabel('Precision')
    ax1.set_ylabel('Energy-per-Token (kWh)', color='tab:blue')
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    _plot_with_band(
        ax1, df, 'mode', 'energy_per_token_kwh',
        energy_stats, 'energy_mean', 'energy_std',
        color='tab:blue', raw_kwargs={'alpha':0.3,'marker':'o'}, band_alpha=0.2,
        normalise_axes=normalise_axes,
        plot_mean=True, plot_band=True, plot_raw=True,
        label_mean='Mean energy (across runs)', label_band=None, label_raw=None
    )
    # Baseline annotation
    if ax1 in normalise_axes:
        first = energy_stats.index[0]
        ax1.axhline(1, linestyle=':', color='red', linewidth=2)
        ax1.text(ax1.get_xlim()[1], 1, f"baseline = {first}", ha='right', va='bottom', color='red')

    _plot_with_band(
        ax2, df, 'mode', 'flops_per_token',
        flops_stats, 'flops_mean', 'flops_std',
        color='tab:red', raw_kwargs={'alpha':0.3,'marker':'s'}, band_alpha=0.2,
        normalise_axes=[],
        plot_mean=True, plot_band=True, plot_raw=True,
        label_mean='FLOPs (constant)', label_band=None, label_raw=None
    )
    ax2.set_ylabel('FLOPs-per-Token', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, loc='best')

    plt.title('Energy & FLOPs-per-Token vs Precision')
    plt.tight_layout()
    plt.show()

# ---------------------------
# Plot: Decoder Temperature 
# ---------------------------

def plot_decoder_temperature(dfs,
                             normalise_axes=None,
                             plot_band=True,
                             plot_raw=True
                             ):
    df = dfs.get('decoding')
    if df is None:
        print("decoding DataFrame not found in dfs.")
        return

    df = df[df['decoder_config_decoding_mode'].isin(['greedy','top_k','top_p'])].copy()
    df['method']      = df['decoder_config_decoding_mode']
    df['temperature'] = df['decoder_temperature']
    colors = {'greedy':'tab:blue','top_k':'tab:green','top_p':'tab:red'}

    fig, ax1 = plt.subplots(figsize=(8,6))
    ax2 = ax1.twinx()

    # Map normalization: allow ['ax1'] to refer to energy axis
    name_to_ax = {'ax1': ax1, 'ax2': ax2}
    if normalise_axes:
        norm_axes = [name_to_ax[n] for n in normalise_axes if n in name_to_ax]
    else:
        norm_axes = []

    ax1.grid(True, axis='y', linestyle='--', alpha=0.4)
    ax1.grid(True, axis='x', linestyle=':',  alpha=0.2)

    ax1.set_xlabel('Decoder Temperature')
    ax1.set_ylabel('Energy-per-Token (kWh)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax2.set_ylabel('FLOPs-per-Token', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    for method, sub in df.groupby('method'):
        color = colors[method]
        if plot_raw:
            if ax1 in norm_axes:
                stats = sub.groupby('temperature')['energy_per_token_kwh'].mean()
                baseline = stats.iloc[0]
                y_raw = sub['energy_per_token_kwh'] / baseline
            else:
                y_raw = sub['energy_per_token_kwh']
            ax1.scatter(sub['temperature'], y_raw,
                        color=color, alpha=0.3, marker='o', label=None)

        stats = sub.groupby('temperature').agg(
            energy_mean=('energy_per_token_kwh','mean'),
            energy_std =('energy_per_token_kwh','std')
        )
        _plot_with_band(
            ax1, sub, 'temperature','energy_per_token_kwh',
            stats, 'energy_mean','energy_std',
            color=color, raw_kwargs={'alpha':0.0}, band_alpha=0.2,
            normalise_axes=norm_axes,
            plot_mean=True,
            plot_band=plot_band,
            plot_raw=False,
            label_mean=f"Mean energy ({method})",
            label_band=None, label_raw=None
        )

    flops_stats = df.groupby('temperature').agg(
        flops_mean=('flops_per_token','mean'),
        flops_std =('flops_per_token','std')
    )
    _plot_with_band(
        ax2, df, 'temperature','flops_per_token',
        flops_stats, 'flops_mean','flops_std',
        color='tab:red', raw_kwargs={'alpha':0.0,'marker':'s'}, band_alpha=0.2,
        normalise_axes=[],
        plot_mean=True,
        plot_band=False,
        plot_raw=False,
        label_mean='FLOPs (constant)',
        label_band=None,
        label_raw=None
    )

    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, labs1+labs2, loc='best')

    # Red baseline annotation for normalization
    if ax1 in norm_axes:
        ax1.axhline(1, linestyle='--', color='red', linewidth=2)
        ax1.text(ax1.get_xlim()[1], 1, 'baseline = greedy decoding', ha='right', va='bottom', color='red')

    plt.title('Energy & FLOPs-per-Token vs Decoder Temperature')
    plt.tight_layout()
    plt.show()
    
# ---------------------------
# Plot: Decoder Method by Temp
# ---------------------------
def plot_decoder_topk_top_p(
    dfs,
    normalise_axes=None,
    plot_band=True,
    plot_raw=True
):

    df = dfs.get('decoding')
    if df is None:
        print("decoding DataFrame not found in dfs.")
        return
    df = df[df['decoder_config_decoding_mode'].isin(['top_k','top_p'])].copy()
    df['temp'] = df['decoder_temperature']

    # Split data
    top_k_df = df[df['decoder_config_decoding_mode']=='top_k']
    top_p_df = df[df['decoder_config_decoding_mode']=='top_p']

    temps_k = sorted(top_k_df['temp'].unique())
    temps_p = sorted(top_p_df['temp'].unique())

    cmap = cm.viridis
    colors_k = {t: cmap(i/len(temps_k)) for i,t in enumerate(temps_k)}
    colors_p = {t: cmap(i/len(temps_p)) for i,t in enumerate(temps_p)}

    # Two vertical panels, same proportions
    fig, (ax_k, ax_p) = plt.subplots(2, 1, figsize=(8, 12), sharex=False)

        # Map names to axes for normalization
    norm_axes = []
    if normalise_axes:
        # if 'ax1' requested, normalize both energy panels
        if 'ax1' in normalise_axes:
            norm_axes = [ax_k, ax_p]
        # also allow targeting just top-p if desired
        if 'ax2' in normalise_axes:
            norm_axes = [ax_p]
        # remove duplicates
        norm_axes = list(dict.fromkeys(norm_axes))

    # --- Top-k subplot ---
    ax_k.grid(True, axis='y', linestyle='--', alpha=0.4)
    ax_k.grid(True, axis='x', linestyle=':',  alpha=0.2)
    ax_k2 = ax_k.twinx()
    ax_k2.set_ylabel('FLOPs-per-Token', color='tab:red')
    ax_k2.tick_params(axis='y', labelcolor='tab:red')

    # Plot constant FLOPs line on ax_k2
    flops_k_stats = top_k_df.groupby('decoder_top_k').agg(
        flops_mean=('flops_per_token','mean'), flops_std=('flops_per_token','std')
    )
    _plot_with_band(
        ax_k2, top_k_df, 'decoder_top_k', 'flops_per_token',
        flops_k_stats, 'flops_mean', 'flops_std',
        color='tab:red', raw_kwargs={'alpha':0.0,'marker':'s'}, band_alpha=0.2,
        normalise_axes=[], plot_mean=True, plot_band=False, plot_raw=False,
        label_mean='FLOPs (constant)', label_band=None, label_raw=None
    )

    # Plot energy per temp
    for t in temps_k:
        sub = top_k_df[top_k_df['temp']==t]
        if plot_raw:
            if ax_k in norm_axes:
                baseline = sub.groupby('decoder_top_k')['energy_per_token_kwh'].mean().iloc[0]
                y_raw = sub['energy_per_token_kwh'] / baseline
            else:
                y_raw = sub['energy_per_token_kwh']
            ax_k.scatter(sub['decoder_top_k'], y_raw,
                         color=colors_k[t], alpha=0.3, marker='o', label=None)
        stats = sub.groupby('decoder_top_k').agg(
            energy_mean=('energy_per_token_kwh','mean'), energy_std=('energy_per_token_kwh','std')
        )
        _plot_with_band(
            ax_k, sub, 'decoder_top_k', 'energy_per_token_kwh',
            stats, 'energy_mean', 'energy_std', color=colors_k[t],
            raw_kwargs={'alpha':0.0}, band_alpha=0.2,
            normalise_axes=norm_axes,
            plot_mean=True, plot_band=plot_band, plot_raw=False,
            label_mean=f"Mean, Temp {t}", label_band=None, label_raw=None
        )
    ax_k.set_xlabel("Top‑k Value")
    ax_k.set_ylabel("Energy‑per‑Token (kWh)", color='tab:blue')
    ax_k.tick_params(axis='y', labelcolor='tab:blue')
    ax_k.set_title("Energy‑per‑Token vs Top‑k")

    # Baseline for greedy decoding
    if ax_k in norm_axes:
        ax_k.axhline(1, linestyle='--', color='red', linewidth=2)
        ax_k.text(ax_k.get_xlim()[1], 1,
                  'baseline = greedy decoding', ha='right', va='bottom', color='red')

    # --- Top-p subplot ---
    ax_p.grid(True, axis='y', linestyle='--', alpha=0.4)
    ax_p.grid(True, axis='x', linestyle=':',  alpha=0.2)
    ax_p2 = ax_p.twinx()
    ax_p2.set_ylabel('FLOPs-per-Token', color='tab:red')
    ax_p2.tick_params(axis='y', labelcolor='tab:red')

    flops_p_stats = top_p_df.groupby('decoder_top_p').agg(
        flops_mean=('flops_per_token','mean'), flops_std=('flops_per_token','std')
    )
    _plot_with_band(
        ax_p2, top_p_df, 'decoder_top_p', 'flops_per_token',
        flops_p_stats, 'flops_mean', 'flops_std',
        color='tab:red', raw_kwargs={'alpha':0.0,'marker':'s'}, band_alpha=0.2,
        normalise_axes=[], plot_mean=True, plot_band=False, plot_raw=False,
        label_mean='FLOPs (constant)', label_band=None, label_raw=None
    )

    for t in temps_p:
        sub = top_p_df[top_p_df['temp']==t]
        if plot_raw:
            if ax_p in norm_axes:
                baseline = sub.groupby('decoder_top_p')['energy_per_token_kwh'].mean().iloc[0]
                y_raw = sub['energy_per_token_kwh'] / baseline
            else:
                y_raw = sub['energy_per_token_kwh']
            ax_p.scatter(sub['decoder_top_p'], y_raw,
                         color=colors_p[t], alpha=0.3, marker='o', label=None)
        stats = sub.groupby('decoder_top_p').agg(
            energy_mean=('energy_per_token_kwh','mean'), energy_std=('energy_per_token_kwh','std')
        )
        _plot_with_band(
            ax_p, sub, 'decoder_top_p', 'energy_per_token_kwh',
            stats, 'energy_mean', 'energy_std', color=colors_p[t],
            raw_kwargs={'alpha':0.0}, band_alpha=0.2,
            normalise_axes=norm_axes,
            plot_mean=True, plot_band=plot_band, plot_raw=False,
            label_mean=f"Mean, Temp {t}", label_band=None, label_raw=None
        )
    ax_p.set_xlabel("Top‑p Value")
    ax_p.set_ylabel("Energy‑per‑Token (kWh)", color='tab:blue')
    ax_p.tick_params(axis='y', labelcolor='tab:blue')
    ax_p.set_title("Energy‑per‑Token vs Top‑p")

    if ax_p in norm_axes:
        ax_p.axhline(1, linestyle='--', color='red', linewidth=2)
        ax_p.text(ax_p.get_xlim()[1], 1,
                  'baseline = greedy decoding', ha='right', va='bottom', color='red')

    # Legend only on top-k
    lines_k, labs_k = ax_k.get_legend_handles_labels()
    lines_k2, labs_k2 = ax_k2.get_legend_handles_labels()
    ax_k.legend(lines_k + lines_k2, labs_k + labs_k2, loc='best', title="Decoder Temp")

    plt.tight_layout()
    plt.show()
    
# ---------------------------
# Plot: Latency
# ---------------------------

def add_latency_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['latency_numeric'] = df.apply(
        lambda r: 0.0 if not r['latency_simulation_simulate']
                    else (r['latency_simulation_delay_min'] +
                          r['latency_simulation_delay_max']) / 2,
        axis=1
    )
    return df

def classify_by_burst_size_and_constant(df: pd.DataFrame):
    df2 = add_latency_numeric(df.copy())
    origin = df2['latency_numeric'] == 0.0
    df2['class_const'] = (
        (df2['latency_simulation_simulate'] & ~df2['latency_simulation_simulate_burst'])
        | origin
    )
    burst_rows = df2['latency_simulation_simulate_burst']
    sizes = sorted(df2.loc[burst_rows, 'latency_simulation_burst_size'].unique())
    for sz in sizes:
        col = f'class_burst_{sz}'
        df2[col] = (
            (df2['latency_simulation_simulate_burst'] &
             (df2['latency_simulation_burst_size'] == sz))
            | origin
        )
    return df2, sizes


def classify_by_burst_interval_and_constant(df: pd.DataFrame):
    df2 = add_latency_numeric(df.copy())
    origin = df2['latency_numeric'] == 0.0
    df2['class_const'] = (
        (df2['latency_simulation_simulate'] & ~df2['latency_simulation_simulate_burst'])
        | origin
    )
    burst_rows = df2['latency_simulation_simulate_burst']
    intervals = sorted(df2.loc[burst_rows, 'latency_simulation_burst_interval'].unique())
    for iv in intervals:
        col = f'class_interval_{iv}'
        df2[col] = (
            (df2['latency_simulation_simulate_burst'] &
             (df2['latency_simulation_burst_interval'] == iv))
            | origin
        )
    return df2, intervals


def plot_latency_by_burst_size(
    dfs,
    normalise_axes=None,
    plot_band=True,
    plot_raw=True
):

    df0 = dfs.get('latency')
    if df0 is None:
        print("latency DataFrame not found in dfs.")
        return
    df1, sizes = classify_by_burst_size_and_constant(df0)

    # prepare classes
    classes = [('Constant latency', 'class_const', 'tab:orange')]
    cmap = cm.get_cmap('Blues', len(sizes)+1)
    for i, sz in enumerate(sizes, start=1):
        classes.append((f'Burst size {sz}', f'class_burst_{sz}', cmap(i)))

    # unified fig size
    fig, ax = plt.subplots(figsize=(8,6))
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    ax.grid(True, axis='x', linestyle=':', alpha=0.2)
    ax.set_xlabel('Latency (ms)')
    ax.set_ylabel('Energy-per-Token (kWh)', color='tab:blue')
    ax.tick_params(axis='y', labelcolor='tab:blue')
    ax.set_title('Energy- vs Latency-per-token, by Burst Size')
    lat_min = df1['latency_numeric'].min()
    lat_max = df1['latency_numeric'].max()
    ax.set_xlim(lat_min, lat_max)

    # map normalisation
    name_to_ax = {'ax1': ax}
    if normalise_axes:
        norm_axes = [name_to_ax[n] for n in normalise_axes if n in name_to_ax]
    else:
        norm_axes = []

    # plot classes
    for label, col, color in classes:
        sub = df1[df1[col]]
        stats = sub.groupby('latency_numeric').agg(
            energy_mean=('energy_per_token_kwh','mean'), energy_std=('energy_per_token_kwh','std')
        )
        _plot_with_band(
            ax, sub, 'latency_numeric', 'energy_per_token_kwh',
            stats, 'energy_mean', 'energy_std',
            color=color, raw_kwargs={'alpha':0.3,'marker':'o'}, band_alpha=0.2,
            normalise_axes=norm_axes,
            plot_mean=True, plot_band=plot_band, plot_raw=plot_raw,
            label_mean=label, label_band='±1 std', label_raw='Raw energy' if plot_raw else None
        )

    # baseline line
    if ax in norm_axes:
        ax.axhline(1, linestyle='--', color='red', linewidth=2)
        ax.text(ax.get_xlim()[1], 1,
                'baseline = no latency', ha='right', va='bottom', color='red')

    handles, labels = ax.get_legend_handles_labels()
    # Modify labels conditionally
    new_labels = [
        'burst' + label + " (ms)" if "Interval" in label else label
        for label in labels
    ]

    # Apply to legend
    ax.legend(handles, new_labels, loc='best', title='Class')
    plt.tight_layout()
    plt.show()


def plot_latency_by_burst_interval(
    dfs,
    normalise_axes=None,
    plot_band=True,
    plot_raw=True
):
    df0 = dfs.get('latency')
    if df0 is None:
        print("latency DataFrame not found in dfs.")
        return
    df1, intervals = classify_by_burst_interval_and_constant(df0)

    classes = [('Constant latency', 'class_const', 'tab:orange')]
    cmap = cm.get_cmap('Blues', len(intervals)+1)
    for i, iv in enumerate(intervals, start=1):
        classes.append((f'Interval {iv}', f'class_interval_{iv}', cmap(i)))

    fig, ax = plt.subplots(figsize=(8,6))
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    ax.grid(True, axis='x', linestyle=':', alpha=0.2)
    ax.set_xlabel('Latency (ms)')
    ax.set_ylabel('Energy-per-Token (kWh)', color='tab:blue')
    ax.tick_params(axis='y', labelcolor='tab:blue')
    ax.set_title('Energy- vs Latency-per-Token, by Burst Interval')
    lat_min = df1['latency_numeric'].min()
    lat_max = df1['latency_numeric'].max()
    ax.set_xlim(lat_min, lat_max)

    name_to_ax = {'ax1': ax}
    if normalise_axes:
        norm_axes = [name_to_ax[n] for n in normalise_axes if n in name_to_ax]
    else:
        norm_axes = []

    for label, col, color in classes:
        sub = df1[df1[col]]
        stats = sub.groupby('latency_numeric').agg(
            energy_mean=('energy_per_token_kwh','mean'), energy_std=('energy_per_token_kwh','std')
        )
        _plot_with_band(
            ax, sub, 'latency_numeric', 'energy_per_token_kwh',
            stats, 'energy_mean', 'energy_std', color=color,
            raw_kwargs={'alpha':0.3,'marker':'o'}, band_alpha=0.2,
            normalise_axes=norm_axes,
            plot_mean=True, plot_band=plot_band, plot_raw=plot_raw,
            label_mean=label, label_band='±1 std', label_raw='Raw energy' if plot_raw else None
        )

    if ax in norm_axes:
        ax.axhline(1, linestyle='--', color='red', linewidth=2)
        ax.text(ax.get_xlim()[1], 1, 'baseline = no latency', ha='right', va='bottom', color='red')

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='best', title='Class')
    plt.tight_layout()
    plt.show()

# ---------------------------
# Function to Plot All Figures
# ---------------------------
def plot_all_vizs(dfs):
    """
    Calls all individual plotting functions.
    """
    # # of proc
    #plot_num_processes(dfs)
    plot_num_processes(dfs, normalise_axes=['ax1'])
    
    # batch size
    #plot_batching(dfs)
    plot_batching(dfs, normalise_axes=['ax1'])
    
    # precision
    #plot_precision(dfs)
    plot_precision(dfs, normalise_axes=['ax1'])
    
    # decoder 
    # -temp
    #plot_decoder_temperature(dfs)
    #plot_decoder_temperature(dfs, normalise_axes=['ax1'])
    plot_decoder_temperature(dfs, plot_band=False, plot_raw=False)
    plot_decoder_temperature(dfs, normalise_axes=['ax1'], plot_band=False, plot_raw=True)
    
    # - top p/k
    #plot_decoder_topk_top_p(dfs)
    #plot_decoder_topk_top_p(dfs, normalise_axes=['ax1'])
    plot_decoder_topk_top_p(dfs, plot_band=False, plot_raw=False)
    plot_decoder_topk_top_p(dfs, normalise_axes=['ax1'], plot_band=False, plot_raw=True)
    
    # latency 
    # - burst size
    #plot_latency_by_burst_size(dfs)
    #plot_latency_by_burst_size(dfs, normalise_axes=['ax1'])
    plot_latency_by_burst_size(dfs, plot_band=False, plot_raw=False)
    plot_latency_by_burst_size(dfs, normalise_axes=['ax1'], plot_band=False, plot_raw=True)
    
    # - burst interval
    #plot_latency_by_burst_interval(dfs)
    #plot_latency_by_burst_interval(dfs, normalise_axes=['ax1'])
    plot_latency_by_burst_interval(dfs, plot_band=False, plot_raw=False)
    plot_latency_by_burst_interval(dfs, normalise_axes=['ax1'], plot_band=False, plot_raw=True)

# ---------------------------
