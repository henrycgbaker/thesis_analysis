import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd
import numpy as np

# Helper function unchanged

def _plot_with_band(ax, raw_df, x_col, y_col, mean_df, mean_col, std_col,
                    color, raw_kwargs=None, band_alpha=0.2,
                    line_kwargs=None,
                    normalise_axes=None,
                    plot_mean=True, plot_band=True, plot_raw=True,
                    label_mean=None):
    raw_kwargs = raw_kwargs or {}
    line_kwargs = line_kwargs or {}
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
    is_normalised = False
    if ax in normalise_axes:
        is_normalised = True
        baseline    = mean_vals[0]
        mean_vals  /= baseline
        std_vals   /= baseline
        lower       = mean_vals - std_vals
        upper       = mean_vals + std_vals
        old_label = ax.get_ylabel()
        if "(normalised)" not in old_label:
            ax.set_ylabel(f"{old_label} (normalised)", color=ax.yaxis.label.get_color())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:g}x"))

    # Scatter raw data
    if plot_raw:
        raw_y = raw_df[y_col] / baseline if baseline is not None else raw_df[y_col]
        ax.scatter(raw_x, raw_y,
                   alpha=raw_kwargs.get('alpha',0.2),
                   marker=raw_kwargs.get('marker','o'),
                   color=raw_kwargs.get('color'),
                   label=None)

    # Plot mean line and band
    x_vals = np.linspace(positions.min(), positions.max(), len(positions)) if len(positions)>1 else positions
    if plot_mean:
        ax.plot(x_vals,
                mean_vals,
                linestyle=line_kwargs.get('linestyle','-'),
                marker=line_kwargs.get('marker',None),
                alpha=line_kwargs.get('alpha',1.0),
                color=line_kwargs.get('color'),
                label=label_mean)
    if plot_band:
        ax.fill_between(x_vals,
                        lower, upper,
                        alpha=band_alpha,
                        color=line_kwargs.get('color'),
                        label=None)

# ---------------------------
# Plot: Decoder Temperature
# ---------------------------

def plot_decoder_temperature(
    dfs,
    normalise_axes=None,
    plot_mean=True,
    plot_band=True,
    plot_raw=True,
    add_baseline_energy=False,
    cycle_id=None,
    model=None
):
    """
    Plot Energy-per-Token vs Decoder Temperature for one or multiple models.

    Parameters:
        dfs: dict of DataFrames with key 'decoding'
        normalise_axes: list of axis names to normalise ['ax1']
        plot_mean: whether to plot mean line
        plot_band: whether to plot std deviation band
        plot_raw: whether to plot raw scatter
        add_baseline_energy: whether to annotate a baseline from greedy decoding
        cycle_id: filter data to a specific cycle_id or None
        model: string or list of model names to filter, or None for all
    """
    df = dfs.get('decoding')
    if df is None:
        print("decoding DataFrame not found in dfs.")
        return

    # Filter by cycle_id
    if cycle_id is not None:
        df = df[df['cycle_id']==cycle_id]
    # Filter by model(s)
    if model is not None:
        models = model if isinstance(model, (list,tuple)) else [model]
        df = df[df['model'].isin(models)]
    else:
        models = sorted(df['model'].unique())

    # Setup styles per model
    linestyles = ['-', '--', '-.', (0,(5,1))]
    markers    = ['o','D','^','s']
    alpha_line    = 1.0
    alpha_scatter = 0.2
    alpha_band    = 0.2

    fig, ax = plt.subplots(figsize=(8,6))
    ax.set_xlabel('Decoder Temperature')
    ax.set_ylabel('Energy-per-Token (kWh)')
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    ax.grid(True, axis='x', linestyle=':',  alpha=0.2)

    # Map normalisation
    norm_axes = []
    if normalise_axes and 'ax1' in normalise_axes:
        norm_axes = [ax]

    # Loop models and methods
    methods = ['greedy','top_k','top_p']
    colors = {'greedy':'tab:blue','top_k':'tab:green','top_p':'tab:red'}
    for m_i, m_name in enumerate(models):
        style = linestyles[m_i % len(linestyles)]
        marker = markers[m_i % len(markers)]
        sub_m = df[df['model']==m_name]
        for method in methods:
            sub = sub_m[sub_m['decoder_config_decoding_mode']==method]
            if sub.empty: continue
            stats = sub.groupby('decoder_temperature').agg(
                energy_mean=('energy_per_token_kwh','mean'),
                energy_std =('energy_per_token_kwh','std')
            )
            _plot_with_band(
                ax, sub, 'decoder_temperature','energy_per_token_kwh',
                stats, 'energy_mean','energy_std',
                color=colors[method],
                raw_kwargs={'alpha':alpha_scatter,'marker':marker,'color':colors[method]},
                line_kwargs={'linestyle':style,'marker':marker,'alpha':alpha_line,'color':colors[method]},
                band_alpha=alpha_band,
                normalise_axes=norm_axes,
                plot_mean=plot_mean,
                plot_band=plot_band,
                plot_raw=plot_raw,
                label_mean=f"{m_name}:{method}"
            )
    if add_baseline_energy:
        # one baseline per model, using greedy config
        for m_i, m_name in enumerate(models):
            base_df = df[(df['model']==m_name) & (df['decoder_config_decoding_mode']=='greedy')]
            if base_df.empty: continue
            stats = base_df.groupby('decoder_temperature')['energy_per_token_kwh'].mean()
            is_norm = bool(norm_axes)
            base = 1.0 if is_norm else stats.iloc[0]
            xs = stats.index.astype(float)
            label = f"Baseline {m_name}:greedy"
            ax.axhline(base, linestyle=':', color='gray', alpha=0.6)
            ax.text(xs[-1], base, label, ha='right', va='bottom', fontsize='small', color='gray', alpha=0.4)

    plt.title('Energy-per-Token vs Decoder Temperature')
    ax.legend(loc='best')
    plt.tight_layout()
    plt.show()
