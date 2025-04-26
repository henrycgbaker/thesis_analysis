import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from typing import Optional, List

# ---------------------------
# Internal plotting helper
# ---------------------------
def _plot_with_band(
    ax,
    x,
    mean_vals,
    std_vals,
    color,
    linestyle,
    marker,
    label,
    alpha_line,
    alpha_band,
    alpha_scatter,
    plot_band,
    plot_raw
):
    if plot_raw:
        ax.scatter(x, mean_vals,
                   marker=marker,
                   alpha=alpha_scatter,
                   color=color)
    ax.plot(x, mean_vals,
            linestyle=linestyle,
            marker=marker,
            color=color,
            label=label,
            alpha=alpha_line)
    if plot_band:
        ax.fill_between(x,
                        mean_vals - std_vals,
                        mean_vals + std_vals,
                        color=color,
                        alpha=alpha_band)


# ---------------------------
# Generic param vs metric
# ---------------------------
def plot_param_vs_metric(
    df: pd.DataFrame,
    param_col: str,
    ax1: str = 'energy_per_token',
    ax2: Optional[str] = None,          
    normalise_axes: List[str] = None,
    plot_mean: bool = True,
    plot_band: bool = True,
    plot_raw: bool = True,
    add_baseline_energy: bool = False,
    add_baseline_throughput: bool = False,
    models: List[str] = None
):
    # metric lookup
    metric_map = {
        'energy_per_token':         {'col': 'energy_per_token_kwh',
                                     'label': 'Energy per Token (kWh)'},
        'throughput_tokens_per_sec':{'col': 'throughput_tokens_per_sec',
                                     'label': 'Throughput (tokens/sec)'}
    }
    if ax1 not in metric_map:
        raise ValueError(f"ax1 must be one of {list(metric_map.keys())}")
    if ax2 is not None and ax2 not in metric_map:
        raise ValueError(f"ax2 must be one of {list(metric_map.keys())} or None")

    normalise_axes = normalise_axes or []
    norm1 = 'ax1' in normalise_axes
    norm2 = 'ax2' in normalise_axes

    # start a fresh figure
    fig, ax_left = plt.subplots(figsize=(8,6))

    # only create a twin if ax2 is given
    ax_right = ax_left.twinx() if ax2 else None

    # determine which models to loop
    if models is None and 'model' in df.columns:
        models = df['model'].dropna().unique().tolist()
    elif models is None:
        models = [None]

    linestyles    = ['-', '--', '-.', (0,(5,1))]
    markers       = ['o','D','^','s']
    alpha_line    = 1.0
    alpha_scatter = 0.2
    alpha_band    = 0.2

    blue = '#1f77b4'
    red  = '#d62728'

    # axis labels
    xlabel = param_col.replace('_',' ').title()
    ax_left.set_xlabel(xlabel)
    ax_left.set_ylabel(metric_map[ax1]['label'], color=blue)
    ax_left.tick_params(axis='y', labelcolor=blue)
    ax_left.xaxis.set_major_locator(MaxNLocator(integer=True))

    if ax_right:
        ax_right.set_ylabel(metric_map[ax2]['label'], color=red)
        ax_right.tick_params(axis='y', labelcolor=red)

    last_stats1 = None

    for i, model in enumerate(models):
        sub = df[df['model']==model] if model is not None else df

        # --- energy stats ---
        stats1 = sub.groupby(param_col)[metric_map[ax1]['col']].agg(['mean','std'])
        last_stats1 = stats1

        if is_numeric_dtype(stats1.index):
            xs = stats1.index.astype(float)
        else:
            xs = np.arange(len(stats1))

        mean1 = stats1['mean'].values
        std1  = stats1['std'].fillna(0).values

        # --- throughput stats if we do ax2 ---
        if ax2:
            stats2 = sub.groupby(param_col)[metric_map[ax2]['col']].agg(['mean','std'])
            mean2  = stats2['mean'].values
            std2   = stats2['std'].fillna(0).values

        # --- normalise if requested ---
        if norm1 and len(mean1):
            base1 = mean1[0]
            mean1 /= base1
            std1  /= base1
            ax_left.yaxis.set_major_formatter(
                FuncFormatter(lambda v, _: f"{v:.2f}x")
            )
        if ax2 and norm2 and len(mean2):
            base2 = mean2[0]
            mean2 /= base2
            std2  /= base2
            ax_right.yaxis.set_major_formatter(
                FuncFormatter(lambda v, _: f"{v:.2f}x")
            )

        ls  = linestyles[i % len(linestyles)]
        mk  = markers[i % len(markers)]
        lbl = str(model) if model is not None else 'all'

        # plot energy
        _plot_with_band(
            ax_left, xs, mean1, std1,
            color=blue,
            linestyle=ls,
            marker=mk,
            label=f"{lbl} (Energy)",
            alpha_line=alpha_line,
            alpha_scatter=alpha_scatter,
            alpha_band=alpha_band,
            plot_band=plot_band,
            plot_raw=plot_raw
        )
        if add_baseline_energy and len(mean1):
            base = 1.0 if norm1 else mean1[0]
            ax_left.axhline(base, linestyle=':', color=blue, alpha=0.4)
            ax_left.text(xs[-1], base,
                         f"Energy baseline ({xlabel}: {stats1.index[0]})",
                         ha='right', va='bottom',
                         fontsize='small', color=blue, alpha=0.4)

        # plot throughput if asked
        if ax2:
            _plot_with_band(
                ax_right, xs, mean2, std2,
                color=red,
                linestyle=ls,
                marker=mk,
                label=f"{lbl} (Throughput)",
                alpha_line=alpha_line,
                alpha_scatter=alpha_scatter,
                alpha_band=alpha_band,
                plot_band=plot_band,
                plot_raw=plot_raw
            )
            if add_baseline_throughput and len(mean2):
                base2 = 1.0 if norm2 else mean2[0]
                ax_right.axhline(base2, linestyle=':', color=red, alpha=0.4)
                ax_right.text(xs[-1], base2,
                              f"Throughput baseline ({xlabel}: {stats2.index[0]})",
                              ha='right', va='bottom',
                              fontsize='small', color=red, alpha=0.4)

    # if x-axis is categorical, relabel the ticks
    if last_stats1 is not None and not is_numeric_dtype(last_stats1.index):
        ax_left.set_xticks(np.arange(len(last_stats1)))
        ax_left.set_xticklabels(last_stats1.index.tolist())

    # tidy up legend & grid
    handles, labels = ax_left.get_legend_handles_labels()
    if ax2:
        h2, l2 = ax_right.get_legend_handles_labels()
        handles += h2; labels += l2
    ax_left.legend(handles, labels, loc='best')

    # if there is NO second axis, hide the right spine completely
    if ax_right is None:
        ax_left.spines['right'].set_visible(False)

    ax_left.grid(True, axis='y', linestyle='--', alpha=0.4)
    ax_left.grid(True, axis='x', linestyle=':',  alpha=0.2)

    # adaptive title
    title = metric_map[ax1]['label']
    if ax2:
        title += " & " + metric_map[ax2]['label']
    title += f" vs {xlabel}"
    plt.title(title)

    plt.tight_layout()
    plt.show()


# ---------------------------
# Wrappers
# ---------------------------
def plot_num_processes(dfs, **kwargs):
    df = dfs['num_processes'].copy()
    df['model'] = df.get('model', None)
    df['num_processes'] = df['num_processes'].astype(int)
    plot_param_vs_metric(df, 'num_processes', **kwargs)


def plot_batching(dfs, **kwargs):
    df = dfs['batching'].copy().rename(
        columns={'batch_size___fixed_batching': 'batch_size'}
    )
    df['model'] = df.get('model', None)
    plot_param_vs_metric(df, 'batch_size', **kwargs)


def plot_precision(dfs, **kwargs):
    """
    Plot Energy & optional Throughput vs precision.
    One line per original model.
    """
    df = dfs.get('precis')
    if df is None:
        print("precis DataFrame not found in dfs.")
        return

    df = df.copy()
    if 'model' not in df.columns:
        df['model'] = None

    def _mode(r):
        if r.get('load_in_4bit'):      return 'INT4'
        if r.get('load_in_8bit'):      return 'INT8'
        if r.get('fp_precision')=='torch.float16': return 'FP16'
        return 'FP32'

    df['precision'] = pd.Categorical(
        df.apply(_mode, axis=1),
        categories=['FP32','FP16','INT8','INT4'],
        ordered=True
    )

    plot_param_vs_metric(
        df,
        param_col='precision',
        **kwargs
    )
