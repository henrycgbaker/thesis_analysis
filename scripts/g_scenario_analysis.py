from typing import List, Dict
import pandas as pd

def get_descriptive_stats(df: pd.DataFrame, models: List[str] = None):
    """
    Compute and print descriptive statistics on the MEAN energy_per_token_kwh per configuration
    for each model, including max/min scenarios, normalized measures, variability, and distribution metrics.
    """
    available_models = df['model'].unique()
    models = [m for m in (models or available_models) if m in available_models]

    print(f"\nModels: {models}")

    for model in models:
        model_df = df[df['model'] == model]
        if model_df.empty:
            print(f"No data for model {model}, skipping.")
            continue

        # Aggregate to mean per configuration
        grouped = (
            model_df
            .groupby('config_name')['energy_per_token_kwh']
            .mean()
            .reset_index(name='mean_energy')
        )

        # Identify max and min mean scenarios
        idx_max = grouped['mean_energy'].idxmax()
        idx_min = grouped['mean_energy'].idxmin()
        max_row = grouped.loc[idx_max]
        min_row = grouped.loc[idx_min]
        max_val = max_row['mean_energy']
        min_val = min_row['mean_energy']

        print(f"\nModel: {model}\n")
        print(f"    - Config w/ MAX Mean Energy: {max_row['config_name']} ({max_val:.4f} kWh)")
        print(f"    - Config w/ MIN Mean Energy: {min_row['config_name']} ({min_val:.4f} kWh)")

        # Range relative to overall mean of means
        overall_mean = grouped['mean_energy'].mean()
        value_range = max_val - min_val
        print(f'\n ==Standard normalisation (relative to mean):==')
        print(f"    - Energy range is {(value_range/overall_mean):.2%} of the MEAN of config means.")

        # Normalize to baseline (min mean configuration)
        grouped['norm_to_min'] = grouped['mean_energy'] / min_val
        grouped['diff_to_min_pct'] = (grouped['mean_energy'] - min_val) / min_val
        print("\n ==Normalized to baseline (min mean config):==")
        print(f"    - Energy range is {(value_range/min_val):.2%} of the MIN of config means.")
        print(grouped[['config_name', 'norm_to_min', 'diff_to_min_pct']])

        # Variability measures on config means
        std_means = grouped['mean_energy'].std()
        cv = std_means / overall_mean
        print(f"\nCoefficient of variation of config means: {cv:.2%}")
        print(f"Standard deviation of config means: {std_means:.4f} kWh ({std_means/overall_mean:.2%} of mean).")

        # Additional distribution metrics on config means
        q1 = grouped['mean_energy'].quantile(0.25)
        median = grouped['mean_energy'].median()
        q3 = grouped['mean_energy'].quantile(0.75)
        iqr = q3 - q1
        skew = grouped['mean_energy'].skew()
        kurt = grouped['mean_energy'].kurtosis()
        count = grouped['mean_energy'].count()

        print(f"\nAdditional metrics (based on config means) for {model}:")
        print(f"        Count: {count}")
        print(f"        Quartiles (25th/50th/75th): {q1:.4f}, {median:.4f}, {q3:.4f}")
        print(f"        IQR: {iqr:.4f}")
        print(f"        Skewness: {skew:.2f}, Kurtosis: {kurt:.2f}")
        print("----")


def compare_energy_to_appliances(
    df,
    avg_len_tokens: int = 300,
    appliances_kwh: Dict[str, float] = None,
    models: List[str] = None
):
    """
    Scenarios:
    1) Full config means
    2) Config means without outliers
    3) By config_name details
    4) Groups: Realistic vs Artificial configs
    5) Comparison: Realistic vs Artificial group means
    """
    # Default appliance energy usages
    if appliances_kwh is None:
        appliances_kwh = {
            "iPhone_charge": 0.01,
            "laptop_charge": 0.05,
            "wifi_router_24h": 0.024,
            "streaming_1hr": 0.05,
            "google_search": 0.0003,
            "kettle": 0.1,
            "shower": 2.6
        }

    # Determine models to include
    available = df['model'].unique()
    models = [m for m in (models or available) if m in available]

    print(f"== ASSUMING AVERAGE LENGTH: {avg_len_tokens} TOKENS ==")
    print(f"Models: {models}\n")

    for model in models:
        print(f"=== Model: {model} ===")
        mdf = df[df['model'] == model]
        if mdf.empty:
            print("No data for this model.\n")
            continue

        # Compute mean energy per config
        config_stats = (
            mdf
            .groupby('config_name')['energy_per_token_kwh']
            .mean()
            .reset_index(name='mean_energy')
        )
        # Add per-response energy
        config_stats['response_energy'] = config_stats['mean_energy'] * avg_len_tokens

        # Scenario 1 & 2 on config means
        mean_vals = config_stats['response_energy']
        std_vals = mean_vals.std()
        full_cfg = config_stats.copy()
        clean_cfg = config_stats[abs(mean_vals - mean_vals.mean()) <= 3 * std_vals]

        for label, subset in [("Full config means", full_cfg), ("Without outlier configs", clean_cfg)]:
            n_cfg = len(subset)
            e_max = subset['response_energy'].max()
            e_min = subset['response_energy'].min()
            e_mean = subset['response_energy'].mean()
            diff = e_max - e_min
            ratio = e_max / e_min if e_min > 0 else float('inf')

            print(f"-- Scenario: {label} ({n_cfg} configs) --")
            print(f"Overall ratio (max/min): {ratio:.2f}")
            print("# responses to match appliance (worst/best/diff/mean):")
            for app, kwh in appliances_kwh.items():
                wc = kwh / e_max if e_max > 0 else float('inf')
                bc = kwh / e_min if e_min > 0 else float('inf')
                dc = kwh / diff if diff > 0 else float('inf')
                mc = kwh / e_mean if e_mean > 0 else float('inf')
                print(f"    {app}: worst {wc:.2f}, best {bc:.2f}, diff {dc:.2f}, mean {mc:.2f}")
            print()

        # Scenario 3: Detailed per-config responses
        print("-- Scenario: By config_name details --")
        for _, row in config_stats.iterrows():
            resp_e = row['response_energy']
            print(f"Config: {row['config_name']} → {resp_e:.5f} kWh per response")
            for app, kwh in appliances_kwh.items():
                print(f"    {app}: {kwh / resp_e:.2f} responses")
        print()

        # Scenario 4: Group summaries (Realistic vs Artificial)
        groups = {
            'Realistic': config_stats[config_stats['config_name'].str.startswith('R')],
            'Artificial': config_stats[config_stats['config_name'].str.startswith('A')]
        }
        for label, grp in groups.items():
            m_energy = grp['response_energy'].mean() if not grp.empty else 0
            n_cfg = len(grp)
            print(f"-- Scenario: Group {label} ({n_cfg} configs) --")
            print(f"Mean kWh per response: {m_energy:.5f}")
            for app, kwh in appliances_kwh.items():
                val = kwh / m_energy if m_energy > 0 else float('inf')
                print(f"    {app}: {val:.2f} responses")
            print()

        # Scenario 5: Compare group means
        e_real = groups['Realistic']['response_energy'].mean() if not groups['Realistic'].empty else 0
        e_art = groups['Artificial']['response_energy'].mean() if not groups['Artificial'].empty else 0
        diff_ga = e_real - e_art
        ratio_ga = e_real / e_art if e_art > 0 else float('inf')
        print("-- Scenario: Realistic vs Artificial --")
        print(f"Realistic mean: {e_real:.5f} kWh, Artificial mean: {e_art:.5f} kWh")
        print(f"Difference: {diff_ga:.5f} kWh, Ratio: {ratio_ga:.2f}x")
        for app, kwh in appliances_kwh.items():
            real_c = kwh / e_real if e_real > 0 else float('inf')
            art_c = kwh / e_art if e_art > 0 else float('inf')
            diff_c = kwh / diff_ga if diff_ga > 0 else float('inf')
            print(f"    {app}: Realistic {real_c:.2f}, Artificial {art_c:.2f}, Diff {diff_c:.2f}")
        print()
    