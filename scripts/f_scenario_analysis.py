from typing import Dict, List

def compare_energy_to_appliances(
    df,
    avg_len_tokens: int = 300,
    appliances_kwh: Dict[str, float] = None,
    models: List[str] = None
):
    """
    Compare energy consumption per response against appliances across scenarios:
      1) Full dataset
      2) Without outliers
      3) By config_name means
      4) Groups: Realistic vs Artificial
      5) Comparison: Realistic vs Artificial means

    Under scenarios 1 & 2, group outputs by appliance, listing:
      - # of responses to match appliance (worst-case)
      - # of responses to match appliance (best-case)
      - # of responses to match appliance (difference)
      - # of responses to match appliance (mean-case)

    Parameters:
    - df: DataFrame with columns ['model','config_name','energy_per_token_kwh']
    - avg_len_tokens: average tokens per response
    - appliances_kwh: mapping appliance -> kWh
    - models: list of models to include
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

    # Determine models
    available = df['model'].unique()
    models = [m for m in (models or available) if m in available]

    print(f"== ASSUMING AVERAGE LENGTH: {avg_len_tokens} TOKENS ==")
    print(f"\nModels: {models}")

    for model in models:
        print(f"\n=== Model: {model} ===")
        mdf = df[df['model'] == model]

        # Scenario subsets
        full = mdf
        clean = mdf[abs(mdf['energy_per_token_kwh'] - mdf['energy_per_token_kwh'].mean()) <= 3 * mdf['energy_per_token_kwh'].std()]

        for label, subset in [("Full dataset", full), ("Without outliers", clean)]:
            print(f"\n-- Scenario: {label} ({len(subset)} obs) --")
            e_max = subset['energy_per_token_kwh'].max() * avg_len_tokens
            e_min = subset['energy_per_token_kwh'].min() * avg_len_tokens
            e_mean = subset['energy_per_token_kwh'].mean() * avg_len_tokens
            diff = e_max - e_min

            # Group by appliance
            print("# of responses to match...")
            for app, kwh in appliances_kwh.items():
                print(f"    ...{app}")
                print(f"            worst-case:     {kwh / e_max:.2f}")
                print(f"            best-case:      {kwh / e_min:.2f}")
                print(f"            diff:           {kwh / diff:.2f}")
                print(f"            mean-case:      {kwh / e_mean:.2f}")
            print()

        # 3: by config_name means
        print(f"-- Scenario: By config_name means --")
        for cfg, group in mdf.groupby('config_name'):
            mean_e = group['energy_per_token_kwh'].mean() * avg_len_tokens
            print(f"Config: {cfg} ({len(group)} obs) → mean energy for a given {avg_len_tokens}-token response: {mean_e:.5f} kWh")
            print("mean # of responses to match...")
            for app, kwh in appliances_kwh.items():
                print(f"            ...{app}: {kwh / mean_e:.2f}")
        print()

        # 4: group summaries
        for label, grp in [("Realistic", mdf[mdf['config_name'].str.startswith('R')]),
                           ("Artificial", mdf[mdf['config_name'].str.startswith('A')])]:
            mean_e = grp['energy_per_token_kwh'].mean() * avg_len_tokens
            print(f"-- Scenario: Group {label} ({len(grp)} obs) --")
            print(f"Mean energy for a given {avg_len_tokens}-token response: {mean_e:.5f} kWh")
            print("grouped-mean # of responses to match...")
            for app, kwh in appliances_kwh.items():
                print(f"            ...{app}: {kwh / mean_e:.2f}")
            print()

        # 5: compare group means
        real = mdf[mdf['config_name'].str.startswith('R')]
        art  = mdf[mdf['config_name'].str.startswith('A')]
        e_real = real['energy_per_token_kwh'].mean() * avg_len_tokens
        e_art = art['energy_per_token_kwh'].mean() * avg_len_tokens
        diff = e_real - e_art
        ratio = e_real / e_art if e_art > 0 else float('inf')
        print(f"-- Scenario: Realistic vs Artificial --")
        print(f"Realistic mean energy:  {e_real:.5f} kWh")
        print(f"Artificial mean energy: {e_art:.5f} kWh")
        print(f"Difference:             {diff:.5f} kWh")
        print(f"Ratio (Realistic/Artificial): {ratio:.2f}x")
        for app, kwh in appliances_kwh.items():
            print(f"# of responses to match {app} (Realistic mean):  {kwh / e_real:.2f}")
            print(f"# of responses to match {app} (Artificial mean): {kwh / e_art:.2f}")
            print(f"# of responses to match {app} (group diff):      {kwh / diff:.2f}")
        print()
    