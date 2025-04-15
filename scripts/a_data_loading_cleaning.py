import pandas as pd
import re

# -----------------------------------------------------------------
# 1. Define helper functions to clean column names and resolve duplicates
# -----------------------------------------------------------------
def clean_column(col: str) -> str:
    """
    Cleans an individual column name:
      - Strips whitespace, replaces non-standard quotes,
      - Applies special column name mappings,
      - Removes the 'variables_' prefix,
      - Simplifies per-process column names using regex patterns.
    """
    col = col.strip()
    col = col.replace("“", "\"").replace("”", "\"")
    
    # Special mappings for some columns.
    special_mappings = {
        "setup_cpu_model": "cpu_model",
        "setup_gpu_model": "gpu_model",
        "model_architecture_total_params": "total_params",
        "model_architecture_architecture": "model_arch"
    }
    if col in special_mappings:
        return special_mappings[col]
    
    # Remove the 'variables_' prefix if it exists.
    if col.startswith("variables_"):
        col = col[len("variables_"):]
    
    # Check for per-process metric patterns (e.g., cpu_power_process_0)
    per_process_patterns = [
        r'(cpu_power_process_\d+)',
        r'(gpu_power_process_\d+)',
        r'(ram_power_process_\d+)',
        r'(cpu_energy_process_\d+)',
        r'(gpu_energy_process_\d+)',
        r'(ram_energy_process_\d+)',
        r'(total_energy_kwh_process_\d+)',
        r'(total_energy_joules_process_\d+)'
    ]
    for pattern in per_process_patterns:
        match = re.search(pattern, col)
        if match:
            return match.group(1)
    
    # For non-per-process columns, search for a known token in the cleaned column.
    tokens = [
        "config_name", "experiment_id", "date_time", "model", "is_encoder_decoder",
        "task_type", "available_gpu_count", "gpu_model", "available_cpu_count", "cpu_model",
        "os", "python_version", "country", "region", "fsdp_use_orig_params", "fsdp_cpu_offload",
        "sharding_strategy", "distributed_type", "num_processes", "max_input_tokens", "max_output_tokens",
        "number_input_prompts", "decode_token_to_text", "decoder_temperature", "decoder_top_k", "decoder_top_p",
        "query_rate", "latency_simulate", "latency_delay_min", "latency_delay_max", "latency_simulate_burst",
        "latency_burst_interval", "latency_burst_size", "fp_precision", "quantization", "load_in_8bit",
        "load_in_4bit", "cached_flops_for_quantised_models", "batch_size___fixed_batching", "adaptive_batching",
        "adaptive_max_tokens", "max_batch_size___adaptive_batching", "inference_type", "backend", "total_params",
        "architecture", "total_input_tokens", "total_generated_tokens", "total_inference_time_sec", 
        "average_latency_ms_per_batch", "throughput_queries_per_sec", "throughput_tokens_per_sec", "flops",
        "gpu_current_memory_allocated_bytes", "gpu_max_memory_allocated_bytes", "gpu_current_memory_reserved_bytes",
        "gpu_max_memory_reserved_bytes", "gpu_utilization_percent", "cpu_usage_percent", "cpu_memory_usage_bytes",
        # Per-process metrics:
        "cpu_power_process_0", "gpu_power_process_0", "ram_power_process_0",
        "cpu_energy_process_0", "gpu_energy_process_0", "ram_energy_process_0",
        "total_energy_kwh_process_0", "total_energy_joules_process_0",
        # Global averages and totals:
        "cpu_power_avg", "gpu_power_avg", "ram_power_avg", "cpu_energy_total", "gpu_energy_total", "ram_energy_total",
        "total_energy_kwh", "total_energy_joules", "tokens_per_joule", "joules_per_token", "flops_per_joule", "joules_per_flop",
        "per-process_emissions"
    ]
    for token in tokens:
        if token in col:
            idx = col.find(token)
            return col[idx:]
    
    return col

def resolve_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    For columns that appear multiple times, this function selects only one instance.
    In special cases (e.g., 'adaptive_batching'), it chooses the column with the desired dtype.
    """
    seen = {}
    for idx, col in enumerate(df.columns):
        seen.setdefault(col, []).append(idx)
    
    chosen_indices = []
    for col, indices in seen.items():
        if len(indices) == 1:
            chosen_indices.append(indices[0])
        else:
            # Example special handling for boolean columns (like adaptive_batching)
            if col == "adaptive_batching":
                bool_idx = None
                for i in indices:
                    if pd.api.types.is_bool_dtype(df.iloc[:, i]):
                        bool_idx = i
                        break
                chosen_indices.append(bool_idx if bool_idx is not None else indices[0])
            else:
                # Otherwise, just keep the first occurrence.
                chosen_indices.append(indices[0])
    
    chosen_indices.sort()  # keep original order as much as possible
    return df.iloc[:, chosen_indices]

def clean_and_reorder_columns(df: pd.DataFrame, desired_order: list) -> pd.DataFrame:
    """
    Cleans the column names and then reorders the DataFrame columns according to `desired_order`.
    Any columns not specified in the order list are appended afterwards.
    """
    mapping = {col: clean_column(col) for col in df.columns}
    df = df.rename(columns=mapping)
    df = resolve_duplicates(df)
    
    # Order columns: first the ones in desired_order, then the rest.
    ordered_cols = [col for col in desired_order if col in df.columns]
    remaining_cols = [col for col in df.columns if col not in desired_order]
    final_order = ordered_cols + remaining_cols
    return df[final_order]

# -----------------------------------------------------------------
# 2. Define  preferred column order (this can be tailored to your needs)
# -----------------------------------------------------------------
desired_order = [
    "config_name",
    "experiment_id",
    "date_time",
    "model",
    # num_process
    "num_processes",
    # batching
    "batch_size___fixed_batching",
    # decodeer
    "decoder_temperature",
    "decoder_top_k",
    "decoder_top_p",
    # latency
    "latency_simulation_simulate"
    "latency_simulation_delay_max",
    "latency_simulation_delay_min",
    "latency_simulation_simulate_burst",
    "latency_simulation_burst_size",
    "latency_simulation_burst_interval",
    # precision / quantisation
    "fp_precision",
    "quantization",
    "load_in_8bit",
    "load_in_4bit",
    "cached_flops_for_quantised_models",
    
    # UNUSED PARAMS
    "sharding_strategy",
    "sharding_config_fsdp_config_use_orig_params",
    "sharding_config_fsdp_config_cpu_offload",
    "adaptive_batching",
    "adaptive_max_tokens",
    "query_rate",
    "total_input_tokens",
    "total_generated_tokens"
    
    # CONSTANT SETUP ====
    "date_time",
    "is_encoder_decoder",
    "task_type",
    "available_gpu_count",
    "gpu_model",
    "available_cpu_count",
    "cpu_model",
    "os",
    "python_version",
    "country",
    "region",
    "distributed_type",
    "decode_token_to_text",
    "inference_type",
    "backend",
    "total_params",
    "model_arch",

    # Validation (should be same):
    "max_input_tokens",
    "max_output_tokens",
    "number_input_prompts",
    
    # RESULTS =====
    # energy
    "total_energy_kwh",
    "total_energy_joules",
    # FLOPS
    "flops",
    "tokens_per_joule",
    "joules_per_token",
    "flops_per_joule",
    "joules_per_flop",
    "total_inference_time_sec", 
    # inference performance
    "average_latency_ms_per_batch",
    "throughput_queries_per_sec",
    "throughput_tokens_per_sec",
    # CPU utilization
    "cpu_usage_percent",
    "cpu_memory_usage_bytes",
    # GPU utilization
    "gpu_utilization_percent_0", "gpu_utilization_percent_1", "gpu_utilization_percent_2", "gpu_utilization_percent_3",
    # Compute mem
    "gpu_current_memory_allocated_bytes",
    "gpu_max_memory_allocated_bytes",
    "gpu_current_memory_reserved_bytes",
    "gpu_max_memory_reserved_bytes",
    # Per-process metrics:
    "cpu_power_process_0", "cpu_power_process_1", "cpu_power_process_2", "cpu_power_process_3",
    "gpu_power_process_0", "gpu_power_process_1", "gpu_power_process_2", "gpu_power_process_3",
    "ram_power_process_0", "ram_power_process_1", "ram_power_process_2", "ram_power_process_3",
    "cpu_energy_process_0", "cpu_energy_process_1", "cpu_energy_process_2", "cpu_energy_process_3",
    "gpu_energy_process_0", "gpu_energy_process_1", "gpu_energy_process_2", "gpu_energy_process_3",
    "ram_energy_process_0", "ram_energy_process_1", "ram_energy_process_2", "ram_energy_process_3",
    "total_energy_kwh_process_0", "total_energy_kwh_process_1", "total_energy_kwh_process_2", "total_energy_kwh_process_3",
    "total_energy_joules_process_0", "total_energy_joules_process_1", "total_energy_joules_process_2", "total_energy_joules_process_3",
    # Global averages and totals:
    "cpu_power_avg",
    "gpu_power_avg",
    "ram_power_avg",
    "cpu_energy_total",
    "gpu_energy_total",
    "ram_energy_total",
    # per-process_emsisisons
    "per-process_emissions_0", "per-process_emissions_1", "per-process_emissions_2","per-process_emissions_3"
]

# -----------------------------------------------------------------
# 3. Load controlled experiments CSV and clean it
# -----------------------------------------------------------------
df_controlled = pd.read_csv("results/controlled_results.csv")
df_controlled_cleaned = clean_and_reorder_columns(df_controlled, desired_order)

print("Columns in cleaned DataFrame:")
print(df_controlled_cleaned.columns)
print("\nSummary statistics:")
print(df_controlled_cleaned.describe())
print("--" * 50)

# -----------------------------------------------------------------
# 4. Create Derived Columns:
#    - flops_per_token: FLOPs divided by total generated tokens.
#    - energy_per_token_kwh: Total energy (kWh) divided by total generated tokens.
#    - divergence_energy_flops: Ratio of energy-per-token to flops-per-token.
# -----------------------------------------------------------------
df_controlled_cleaned['flops_per_token'] = (
    df_controlled_cleaned['flops'] / df_controlled_cleaned['total_generated_tokens']
)
df_controlled_cleaned['energy_per_token_kwh'] = (
    df_controlled_cleaned['total_energy_kwh'] / df_controlled_cleaned['total_generated_tokens']
)
df_controlled_cleaned['divergence_energy_flops'] = (
    df_controlled_cleaned['energy_per_token_kwh'] / df_controlled_cleaned['flops_per_token']
)

# -----------------------------------------------------------------
# 5. Verify that FLOPs are constant across runs (as per your experimental design)
# -----------------------------------------------------------------
unique_flops = df_controlled_cleaned['flops'].unique()
print("Unique FLOPs values in controlled experiments:", unique_flops)
print("--" * 50)


# -----------------------------------------------------------------
# 6. drop columns I don't need
# -----------------------------------------------------------------

columns_to_drop = [
    "sharding_strategy",
    "sharding_config_fsdp_config_use_orig_params",
    "sharding_config_fsdp_config_cpu_offload",
    "adaptive_batching",
    "adaptive_max_tokens",
    "query_rate",
    "is_encoder_decoder",
    "task_type",
    "available_gpu_count",
    "gpu_model",
    "available_cpu_count",
    "cpu_model",
    "os",
    "python_version",
    "country",
    "region",
    "distributed_type",
    "decode_token_to_text",
    "inference_type",
    "backend",
    "model_arch",
    "gpu_current_memory_allocated_bytes",
    "gpu_max_memory_allocated_bytes",
    "gpu_current_memory_reserved_bytes",
    "gpu_max_memory_reserved_bytes",
    "per-process_emissions_0", "per-process_emissions_1", "per-process_emissions_2","per-process_emissions_3" # OR IS THIS NICE TO HAVE?
]

df_controlled_cleaned_dropped = df_controlled_cleaned.drop(columns=columns_to_drop, errors='ignore')
print(f"Dropped columns: {columns_to_drop}")
print("Columns after dropping:")
print(df_controlled_cleaned_dropped.columns)
print("--" * 50)

# second round of dropping (at some point come back to these)
columns_to_drop_2 = [
    "cpu_usage_percent",
    "cpu_memory_usage_bytes",
    "gpu_utilization_percent_0",
    "gpu_utilization_percent_1",
    "gpu_utilization_percent_2",
    "gpu_utilization_percent_3",
    "gpu_current_memory_allocated_bytes",
    "gpu_max_memory_allocated_bytes",
    "cpu_current_memory_allocated_bytes",
    "cpu_max_memory_allocated_bytes",
    "cpu_power_process_0",
    "cpu_power_process_1",
    "cpu_power_process_2",
    "cpu_power_process_3",
    "gpu_power_process_0",
    "gpu_power_process_1",
    "gpu_power_process_2",
    "gpu_power_process_3",
    "ram_power_process_0",
    "ram_power_process_1",
    "ram_power_process_2",
    "ram_power_process_3",
    "cpu_energy_process_0",
    "cpu_energy_process_1",
    "cpu_energy_process_2",
    "cpu_energy_process_3",
    "gpu_energy_process_0",
    "gpu_energy_process_1",
    "gpu_energy_process_2",
    "gpu_energy_process_3",
    "ram_energy_process_0",
    "ram_energy_process_1",
    "ram_energy_process_2",
    "ram_energy_process_3",
    "total_energy_joules_process_0",
    "total_energy_joules_process_1",
    "total_energy_joules_process_2",
    "total_energy_joules_process_3",
    "cpu_power_avg",
    "ram_energy_total",
    "models"
]

df_controlled_cleaned_dropped_2 = df_controlled_cleaned_dropped.drop(columns=columns_to_drop_2, errors='ignore')
print(f"Dropped columns: {columns_to_drop_2}")
print("Columns after dropping:")
print(df_controlled_cleaned_dropped_2.columns)
print("--" * 50)

# -----------------------------------------------------------------
# 6. if necessary identify why flops different
# -----------------------------------------------------------------
def identify_flop_differentiators(df, flops_col='flops', exclude_cols=None):
    """
    Identify columns that are constant within each FLOP group but differ between groups.
    
    Parameters:
      df (pd.DataFrame): The DataFrame containing the data.
      flops_col (str): Name of the column used for grouping the FLOPs.
      exclude_cols (list, optional): List of columns to exclude from the comparison.
      
    Returns:
      dict: A dictionary where keys are the column names that differentiate FLOP groups,
            and values are dictionaries mapping each unique FLOP value to the constant value
            observed in that group.
            
    Example output:
    {
      'config_name': {1034544128000: 'A1_Max_Throughput_Exploit', 
                      16949970993152: 'A5_Parallel_Overdrive'},
      'some_other_col': {1034544128000: 'value1', 
                         16949970993152: 'value2'}
    }
    """
    # Optionally exclude some columns, including the flops column itself.
    if exclude_cols is None:
        exclude_cols = []
    exclude_cols = set(exclude_cols + [flops_col])
    
    differentiators = {}
    unique_flops = df[flops_col].unique()
    
    # Loop over each column in df excluding the ones in exclude_cols.
    for col in df.columns:
        if col in exclude_cols:
            continue
        
        # For each group (by flops), get the unique values for this column.
        group_values = {}
        valid = True  # assume column is constant per group unless we find more than one unique value.
        for flop in unique_flops:
            values = df[df[flops_col] == flop][col].unique()
            if len(values) == 1:
                group_values[flop] = values[0]
            else:
                # If any FLOP group has more than one value, then this column doesn't differentiate consistently.
                valid = False
                break
        # Check if the column is valid and if it truly differentiates between groups.
        if valid and len(set(group_values.values())) > 1:
            differentiators[col] = group_values
    
    return differentiators

# implement this if necessary
# -----------------------------------------------------------------