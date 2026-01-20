#!/usr/bin/env python3
"""Aggregate Phase 1 experiment results into a single DataFrame."""

import json
from pathlib import Path
import pandas as pd

RESULTS_DIR = Path(__file__).parent / "results"


def extract_metrics(json_path: Path) -> dict | None:
    """Extract key metrics from a multi-cycle result file."""
    try:
        with open(json_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error reading {json_path}: {e}")
        return None

    # Extract path components for categorisation
    parts = json_path.relative_to(RESULTS_DIR).parts
    phase = parts[0]  # e.g., "1a_sanity"
    config_name = parts[1]  # e.g., "baseline"

    stats = data.get("statistics", {})
    config = data.get("effective_config", {})

    # Get first cycle for detailed config info
    cycle_results = data.get("cycle_results", [])
    first_cycle = cycle_results[0] if cycle_results else {}
    process_results = first_cycle.get("process_results", [])
    first_process = process_results[0] if process_results else {}

    return {
        # Categorisation
        "phase": phase,
        "config_name": config_name,
        "experiment_id": data.get("experiment_id"),

        # Core metrics
        "energy_mean_j": stats.get("energy_mean_j"),
        "energy_std_j": stats.get("energy_std_j"),
        "energy_cv": stats.get("energy_cv"),
        "throughput_mean_tps": stats.get("throughput_mean_tps"),
        "throughput_std_tps": stats.get("throughput_std_tps"),
        "throughput_cv": stats.get("throughput_cv"),
        "efficiency_mean_tpj": stats.get("efficiency_mean_tpj"),
        "latency_mean_ms": stats.get("latency_mean_ms"),

        # Tokens
        "total_tokens": first_cycle.get("total_tokens"),

        # Config details
        "backend": data.get("backend"),
        "model_name": config.get("model_name"),
        "num_cycles": data.get("num_cycles"),
        "num_prompts": config.get("num_input_prompts"),
        "max_input_tokens": config.get("max_input_tokens"),
        "max_output_tokens": config.get("max_output_tokens"),

        # Batching
        "batch_size": config.get("batching_options", {}).get("batch_size"),
        "batch_strategy": config.get("batching_options", {}).get("strategy"),

        # Precision & Quantisation
        "fp_precision": config.get("fp_precision"),
        "quantization": config.get("quantization_config", {}).get("quantization"),
        "load_in_4bit": config.get("quantization_config", {}).get("load_in_4bit"),
        "load_in_8bit": config.get("quantization_config", {}).get("load_in_8bit"),

        # Sharding
        "sharding_strategy": config.get("sharding_config", {}).get("strategy"),
        "num_shards": config.get("sharding_config", {}).get("num_shards"),
        "num_processes": config.get("num_processes"),
        "num_gpus": len(config.get("gpu_list", [1])),

        # Decoder
        "temperature": config.get("decoder_config", {}).get("temperature"),
        "do_sample": config.get("decoder_config", {}).get("do_sample"),
        "top_p": config.get("decoder_config", {}).get("top_p"),
        "top_k": config.get("decoder_config", {}).get("top_k"),
        "decoder_preset": config.get("decoder_config", {}).get("preset"),

        # Traffic simulation
        "latency_enabled": config.get("latency_simulation", {}).get("enabled"),
        "latency_mode": config.get("latency_simulation", {}).get("mode"),
        "target_qps": config.get("latency_simulation", {}).get("target_qps"),

        # Hardware
        "gpu_name": first_process.get("gpu_name"),
    }


def aggregate_all_results() -> pd.DataFrame:
    """Find and aggregate all multi-cycle result files."""
    results = []

    for json_path in RESULTS_DIR.rglob("multi_cycle/*.json"):
        metrics = extract_metrics(json_path)
        if metrics:
            results.append(metrics)

    df = pd.DataFrame(results)

    # Sort by phase and config name
    df = df.sort_values(["phase", "config_name"]).reset_index(drop=True)

    return df


def main():
    print("Aggregating Phase 1 results...")
    df = aggregate_all_results()

    print(f"\nFound {len(df)} experiments:")
    print(df.groupby("phase").size())

    # Save to CSV
    output_path = RESULTS_DIR / "aggregated" / "phase1_summary.csv"
    output_path.parent.mkdir(exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print("PHASE 1 RESULTS SUMMARY")
    print("=" * 80)

    summary_cols = [
        "phase", "config_name", "backend",
        "throughput_mean_tps", "energy_mean_j", "efficiency_mean_tpj",
        "throughput_cv", "energy_cv"
    ]

    print(df[summary_cols].to_string(index=False))

    return df


if __name__ == "__main__":
    df = main()
