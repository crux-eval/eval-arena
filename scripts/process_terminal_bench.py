#!/usr/bin/env python3
"""
Process Terminal Bench evaluation results into standardized format.
"""

from collections import defaultdict
import json
import glob
import numpy as np
import pandas as pd
import os
import re

def process_terminal_bench(raw_data_dir='raw-data/terminal-bench-core@0.1.1'):
    """
    Process Terminal Bench evaluation results as of 1/14/2026.

    See TERMINAL_BENCH_PROCESSING.md for processing guide.
    """

    records = []

    # Pattern to parse directory names: YYYYMMDD_<agent-name>_<model-name>
    dir_pattern = re.compile(r'^\d{8}_(.+)_(.+)$')

    for exp_dir in sorted(glob.glob(f"{raw_data_dir}/*")):
        if not os.path.isdir(exp_dir):
            continue

        exp_name = os.path.basename(exp_dir)

        # Special case: ob1-09-10-25 directory
        if exp_name == 'ob1-09-10-25':
            agent_name = 'ob1_agent'
            model_name = 'sdk'
            model_full_name = 'ob1_agent_sdk'
            print(f"Processing {exp_name} (special case: ob1_agent_sdk)...")
        else:
            # Parse agent and model from directory name
            match = dir_pattern.match(exp_name)
            if not match:
                print(f"Skipping {exp_name}: doesn't match expected pattern")
                continue

            agent_name = match.group(1)
            model_name = match.group(2)
            model_full_name = f"{agent_name}_{model_name}"
            print(f"Processing {exp_name}...")

        # Find all results.json files in this experiment directory
        result_files = []

        # Case 1: Aggregated results.json at the root of the experiment directory
        # (e.g., orchestrator_claude-4-sonnet has only one run with single aggregated results.json)
        root_results = os.path.join(exp_dir, 'results.json')
        if os.path.exists(root_results):
            result_files.append(root_results)

        # Case 2: Direct results files in the experiment directory (timestamped files)
        if not result_files:
            direct_results = glob.glob(f"{exp_dir}/*_results.json")
            result_files.extend(direct_results)

        # Case 3: Aggregated results.json in immediate subdirectories (run directories)
        if not result_files:
            for subdir in glob.glob(f"{exp_dir}/*/"):
                results_file = os.path.join(subdir, 'results.json')
                if os.path.exists(results_file):
                    result_files.append(results_file)

        if not result_files:
            print(f"  Warning: No results files found in {exp_name}")
            continue

        print(f"  Found {len(result_files)} result files")

        # Track task results across runs for pass@k calculation
        task_results = defaultdict(list)  # task_id -> list of pass/fail (1/0)

        for result_file in result_files:
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)

                # Process each task in this run
                for task_result in data.get('results', []):
                    task_id = task_result.get('task_id')
                    is_resolved = task_result.get('is_resolved', False)

                    if task_id:
                        task_results[task_id].append(1 if is_resolved else 0)

            except Exception as e:
                print(f"  Error processing {result_file}: {e}")
                continue

        # Create records for each task
        for task_id, results_list in task_results.items():
            # Calculate pass@1 as the success rate across all runs
            pass_at_1 = np.mean(results_list)
            count = len(results_list)

            records.append({
                'benchmark_id': 'terminal-bench',
                'model': model_full_name,
                'example_id': task_id,
                'pass1': pass_at_1,
                'count': count,
                'agent': agent_name,
                'base_model': model_name
            })

    # Create DataFrame
    df = pd.DataFrame(records)

    if len(df) == 0:
        print("\nWarning: No records were created!")
        return df

    print(f"\nProcessed {len(df)} task results (before deduplication)")

    # Deduplicate: merge records with same model + example_id
    # This handles cases where the same model was run multiple times
    df_grouped = df.groupby(['benchmark_id', 'model', 'example_id', 'agent', 'base_model']).agg({
        'pass1': lambda x: np.mean(x),  # Average pass1 across duplicate experiments
        'count': lambda x: np.sum(x)     # Sum counts from duplicate experiments
    }).reset_index()

    df = df_grouped

    print(f"After deduplication: {len(df)} task results")
    print(f"Unique models: {df['model'].nunique()}")
    print(f"Unique tasks: {df['example_id'].nunique()}")

    # Display summary statistics
    print("\nSummary by model:")
    summary = df.groupby('model').agg({
        'pass1': 'mean',
        'example_id': 'count',
        'count': 'first'
    }).round(3)
    summary.columns = ['avg_pass1', 'n_tasks', 'n_runs']
    print(summary.sort_values('avg_pass1', ascending=False))

    return df


if __name__ == '__main__':
    # Process the data
    print("Starting Terminal Bench data processing...")
    print("=" * 60)

    df_terminal_bench = process_terminal_bench()

    # Save to file
    if len(df_terminal_bench) > 0:
        output_file = 'data/terminal-bench.jsonl'

        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)

        # Select standard fields for output
        df_output = df_terminal_bench[['benchmark_id', 'model', 'example_id', 'pass1', 'count']]
        df_output.to_json(output_file, orient='records', lines=True)

        print(f"\n{'=' * 60}")
        print(f"Saved {len(df_output)} records to {output_file}")
        print("=" * 60)

        # Display first few records
        print("\nFirst 10 records:")
        print(df_output.head(10).to_string())

        print("\n✓ Processing complete!")
    else:
        print("\n✗ No data was processed. Please check your raw data directory.")
