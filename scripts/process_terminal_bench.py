#!/usr/bin/env python3
"""
Process Terminal Bench evaluation results into standardized format.

A standalone Python script [process_terminal_bench.py](process_terminal_bench.py) is provided to convert
Terminal Bench raw evaluation data from [terminal-bench-core@0.1.1](https://github.com/laude-institute/terminal-bench-leaderboard/tree/main/results/terminal-bench-core%400.1.1) into the standardized format used by eval-arena.

The function expects raw data in this directory structure:
```
raw-data/terminal-bench-core@0.1.1/
├── YYYYMMDD_<agent-name>_<model-name>/
│   ├── YYYY-MM-DD__HH-MM-SS__results.json  (5 files, one per run)
│   └── ...
└── YYYYMMDD_<agent-name>_<model-name>/
    ├── <run-name-1>/
    │   └── results.json  (aggregated results for this run)
    ├── <run-name-2>/
    │   └── results.json
    └── ...
```
"""

from collections import defaultdict
import json
import glob
import numpy as np
import pandas as pd
import os

def process_terminal_bench(raw_data_dir='raw-data/terminal-bench-core@0.1.1'):
    """
    Convert terminal-bench-core@0.1.1 evaluation results into the standardized format used by eval-arena.

    Output will be saved to 'data/terminal-bench.jsonl'
    """

    records = []

    for exp_dir in sorted(glob.glob(f"{raw_data_dir}/*")):
        if not os.path.isdir(exp_dir):
            continue

        exp_name = os.path.basename(exp_dir)

        # Use the experiment folder name as the model identifier
        model_full_name = exp_name
        print(f"Processing {exp_name}...")

        # Find all results.json files in this experiment directory
        result_files = []

        # Case 1: Aggregated results.json at the root of the experiment directory
        # (e.g., orchestrator_claude-4-sonnet has only one run with single aggregated results.json)
        root_results = os.path.join(exp_dir, 'results.json')
        if os.path.exists(root_results):
            result_files.append(root_results)

        # Case 2: Direct results files in the experiment directory
        # (e.g., <experiment_name>/<run_id>_results.json)
        if not result_files:
            direct_results = glob.glob(f"{exp_dir}/*_results.json")
            result_files.extend(direct_results)

        # Case 3: Aggregated results.json in immediate subdirectories (run directories)
        # (e.g., <experiment_name>/<run_id>/results.json)
        if not result_files:
            for subdir in glob.glob(f"{exp_dir}/*/"):
                results_file = os.path.join(subdir, 'results.json')
                if os.path.exists(results_file):
                    result_files.append(results_file)

        if not result_files:
            print(f"  Warning: No results files found in {exp_name}")
            continue

        print(f"  Found {len(result_files)} result files")

        # Track task results across runs for pass1 calculation
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
            # Calculate pass1 as the success rate across all runs
            pass1 = np.mean(results_list)
            count = len(results_list)

            records.append({
                'benchmark_id': 'terminal-bench',
                'model': model_full_name,
                'example_id': task_id,
                'pass1': pass1,
                'count': count
            })

    # Create DataFrame
    df = pd.DataFrame(records)

    if len(df) == 0:
        print("\nWarning: No records were created!")
        return df

    print(f"\nProcessed {len(df)} task results")
    print(f"Unique models: {df['model'].nunique()}")
    print(f"Unique tasks: {df['example_id'].nunique()}")

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
