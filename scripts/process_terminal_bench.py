#!/usr/bin/env python3
"""
Process Terminal Bench evaluation results into standardized format.

Supports both Terminal Bench 1.0 and 2.0 data formats.

## 1.0 Format ([terminal-bench-core@0.1.1](https://github.com/laude-institute/terminal-bench-leaderboard/tree/main/results/terminal-bench-core%400.1.1))
```
raw-data/terminal-bench/1.0/
├── YYYYMMDD_<agent-name>_<model-name>/
│   ├── YYYY-MM-DD__HH-MM-SS__results.json  (5 files, one per run)
│   └── ...
└── YYYYMMDD_<agent-name>_<model-name>/
    ├── <run-name-1>/
    │   └── results.json  (aggregated results with 'is_resolved' field)
    └── ...
```

## 2.0 Format ([terminal-bench-2-leaderboard](https://huggingface.co/datasets/alexgshaw/terminal-bench-2-leaderboard))
```
raw-data/terminal-bench/2.0/
├── <Agent>__<Model>/
│   ├── YYYY-MM-DD__HH-MM-SS/
│       └── result.json  (aggregated with reward_stats and exception_stats)
│   └── ...
└── ...
```
"""

from collections import defaultdict
import json
import glob
import numpy as np
import pandas as pd
import os


def process_terminal_bench_1_0(raw_data_dir='raw-data/terminal-bench/1.0'):
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
        root_results = os.path.join(exp_dir, 'results.json')
        if os.path.exists(root_results):
            result_files.append(root_results)

        # Case 2: Direct results files in the experiment directory
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
            pass1 = np.mean(results_list)
            count = len(results_list)

            records.append({
                'benchmark_id': 'terminal-bench-1.0',
                'model': model_full_name,
                'example_id': task_id,
                'pass1': pass1,
                'count': count
            })

    df = pd.DataFrame(records)

    if len(df) == 0:
        print("\nWarning: No records were created!")
        return df

    print(f"\nProcessed {len(df)} task results")
    print(f"Unique models: {df['model'].nunique()}")
    print(f"Unique tasks: {df['example_id'].nunique()}")

    return df


def process_terminal_bench_2_0(raw_data_dir='raw-data/terminal-bench/2.0'):
    records = []

    for model_dir in sorted(os.listdir(raw_data_dir)):
        model_path = os.path.join(raw_data_dir, model_dir)
        if not os.path.isdir(model_path) or model_dir.startswith('.'):
            continue

        print(f"Processing {model_dir}...")

        # Track task results across all runs
        task_results = defaultdict(list)  # task_name -> list of pass/fail (1.0/0.0)

        # Find all run directories (date-stamped folders)
        for run_dir in sorted(os.listdir(model_path)):
            run_path = os.path.join(model_path, run_dir)
            if not os.path.isdir(run_path):
                continue

            result_file = os.path.join(run_path, "result.json")
            if not os.path.exists(result_file):
                continue

            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)

                # Extract from stats.evals
                stats = data.get("stats", {}).get("evals", {})

                for _, eval_data in stats.items():
                    reward_stats = eval_data.get("reward_stats", {}).get("reward", {})
                    exception_stats = eval_data.get("exception_stats", {})

                    # Collect all trial IDs
                    all_trial_ids = set().union(*reward_stats.values(), *exception_stats.values())

                    # Get successful trials
                    success_ids = set(reward_stats.get("1.0", []))

                    # Mark each trial as success (1.0) or failure (0.0)
                    for trial_id in all_trial_ids:
                        parts = trial_id.rsplit("__", 1)
                        if len(parts) == 2:
                            task_name = parts[0]
                            task_results[task_name].append(1.0 if trial_id in success_ids else 0.0)

            except Exception as e:
                print(f"  Error processing {result_file}: {e}")
                continue

        print(f"  Found {len(task_results)} tasks")

        # Create records for each task
        for task_name, results_list in task_results.items():
            pass1 = np.mean(results_list)
            count = len(results_list)

            records.append({
                'benchmark_id': 'terminal-bench-2.0',
                'model': model_dir,
                'example_id': task_name,
                'pass1': pass1,
                'count': count
            })

    df = pd.DataFrame(records)

    if len(df) == 0:
        print("\nWarning: No records were created!")
        return df

    print(f"\nProcessed {len(df)} task results")
    print(f"Unique models: {df['model'].nunique()}")
    print(f"Unique tasks: {df['example_id'].nunique()}")

    return df


if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)

    if os.path.isdir('raw-data/terminal-bench/1.0'):
        print("=" * 60)
        print("Processing Terminal Bench 1.0...")
        print("=" * 60)

        df_1_0 = process_terminal_bench_1_0()

        if len(df_1_0) > 0:
            output_file = 'data/terminal-bench-1.0.jsonl'
            df_1_0.to_json(output_file, orient='records', lines=True)
            print(f"Saved {len(df_1_0)} records to {output_file}")

    if os.path.isdir('raw-data/terminal-bench/2.0'):
        print("\n" + "=" * 60)
        print("Processing Terminal Bench 2.0...")
        print("=" * 60)

        df_2_0 = process_terminal_bench_2_0()

        if len(df_2_0) > 0:
            output_file = 'data/terminal-bench-2.0.jsonl'
            df_2_0.to_json(output_file, orient='records', lines=True)
            print(f"Saved {len(df_2_0)} records to {output_file}")

    print("\n" + "=" * 60)
    print("Processing complete!")
    print("=" * 60)
