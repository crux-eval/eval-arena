#!/usr/bin/env python3
"""
Process SWE-bench Bash-Only evaluation results into standardized format.

Raw data source: [SWE-bench/experiments](https://github.com/SWE-bench/experiments/tree/main/evaluation/bash-only)

The script expects raw data in this directory structure:
```
raw-data/swebench-bash/
├── <run-name>/
│   └── per_instance_details.json
└── <run-name>/
    └── per_instance_details.json
```

Each per_instance_details.json contains:
```json
{
    "<instance_id>": {
        "cost": float,
        "api_calls": int,
        "resolved": bool
    },
    ...
}
```

Output will be saved to:
- data/swebench-bash.jsonl
"""

import json
import os


RAW_DATA_DIR = "raw-data/swebench-bash"


def list_model_directories():
    """List all model directories from local raw-data folder."""
    directories = []
    for item in os.listdir(RAW_DATA_DIR):
        item_path = os.path.join(RAW_DATA_DIR, item)
        if os.path.isdir(item_path):
            directories.append(item)
    return sorted(directories)


def extract_model_name(dir_name):
    """Use folder name as model name."""
    return dir_name


def load_per_instance_details(dir_name):
    """Load per_instance_details.json from a model directory."""
    file_path = os.path.join(RAW_DATA_DIR, dir_name, "per_instance_details.json")
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"  Warning: Could not find {file_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"  Warning: Could not parse {file_path}: {e}")
        return None


def process_swebench_bash():
    """
    Process SWE-bench bash-only evaluation results into standardized format.

    Returns a list of records in the standard eval-arena format.
    """
    print("Scanning local model directories...")
    directories = list_model_directories()
    print(f"Found {len(directories)} model directories")

    records = []

    for dir_name in directories:
        print(f"Processing {dir_name}...")

        model_name = extract_model_name(dir_name)
        details = load_per_instance_details(dir_name)

        if details is None:
            continue

        resolved_count = 0
        total_count = 0

        for instance_id, instance_data in details.items():
            is_resolved = instance_data.get('resolved', False)

            records.append({
                'benchmark_id': 'swebench-bash',
                'model': model_name,
                'example_id': instance_id,
                'pass1': 1 if is_resolved else 0,
                'count': 1
            })

            if is_resolved:
                resolved_count += 1
            total_count += 1

        print(f"  {resolved_count}/{total_count} resolved ({100*resolved_count/total_count:.1f}%)")

    return records


def main():
    print("Starting SWE-bench Bash-Only data processing...")
    print("=" * 60)

    records = process_swebench_bash()

    if not records:
        print(f"\nNo data was processed. Please check that {RAW_DATA_DIR}/ contains model directories.")
        return

    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)

    output_file = 'data/swebench-bash.jsonl'

    with open(output_file, 'w') as f:
        for record in records:
            f.write(json.dumps(record) + '\n')

    print(f"\n{'=' * 60}")
    print(f"Saved {len(records)} records to {output_file}")
    print("=" * 60)

    # Summary statistics
    models = set(r['model'] for r in records)
    examples = set(r['example_id'] for r in records)

    print(f"\nSummary:")
    print(f"  Unique models: {len(models)}")
    print(f"  Unique examples: {len(examples)}")

    # Display first few records
    print("\nFirst 5 records:")
    for record in records[:5]:
        print(f"  {json.dumps(record)}")

    print("\nProcessing complete!")


if __name__ == '__main__':
    main()
