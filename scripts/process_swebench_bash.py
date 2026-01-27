#!/usr/bin/env python3
"""
Process SWE-bench Bash-Only evaluation results into standardized format.

This script fetches and processes raw evaluation data from:
https://github.com/SWE-bench/experiments/tree/main/evaluation/bash-only

Each subdirectory under bash-only contains results for a model, with naming format:
    YYYYMMDD_mini-vX.X.X_<model-name>

Each directory contains a per_instance_details.json file with:
    {
        "<instance_id>": {
            "cost": float,
            "api_calls": int,
            "resolved": bool
        },
        ...
    }

Output format (one JSON object per line):
    {"benchmark_id": "swebench-bash", "model": "<model-name>", "example_id": "<instance_id>", "pass1": 0|1, "count": 1}
"""

import json
import os
import re
import urllib.request
import urllib.error


GITHUB_API_BASE = "https://api.github.com/repos/SWE-bench/experiments/contents/evaluation/bash-only"
RAW_BASE = "https://raw.githubusercontent.com/SWE-bench/experiments/main/evaluation/bash-only"


def fetch_json(url):
    """Fetch JSON from a URL."""
    req = urllib.request.Request(url)
    req.add_header('User-Agent', 'eval-arena-processor')
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode('utf-8'))


def list_model_directories():
    """List all model directories from GitHub API."""
    data = fetch_json(GITHUB_API_BASE)
    directories = []
    for item in data:
        if item['type'] == 'dir':
            directories.append(item['name'])
    return sorted(directories)


def extract_model_name(dir_name):
    """
    Extract model name from directory name.
    """
    # Match YYYYMMDD_mini-vX.X.X followed by _ or -
    match = re.match(r'^\d{8}_mini-v[\d.]+[_-](.+)$', dir_name)
    if match:
        return match.group(1)
    return dir_name


def fetch_per_instance_details(dir_name):
    """Fetch per_instance_details.json for a model directory."""
    url = f"{RAW_BASE}/{dir_name}/per_instance_details.json"
    try:
        return fetch_json(url)
    except urllib.error.HTTPError as e:
        print(f"  Warning: Could not fetch {url}: {e}")
        return None


def process_swebench_bash():
    """
    Process SWE-bench bash-only evaluation results into standardized format.

    Returns a list of records in the standard eval-arena format.
    """
    print("Fetching list of model directories...")
    directories = list_model_directories()
    print(f"Found {len(directories)} model directories")

    records = []

    for dir_name in directories:
        print(f"Processing {dir_name}...")

        model_name = extract_model_name(dir_name)
        details = fetch_per_instance_details(dir_name)

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
        print("\nNo data was processed. Please check network connectivity.")
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
