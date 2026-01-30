"""
Download SWE-bench instance IDs (test split) from HuggingFace Datasets.

Usage:
    python scripts/download_swebench_instance_ids.py <variant>

Available variants:
    default      - SWE-bench
    verified     - SWE-bench Verified
    lite         - SWE-bench Lite
    multimodal   - SWE-bench Multimodal
    multilingual - SWE-bench Multilingual

Examples:
    python scripts/download_swebench_instance_ids.py multimodal

Requires: pip install datasets

Output:
    raw-data/swebench-experiments/evaluation/<variant>/instance_ids.json
"""

import json
import os
import sys

from datasets import load_dataset


# Map variant names to HuggingFace dataset names
VARIANTS = {
    'default': 'SWE-bench/SWE-bench',
    'verified': 'SWE-bench/SWE-bench_Verified',
    'lite': 'SWE-bench/SWE-bench_Lite',
    'multimodal': 'SWE-bench/SWE-bench_Multimodal',
    'multilingual': 'SWE-bench/SWE-bench_Multilingual',
}

BASE_OUTPUT_PATH = "raw-data/swebench-experiments/evaluation"


def download_instance_ids(variant):
    if variant not in VARIANTS:
        print(f"Error: Unknown variant '{variant}'")
        print(f"Available variants: {', '.join(VARIANTS.keys())}")
        sys.exit(1)

    dataset_name = VARIANTS[variant]
    output_path = f"{BASE_OUTPUT_PATH}/{variant}/instance_ids.json"

    print(f"Loading {dataset_name} from HuggingFace...")
    ds = load_dataset(dataset_name, split='test')

    instance_ids = [row['instance_id'] for row in ds]
    print(f"Found {len(instance_ids)} instance IDs")

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(instance_ids, f, indent=2)

    print(f"Saved to {output_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/download_swebench_instance_ids.py <variant>")
        print(f"Available variants: {', '.join(VARIANTS.keys())}")
        sys.exit(1)

    variant = sys.argv[1].lower()
    download_instance_ids(variant)


if __name__ == '__main__':
    main()
