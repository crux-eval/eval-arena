#!/usr/bin/env python3
"""
Convert SWE-bench raw evaluation data into the standardized format used by eval-arena.

Usage:
    python scripts/process_swebench.py [<variant> ...]

Examples:
    python scripts/process_swebench.py              # run all variants
    python scripts/process_swebench.py lite         # run only lite
    python scripts/process_swebench.py lite verified # run lite and verified

Raw data source: [SWE-bench/experiments](https://github.com/SWE-bench/experiments/tree/main/evaluation/
The script expects raw data in this directory structure:
```
raw-data/swebench-experiments/evaluation/
├── lite/
│   ├── instance_ids.json       # from HuggingFace (via download_swebench_instance_ids.py)
│   └── <run-name>/
│       ├── metadata.yaml       # contains info.name for display name
│       └── results/
│           └── results.json
├── verified/
│   └── ...
├── test/
│   └── ...
└── multimodal/
    └── ...
```

Output will be saved to:
- data/swebench-lite.jsonl
- data/swebench-verified.jsonl
- data/swebench-test.jsonl
- data/swebench-multimodal.jsonl
"""

import json
import glob
import pandas as pd
import os
import yaml


BASE_PATH = "raw-data/swebench-experiments/evaluation"


def get_model_name(swetype, dir_name):
    """Extract model name from metadata.yaml, fallback to folder name."""
    metadata_path = f"{BASE_PATH}/{swetype}/{dir_name}/metadata.yaml"
    try:
        with open(metadata_path, 'r') as f:
            metadata = yaml.safe_load(f)
        if metadata and 'info' in metadata and 'name' in metadata['info']:
            return metadata['info']['name']
    except (FileNotFoundError, yaml.YAMLError):
        pass
    return dir_name


def load_instance_ids(id_path):
    with open(id_path) as id_file:
        return set(json.load(id_file))


def process_swe(swetype, id_path):
    ids = load_instance_ids(id_path)
    print(f'len of {swetype}', len(ids))
    print('model', 'total', 'deduped')
    for fname in glob.glob(f"{BASE_PATH}/{swetype}/*/results/results.json"):
        dir_name = fname.split('/')[-3]
        mname = get_model_name(swetype, dir_name)
        try:
            with open(fname) as f:
                res = json.load(f)
        except:
            print('not jsonl', fname)

        # Data completeness check
        if swetype == 'multimodal':
            resolved_list = res.get("resolved", [])
            if isinstance(resolved_list, int):
                print(f"  results for {fname} has summary only: {resolved_list} resolved")
                continue
            total_list = resolved_list + res.get("no_generation", []) + res.get("no_logs", [])
        else:
            total_list = res.get("generated", []) + res.get("no_generation", [])

        print(mname, len(total_list), len(set(total_list)))
        if len(set(total_list)) != len(ids):
            print(f"  results for {fname} is possibly incomplete ({len(set(total_list))}/{len(ids)})")

    records = []
    for fname in glob.glob(f"{BASE_PATH}/{swetype}/*/results/results.json"):
        dir_name = fname.split('/')[-3]
        mname = get_model_name(swetype, dir_name)
        with open(fname, 'r') as f:
            result = json.load(f)
        if 'resolved' in result:
            resolved_raw = result['resolved']
            # Skip if resolved is just a count (multimodal result issue)
            if isinstance(resolved_raw, int):
                continue
            resolved = set(resolved_raw)
        else:
            resolved = set()

        for id in ids:
            records.append({
                'benchmark_id': f'swebench-{swetype}',
                'model': mname,
                'example_id': id,
                'pass1': 1 if id in resolved else 0,
                'count': 1,  # all single submission
            })
    dfo = pd.DataFrame(records)
    print(swetype)
    print(dfo["model"].describe())
    dfo.to_json(f'data/swebench-{swetype}.jsonl', orient='records', lines=True)


SUPPORTED_VARIANTS = ['lite', 'verified', 'test', 'multimodal']


if __name__ == '__main__':
    import sys

    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)

    # Use command-line args if provided, otherwise run all variants
    variants = sys.argv[1:] if len(sys.argv) > 1 else SUPPORTED_VARIANTS

    for swetype in variants:
        if swetype not in SUPPORTED_VARIANTS:
            print(f"Error: Unknown variant '{swetype}'")
            print(f"Available variants: {', '.join(SUPPORTED_VARIANTS)}")
            sys.exit(1)
        id_path = f'{BASE_PATH}/{swetype}/instance_ids.json'
        process_swe(swetype, id_path)
