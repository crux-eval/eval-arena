#!/usr/bin/env python3
"""
python scripts/process_swebench.py

Convert SWE-bench raw evaluation data into the
standardized format used by eval-arena.

Raw data source: [SWE-bench/experiments](https://github.com/SWE-bench/experiments/tree/main/evaluation/
The script expects raw data in this directory structure:
```
raw-data/swebench-experiments/evaluation/
├── lite/
│   ├── <run-name>/
│   │   ├── metadata.yaml       # contains info.name for display name
│   │   └── results/
│   │       └── results.json
│   └── ...
├── verified/
│   └── ...
├── test/
│   └── ...
└── multimodal/
    ├── instance_ids.json       # from HuggingFace dataset
    └── <run-name>/
        ├── metadata.yaml
        └── results/
            └── results.json
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


def load_instance_ids(swetype, id_path):
    """Load instance IDs for a benchmark type.

    For multimodal: loads from instance_ids.json (list of IDs from HuggingFace dataset)
    For others: loads from reference model's results.json (generated + no_generation)
    """
    with open(id_path) as id_file:
        res = json.load(id_file)

    if swetype == 'multimodal':
        return set(res)
    else:
        return set(res['generated']) | set(res['no_generation'])


def process_swe(swetype, id_path):
    ids = load_instance_ids(swetype, id_path)
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


if __name__ == '__main__':
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)

    swetype = 'lite'
    id_path = 'raw-data/swebench-experiments/evaluation/lite/20231010_rag_claude2/results/results.json'
    process_swe(swetype, id_path)

    swetype = 'verified'
    id_path = 'raw-data/swebench-experiments/evaluation/verified/20231010_rag_claude2/results/results.json'
    process_swe(swetype, id_path)

    swetype = 'test'
    id_path = 'raw-data/swebench-experiments/evaluation/test/20231010_rag_claude2/results/results.json'
    process_swe(swetype, id_path)

    swetype = 'multimodal'
    id_path = 'raw-data/swebench-experiments/evaluation/multimodal/instance_ids.json'
    process_swe(swetype, id_path)
