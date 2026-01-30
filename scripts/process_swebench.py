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
│   ├── <model-name>/
│   │   └── results/
│   │       └── results.json
│   └── ...
├── verified/
│   └── ...
└── test/
    └── ...
```

Output will be saved to:
- data/swebench-lite.jsonl
- data/swebench-verified.jsonl
- data/swebench-test.jsonl
"""

import json
import glob
import pandas as pd
import os


def process_swe(swetype, id_path):
    with open(id_path) as id_file:
        res = json.load(id_file)
    # print(res)
    ids = set(res['generated']) | set(res['no_generation'])
    print(f'len of {swetype}', len(ids))
    print('model', 'total', 'deduped')
    for fname in glob.glob(f"raw-data/swebench-experiments/evaluation/{swetype}/*/results/results.json"):
        mname = fname.split('/')[-3]
        try:
            with open(fname) as f:
                res = json.load(f)
        except:
            print('not jsonl', fname)

        total_list = res.get("generated", []) + res.get("no_generation", [])
        print(mname, len(total_list), len(set(total_list)))
        if len(total_list) != len(ids):
            print(f"results for {fname} is possibly incomplete")

    records = []
    for fname in glob.glob(f"raw-data/swebench-experiments/evaluation/{swetype}/*/results/results.json"):
        mname = fname.split('/')[-3]
        with open(fname, 'r') as f:
            result = json.load(f)
        if 'resolved' in result:
            resolved = set(result['resolved'])
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
