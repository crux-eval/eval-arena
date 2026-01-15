# Terminal Bench Data Processing Guide

## Overview

A standalone Python script [process_terminal_bench.py](process_terminal_bench.py) is provided to convert Terminal Bench raw evaluation data from [terminal-bench-core@0.1.1](https://github.com/laude-institute/terminal-bench-leaderboard/tree/main/results/terminal-bench-core%400.1.1) into the standardized format used by eval-arena.

## Input Data Structure

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

## What the Function Does

The `process_terminal_bench()` function:

1. **Scans all experiment directories** matching the pattern `YYYYMMDD_<agent-name>_<model-name>`
2. **Extracts agent and model names** from directory names
3. **Finds all results files** (both direct `*_results.json` and nested `results.json`)
4. **Aggregates results across runs** for each task:
   - Collects all attempts (typically 5) for each task
   - Calculates `pass@1` as the mean success rate across runs
   - Tracks the number of runs (`count`)
5. **Creates standardized output** with fields:
   - `benchmark_id`: "terminal-bench"
   - `model`: "{agent_name}_{model_name}" (e.g., "goose_claude-4-sonnet")
   - `example_id`: task identifier (e.g., "eval-mteb", "swe-bench-fsspec")
   - `pass1`: success rate across runs (0.0 to 1.0)
   - `count`: number of runs for this task (typically 5)

## How to Use

1. **Place your raw data** in `raw-data/terminal-bench-core@0.1.1/`

2. **Run the processing script**:
   ```bash
   python process_terminal_bench.py
   ```

3. **Output will be saved** to `data/terminal-bench.jsonl`

## Output Format

The output follows the standard eval-arena format:

```json
{"benchmark_id": "terminal-bench", "model": "goose_claude-4-sonnet", "example_id": "eval-mteb", "pass1": 1.0, "count": 5}
{"benchmark_id": "terminal-bench", "model": "goose_claude-4-sonnet", "example_id": "swe-bench-fsspec", "pass1": 0.8, "count": 5}
```

## Summary Statistics

The function automatically displays:
- Total records processed
- Number of unique models
- Number of unique tasks
- Per-model summary with:
  - Average pass@1 across all tasks
  - Number of tasks evaluated
  - Number of runs per task

## Customization

If your data is in a different location, you can modify the function call:

```python
df_terminal_bench = process_terminal_bench('path/to/your/data')
```

## Notes

- The function handles both directory structures (direct results files and nested subdirectories)
- It has special handling for directories that don't match the expected naming pattern
- Errors in individual result files are logged but don't stop the overall processing
- The `model` field combines agent and model name to create unique identifiers
