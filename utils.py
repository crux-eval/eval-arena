import glob
import json
import logging
from typing import Any, Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


def load_jsonl_files(pattern: str) -> List[Dict[str, Any]]:
    """Load JSONL files matching the given glob pattern."""
    records = []
    for fname in glob.glob(pattern):
        with open(fname, 'rt') as f:
            lines = [json.loads(line) for line in f.readlines()]
            logger.info(f"Loaded {len(lines)} from {fname}")
            records.extend(lines)
    logger.info(f"In total loaded {len(records)} records from {pattern}")
    return records

def fill_count(df) -> pd.DataFrame:
    for benchmark_id in df['benchmark_id'].unique():
        benchmark_df = df[df['benchmark_id'] == benchmark_id]
        if any(benchmark_df["count"].isna()):
            logger.info(f"no count on {benchmark_id=}, filling count=1")
            assert all(benchmark_df["count"].isna())
            df.loc[df['benchmark_id'] == benchmark_id, "count"] = 1
    return df

def check_data(df) -> None:
    """
    df has columns benchmark_id, example_id, model, count [Optional]
    validates that each model and example_id only appears once per benchmark
    If a model do appear, it must be complete for each benchmark and have all the example_ids exactly once 
    raise errors if these basic conditions are not satisfied
    If a count is present for benchmark_id, model, then it must be the same count for all example_id
    Claude generated validation code, partly verified by human.
    """
    if len(df) == 0:
        raise ValueError(f"No data to work with")

    # Check for duplicates: each (benchmark_id, example_id, model) should appear exactly once
    duplicates = df.groupby(['benchmark_id', 'example_id', 'model']).size()
    if (duplicates > 1).any():
        dup_entries = duplicates[duplicates > 1]
        raise ValueError(f"Duplicate entries found: {dup_entries.to_dict()}")
    
    # For each benchmark, get all unique example_ids
    for benchmark_id in df['benchmark_id'].unique():
        benchmark_df = df[df['benchmark_id'] == benchmark_id]
        all_example_ids = set(benchmark_df['example_id'].unique())
        
        # Check that each model has all example_ids for this benchmark
        for model in benchmark_df['model'].unique():
            model_example_ids = set(benchmark_df[benchmark_df['model'] == model]['example_id'].unique())
            if model_example_ids != all_example_ids:
                missing = all_example_ids - model_example_ids
                extra = model_example_ids - all_example_ids
                raise ValueError(
                    f"Model '{model}' in benchmark '{benchmark_id}' is incomplete. "
                    f"Missing: {missing}, Extra: {extra}"
                )
    
    # Check that count is consistent for each (benchmark_id, model) combination
    if 'count' in df.columns:
        count_check = df.groupby(['benchmark_id', 'model'])['count'].nunique()
        if (count_check > 1).any():
            inconsistent = count_check[count_check > 1]
            logging.warning(f"Inconsistent counts found for (benchmark_id, model): {inconsistent.to_dict()}")
    
    # warn if some benchmark only has 1 model, which cause trouble for paired comparisons
    for benchmark_id in df['benchmark_id'].unique():
        benchmark_df = df[df['benchmark_id'] == benchmark_id]
        num_models = benchmark_df['model'].nunique()
        if num_models == 1:
            raise ValueError(
                f"Benchmark '{benchmark_id}' only has 1 model ('{benchmark_df['model'].iloc[0]}'), "
                f"which will cause issues for paired comparisons"
            )
    
    logger.info("Validation passed successfully")
  