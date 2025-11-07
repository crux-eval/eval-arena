import glob
import json
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def load_jsonl_files(pattern: str) -> List[Dict[str, Any]]:
    """Load JSONL files matching the given glob pattern."""
    records = []
    for fname in glob.glob(pattern):
        with open(fname, 'rt') as f:
            records.extend([json.loads(line) for line in f.readlines()])
    logger.info(f"Loaded {len(records)} records from {pattern}")
    return records
