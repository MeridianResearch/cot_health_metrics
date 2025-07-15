#!/usr/bin/env python3
"""
use it:

-as a library-
 - from data_loader import load_prompts
 - prompts = load_prompts("data/alpaca_500_samples.json", max_samples=500)
 - returns a list of dicts with keys: prompt_id, instruction, input, output, prompt_hash

-by itself-
```
python src/data_loader.py \
    --data-path data/alpaca_500_samples.json \
    --max-samples 5
```

-> that will:
    - read the JSON array of prompt objects
    - write progress logs to logs/data_loader_<timestamp>.log
      (see one log line every LOG_EVERY samples)
    - print the first prompt sample to stdout for a sanity check
"""

import argparse
import json
import logging
import time
from typing import List, Dict

# Settings
LOG_EVERY = 50

def setup_logger(name: str, log_file: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    fh = logging.FileHandler(log_file)
    fh.setLevel(level)
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger

def load_prompts(json_path: str, max_samples: int = None) -> List[Dict]:
    """Read the JSON file of prompts and return as a list of dicts"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    if max_samples is not None:
        data = data[:max_samples]
    return data

def main():
    parser = argparse.ArgumentParser(description="Data Loader for CoT Health Metrics")
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to prompts JSON file"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to load"
    )
    args = parser.parse_args()

    timestamp = int(time.time())
    logger = setup_logger(
        "data_loader",
        f"logs/data_loader_{timestamp}.log"
    )
    logger.info(f"Loading data from {args.data_path}")
    prompts = load_prompts(args.data_path, args.max_samples)
    logger.info(f"Loaded {len(prompts)} samples")

    for idx, sample in enumerate(prompts):
        if idx % LOG_EVERY == 0:
            logger.info(f"Sample {idx}: prompt_id={sample.get('prompt_id')}")

    # to print first sample to stdout
    if prompts:
        print(json.dumps(prompts[0], indent=2))

if __name__ == "__main__":
    main()
