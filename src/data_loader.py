#!/usr/bin/env python3
"""
use it:

-as a library-
 - from data_loader import load_prompts, load_filler_texts
 - prompts = load_prompts("data/alpaca_500_samples.json", max_samples=500)
 - filler_texts = load_filler_texts("data/filler_texts.json")
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
import os
import pandas as pd
from typing import List, Dict, Optional

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


def load_prompts(json_path: str, max_samples: Optional[int] = None) -> List[Dict]:
    """Read the JSON file of prompts and return as a list of dicts"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    if max_samples is not None:
        data = data[:max_samples]
    return data


def load_csv_dataset(csv_path: str, max_samples: Optional[int] = None) -> List[Dict]:
    """Load a CSV dataset and convert to list of dicts.

    Args:
        csv_path: Path to the CSV file
        max_samples: Optional maximum number of samples to load

    Returns:
        List of dictionaries representing each row
    """
    df = pd.read_csv(csv_path)
    if max_samples is not None:
        df = df.head(max_samples)
    return df.to_dict('records')


def load_filler_texts(json_path: str = "data/filler_texts.json") -> Dict[str, str]:
    """Load filler texts from JSON file.

    Args:
        json_path: Path to the filler texts JSON file

    Returns:
        Dictionary mapping filler text names to their content

    Raises:
        FileNotFoundError: If the filler texts file doesn't exist
        KeyError: If the JSON structure is invalid
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Filler texts file not found: {json_path}")

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Validate JSON structure
        if "filler_texts" not in data:
            raise KeyError("JSON file must contain 'filler_texts' key")

        return data["filler_texts"]

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in filler texts file: {e}")


def get_filler_text(filler_name: str, filler_texts_path: str = "data/filler_texts.json") -> str:
    """Get a specific filler text by name.

    Args:
        filler_name: Name of the filler text (e.g., 'lorem_ipsum', 'cicero_original')
        filler_texts_path: Path to the filler texts JSON file

    Returns:
        The filler text content

    Raises:
        ValueError: If the filler text name is not found
    """
    filler_texts = load_filler_texts(filler_texts_path)

    if filler_name not in filler_texts:
        available_names = list(filler_texts.keys())
        raise ValueError(f"Filler text '{filler_name}' not found. Available options: {available_names}")

    return filler_texts[filler_name]


def list_available_filler_texts(filler_texts_path: str = "data/filler_texts.json") -> List[str]:
    """List all available filler text names.

    Args:
        filler_texts_path: Path to the filler texts JSON file

    Returns:
        List of available filler text names
    """
    try:
        filler_texts = load_filler_texts(filler_texts_path)
        return list(filler_texts.keys())
    except (FileNotFoundError, ValueError, KeyError):
        return []


def main():
    parser = argparse.ArgumentParser(description="Data Loader for CoT Health Metrics")
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to prompts JSON file"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to load"
    )
    parser.add_argument(
        "--test-filler-texts",
        action="store_true",
        help="Test loading filler texts instead of prompts"
    )
    parser.add_argument(
        "--filler-texts-path",
        type=str,
        default="data/filler_texts.json",
        help="Path to filler texts JSON file"
    )
    args = parser.parse_args()

    timestamp = int(time.time())
    logger = setup_logger(
        "data_loader",
        f"logs/data_loader_{timestamp}.log"
    )

    if args.test_filler_texts:
        # Test filler texts loading
        logger.info(f"Testing filler texts from {args.filler_texts_path}")
        try:
            available_texts = list_available_filler_texts(args.filler_texts_path)
            logger.info(f"Available filler texts: {available_texts}")
            print(f"Available filler texts: {available_texts}")

            # Show a sample of each
            for name in available_texts:
                text = get_filler_text(name, args.filler_texts_path)
                preview = text[:100] + "..." if len(text) > 100 else text
                print(f"\n{name}: {preview}")

        except Exception as e:
            logger.error(f"Error loading filler texts: {e}")
            print(f"Error: {e}")

    elif args.data_path:
        # Original prompts loading functionality
        logger.info(f"Loading data from {args.data_path}")
        prompts = load_prompts(args.data_path, args.max_samples)
        logger.info(f"Loaded {len(prompts)} samples")

        for idx, sample in enumerate(prompts):
            if idx % LOG_EVERY == 0:
                logger.info(f"Sample {idx}: prompt_id={sample.get('prompt_id')}")

        # to print first sample to stdout
        if prompts:
            print(json.dumps(prompts[0], indent=2))

    else:
        print("Please specify either --data-path for prompts or --test-filler-texts for filler texts")
        parser.print_help()


if __name__ == "__main__":
    main()