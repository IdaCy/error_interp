#!/usr/bin/env python
import os
import json
import logging
import argparse

# Default parameters
DEFAULT_FILE_PATH = "prompts.json"
DEFAULT_PROMPT_KEY = "normal"
DEFAULT_MAX_SAMPLES = None  # No limit by default

def load_json_prompts(file_path=DEFAULT_FILE_PATH,
                      prompt_key=DEFAULT_PROMPT_KEY,
                      max_samples=DEFAULT_MAX_SAMPLES,
                      logger=None):
    """
    Load a JSON file (a list of objects). Return a list of (original_idx, text)
    for each row that has a non-empty value for 'prompt_key'.
    """
    if logger is None:
        logger = logging.getLogger("polAIlogger")
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Could not find JSON file: {file_path}")

    logger.debug(f"Reading JSON from: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("JSON must be a list of objects.")

    results = []
    for i, row in enumerate(data):
        text = row.get(prompt_key, "").strip()
        if text:
            results.append((i, text))

    if max_samples is not None and max_samples < len(results):
        results = results[:max_samples]

    logger.info(f"Loaded {len(results)} items from '{file_path}' with prompt_key='{prompt_key}'.")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load JSON prompts from a file.")
    parser.add_argument("--file_path", type=str, default=DEFAULT_FILE_PATH,
                        help="Path to the JSON file containing prompts.")
    parser.add_argument("--prompt_key", type=str, default=DEFAULT_PROMPT_KEY,
                        help="Key in each JSON object to extract text from.")
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="Maximum samples to load (-1 for no limit).")
    args = parser.parse_args()

    # Setup a basic logger if none is provided
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("polAIlogger")

    # Use None for no limit if a negative value is provided.
    max_samples_val = None if args.max_samples < 0 else args.max_samples

    prompts = load_json_prompts(file_path=args.file_path,
                                prompt_key=args.prompt_key,
                                max_samples=max_samples_val,
                                logger=logger)
    logger.info(f"Loaded {len(prompts)} prompts.")
