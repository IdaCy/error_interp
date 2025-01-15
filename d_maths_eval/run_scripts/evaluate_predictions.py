#!/usr/bin/env python
import os
import logging
import torch
from glob import glob
import argparse

# Import the parsing function from the separate module
from d_maths_eval.run_scripts.parse_predictions import parse_prediction_text

# Default parameters
DEFAULT_PT_FILE = "output/extractions/gemma2bit/normal"
DEFAULT_LOG_FILE = "logs/extraction_log.txt"
DEFAULT_ALLOWED_OPERATIONS = "+-*÷"
DEFAULT_RESULTS_FILE = "logs/calculation_results.txt"

def evaluate_predictions(pt_file=DEFAULT_PT_FILE,
                         log_file=DEFAULT_LOG_FILE,
                         allowed_operations=DEFAULT_ALLOWED_OPERATIONS,
                         results_file=DEFAULT_RESULTS_FILE):
    """
    Evaluates the predictions stored in .pt files (or a single .pt file) by checking if the
    predicted answer (last number in the text) correctly corresponds to the calculation
    from the first two numbers and the allowed operation symbol.
    
    Detailed logging is written to the log file and a summary to the results file.
    Returns the overall percentage of correct predictions.
    """
    # Ensure directories for log and results files exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    # Set up logging (only to file; disable propagation to prevent console output)
    logger = logging.getLogger("PTExtractionLogger")
    logger.setLevel(logging.DEBUG)
    if logger.hasHandlers():
        logger.handlers.clear()
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    logger.propagate = False

    logger.info("=== Starting extraction evaluation ===")
    logger.info(f"PT_FILE (dir or file) = {pt_file}")
    logger.info(f"LOG_FILE = {log_file}")
    logger.info(f"ALLOWED_OPERATIONS = {allowed_operations}")
    logger.info(f"RESULTS_FILE = {results_file}")

    with open(results_file, "w", encoding="utf-8") as summary_f:
        if os.path.isdir(pt_file):
            pt_paths = sorted(glob(os.path.join(pt_file, "*.pt")))
        else:
            pt_paths = [pt_file]
        logger.info(f"Found {len(pt_paths)} .pt file(s) to process.")

        overall_total = 0
        overall_correct = 0

        for pt_path in pt_paths:
            logger.info("--------------------------------------------------")
            logger.info(f"Processing file: {pt_path}")
            logger.info("------")
            try:
                data = torch.load(pt_path)
            except Exception as e:
                logger.error(f"Failed to load {pt_path}: {str(e)}")
                continue

            predictions = data.get("final_predictions", [])
            logger.info(f"Number of predictions in this file: {len(predictions)}")

            file_correct = 0
            file_total = 0

            for idx, pred_text in enumerate(predictions):
                parsed = parse_prediction_text(pred_text, allowed_operations)
                logger.debug(
                    f"[File={os.path.basename(pt_path)} | Idx={idx}] "
                    f"Extracted -> first_num={parsed['first_num']}, "
                    f"operation={parsed['operation']}, "
                    f"second_num={parsed['second_num']}, "
                    f"predicted_answer={parsed['predicted_answer']}, "
                    f"computed_answer={parsed['computed_answer']}, "
                    f"correct={parsed['correct']}"
                )
                file_total += 1
                if parsed["correct"]:
                    file_correct += 1

            accuracy = (file_correct / file_total * 100) if file_total > 0 else 0
            logger.info(
                f"File summary: correct={file_correct}, total={file_total}, accuracy={accuracy:.2f}%"
            )
            summary_f.write(
                f"File: {pt_path}, correct={file_correct}, total={file_total}, accuracy={accuracy:.2f}%\n"
            )

            overall_total += file_total
            overall_correct += file_correct

        overall_acc = (overall_correct / overall_total * 100) if overall_total > 0 else 0
        logger.info("==================================================")
        logger.info(
            f"Overall correctness across all files: {overall_correct} / {overall_total} ({overall_acc:.2f}%)"
        )
        summary_f.write(
            "==================================================\n"
            f"Overall correctness: {overall_correct}/{overall_total} ({overall_acc:.2f}%)\n"
        )

    return overall_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate prediction calculations from .pt files."
    )
    parser.add_argument("--pt_file", type=str, default=DEFAULT_PT_FILE,
                        help="Directory containing .pt files or a single .pt file.")
    parser.add_argument("--log_file", type=str, default=DEFAULT_LOG_FILE,
                        help="Path to the log file.")
    parser.add_argument("--allowed_operations", type=str, default=DEFAULT_ALLOWED_OPERATIONS,
                        help="String of allowed operation symbols.")
    parser.add_argument("--results_file", type=str, default=DEFAULT_RESULTS_FILE,
                        help="Path to the results summary file.")
    args = parser.parse_args()

    # Setup basic logging if needed
    logging.basicConfig(level=logging.INFO)
    overall_accuracy = evaluate_predictions(
        pt_file=args.pt_file,
        log_file=args.log_file,
        allowed_operations=args.allowed_operations,
        results_file=args.results_file,
    )
    print(f"Overall accuracy: {overall_accuracy:.2f}%")
