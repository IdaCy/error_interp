#!/usr/bin/env python
import os
import logging
import argparse

# Default values for the logger
DEFAULT_LOG_FILE = "logs/inference.log"
DEFAULT_CONSOLE_LEVEL = logging.INFO
DEFAULT_FILE_LEVEL = logging.DEBUG

def init_logger(log_file=DEFAULT_LOG_FILE,
                console_level=DEFAULT_CONSOLE_LEVEL,
                file_level=DEFAULT_FILE_LEVEL):
    """
    Creates a logger that writes detailed logs to a file
    and a more concise output to the console.
    """
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.CRITICAL)  # no propagation

    logger = logging.getLogger("polAIlogger")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(file_level)
    file_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                 datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(file_fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(console_level)
    console_fmt = logging.Formatter("[%(levelname)s] %(message)s")
    ch.setFormatter(console_fmt)
    logger.addHandler(ch)

    logger.info("Logger initialized.")
    return logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize the logger.")
    parser.add_argument("--log_file", type=str, default=DEFAULT_LOG_FILE)
    parser.add_argument("--console_level", type=int, default=DEFAULT_CONSOLE_LEVEL,
                        help="Numeric logging level (e.g., 20 for INFO)")
    parser.add_argument("--file_level", type=int, default=DEFAULT_FILE_LEVEL,
                        help="Numeric logging level (e.g., 10 for DEBUG)")
    args = parser.parse_args()

    # Create logger and log a test message
    logger = init_logger(log_file=args.log_file,
                         console_level=args.console_level,
                         file_level=args.file_level)
    logger.info("Logger started via command line.")
