#!/usr/bin/env python
import os
import torch
import logging
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

# Default parameters
DEFAULT_MODEL_NAME = "google/gemma-2-9b-it"
DEFAULT_USE_BFLOAT16 = True

def load_model(model_name=DEFAULT_MODEL_NAME,
               use_bfloat16=DEFAULT_USE_BFLOAT16,
               hf_token=None,
               logger=None):
    """
    Loads the specified model and tokenizer from Hugging Face.
    """
    if logger is None:
        logger = logging.getLogger("polAIlogger")

    if not torch.cuda.is_available():
        raise RuntimeError("No GPU found. Please enable a GPU in Colab.")

    logger.info(f"Loading tokenizer from '{model_name}'")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.debug("No pad_token found; using eos_token as pad_token.")

    gpu_mem = torch.cuda.get_device_properties(0).total_memory // (1024**3)
    max_memory = {0: f"{int(gpu_mem * 0.9)}GB"}
    logger.info(f"Loading model '{model_name}' (bfloat16={use_bfloat16}) with device_map=auto")

    dtype = torch.bfloat16 if use_bfloat16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
        max_memory=max_memory,
        use_auth_token=hf_token
    )
    model.eval()
    logger.info("Model loaded successfully.")
    return model, tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load a Hugging Face model.")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--use_bfloat16", type=lambda x: (str(x).lower() == 'true'),
                        default=DEFAULT_USE_BFLOAT16)
    parser.add_argument("--hf_token", type=str, default=None)
    args = parser.parse_args()

    # Setup a basic logger if none provided
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("polAIlogger")
    load_model(model_name=args.model_name, use_bfloat16=args.use_bfloat16,
               hf_token=args.hf_token, logger=logger)
