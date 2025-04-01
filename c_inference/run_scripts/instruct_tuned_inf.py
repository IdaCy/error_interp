#!/usr/bin/env python
import os
import math
import json
import logging
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from b_utils.run_scripts.load_model import load_model

# Default parameters
DEFAULT_OUTPUT_DIR = "output/"
DEFAULT_BATCH_SIZE = 4
DEFAULT_MAX_SEQ_LENGTH = 2048
DEFAULT_TOP_K_LOGITS = 10

DEFAULT_GENERATION_KWARGS = {
    "do_sample": True,
    "max_new_tokens": 100,
    "temperature": 0.7,
    "top_p": 0.9,
    "repetition_penalty": 1.2
}

def run_inf(model,
            tokenizer,
            data,
            output_dir=DEFAULT_OUTPUT_DIR,
            batch_size=DEFAULT_BATCH_SIZE,
            max_seq_length=DEFAULT_MAX_SEQ_LENGTH,
            extract_hidden_layers=None,
            extract_attention_layers=None,
            top_k_logits=DEFAULT_TOP_K_LOGITS,
            logger=None,
            generation_kwargs=None):
    """
    Runs inference on provided data and saves results batchwise.
    """
    if logger is None:
        logger = logging.getLogger("polAIlogger")

    if extract_hidden_layers is None:
        extract_hidden_layers = [0, 5, 10, 15]
    if extract_attention_layers is None:
        extract_attention_layers = [0, 5, 10, 15]

    os.makedirs(output_dir, exist_ok=True)

    logger.info("Clearing CUDA cache before starting.")
    torch.cuda.empty_cache()

    if generation_kwargs is None:
        generation_kwargs = DEFAULT_GENERATION_KWARGS

    total_samples = len(data)
    total_batches = math.ceil(total_samples / batch_size)
    logger.warning(f"=== Starting inference. #samples={total_samples}, batch_size={batch_size} ===")

    for batch_idx in range(total_batches):
        start_i = batch_idx * batch_size
        end_i = min((batch_idx + 1) * batch_size, total_samples)
        batch_items = data[start_i:end_i]
        batch_indices = [x[0] for x in batch_items]
        batch_texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": x[1]}],
                tokenize=False,
                add_generation_prompt=True
            )
            for x in batch_items
        ]

        if batch_idx % 20 == 0:
            logger.info(f"Processing batch {batch_idx+1}/{total_batches} (samples {start_i}-{end_i-1})")

        encodings = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt"
        )
        input_ids = encodings["input_ids"].cuda()
        attention_mask = encodings["attention_mask"].cuda()

        try:
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    output_attentions=True
                )
                gen_out = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    **generation_kwargs
                )

            hidden_map = {}
            for layer_idx in extract_hidden_layers:
                if layer_idx < len(outputs.hidden_states):
                    hidden_map[f"layer_{layer_idx}"] = outputs.hidden_states[layer_idx].cpu()

            attn_map = {}
            for layer_idx in extract_attention_layers:
                if layer_idx < len(outputs.attentions):
                    attn_map[f"layer_{layer_idx}"] = outputs.attentions[layer_idx].cpu()

            logits = outputs.logits
            topk_vals, topk_indices = torch.topk(logits, k=top_k_logits, dim=-1)
            topk_vals = topk_vals.cpu()
            topk_indices = topk_indices.cpu()

            decoded_preds = [
                tokenizer.decode(o, skip_special_tokens=True) for o in gen_out.cpu()
            ]

            out_dict = {
                "hidden_states": hidden_map,
                "attentions": attn_map,
                "topk_vals": topk_vals,
                "topk_indices": topk_indices,
                "input_ids": input_ids.cpu(),
                "final_predictions": decoded_preds,
                "original_indices": batch_indices
            }

            save_name = f"activations_{start_i:05d}_{end_i:05d}.pt"
            save_path = os.path.join(output_dir, save_name)
            torch.save(out_dict, save_path)
            logger.debug(f"Saved batch => {save_path}")

        except torch.cuda.OutOfMemoryError:
            logger.error(f"OOM error on batch {batch_idx}. Clearing cache and continuing.")
            torch.cuda.empty_cache()
        except Exception as ex:
            logger.exception(f"Error on batch {batch_idx}: {ex}")

    logger.warning("=== Inference Complete ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a dataset.")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max_seq_length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    parser.add_argument("--top_k_logits", type=int, default=DEFAULT_TOP_K_LOGITS)
    args = parser.parse_args()

    # Setup a basic logger if needed
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("polAIlogger")
    
    # For demonstration, create some dummy data: a list of (index, text) pairs.
    dummy_data = [(i, f"Sample text {i}") for i in range(10)]

    # Load model and tokenizer (you can pass command-line args as needed)
    model, tokenizer = load_model(logger=logger)

    run_inf(model=model,
            tokenizer=tokenizer,
            data=dummy_data,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
            top_k_logits=args.top_k_logits,
            logger=logger)
