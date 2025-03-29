#!/usr/bin/env python
import os
import math
import logging
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from b_utils.run_scripts.load_model import load_model

# Default parameters
DEFAULT_OUTPUT_DIR = "output_main_patched/"
DEFAULT_BATCH_SIZE = 4
DEFAULT_MAX_SEQ_LENGTH = 2048
DEFAULT_TOP_K_LOGITS = 10
DEFAULT_NUM_LAYERS = 26  # total number of layers in the model

DEFAULT_GENERATION_KWARGS = {
    "do_sample": True,
    "max_new_tokens": 100,
    "temperature": 0.7,
    "top_p": 0.9,
    "repetition_penalty": 1.2
}

def load_normal_attention(normal_attention_dir, logger=None):
    """
    Loads all saved normal attention .pt files from a directory into a dictionary.
    Each sample is keyed by its original index.
    """
    if logger is None:
        logger = logging.getLogger()

    normal_attn_map_all = {}
    for fname in os.listdir(normal_attention_dir):
        if not fname.endswith(".pt"):
            continue
        fpath = os.path.join(normal_attention_dir, fname)
        logger.info(f"Loading normal attention from {fpath} ...")
        saved_dict = torch.load(fpath, map_location="cpu")
        indices = saved_dict["original_indices"]
        attn_map = saved_dict["attentions"]
        # Save each attention map in a dictionary
        for idx in range(len(indices)):
            sample_idx = indices[idx]
            single_sample_map = {}
            for layer_key, layer_tensor in attn_map.items():
                # layer_tensor shape: [batch_size, num_heads, seq_len, seq_len]
                single_sample_map[layer_key] = layer_tensor[idx]  # shape: [num_heads, seq_len, seq_len]
            normal_attn_map_all[sample_idx] = single_sample_map
    return normal_attn_map_all

def patch_attention(current_attention, normal_attention, scale=1.0):
    """
    Blends the current (typo) attention with the normal one.
      - scale=1.0 => fully replace with normal
      - scale=0.5 => average them, etc.
    Both inputs should have the same shape.
    """
    patched = (1.0 - scale) * current_attention + scale * normal_attention
    return patched

def run_inf_main_patched(model,
                         tokenizer,
                         data,
                         normal_attn_map_all,
                         patch_scale=1.0,
                         patch_layers=None,
                         output_dir=DEFAULT_OUTPUT_DIR,
                         batch_size=DEFAULT_BATCH_SIZE,
                         max_seq_length=DEFAULT_MAX_SEQ_LENGTH,
                         top_k_logits=DEFAULT_TOP_K_LOGITS,
                         logger=None,
                         generation_kwargs=None,
                         extract_attention_layers=None):
    """
    Runs inference on 'main' (typo) data while patching the attention using the saved normal attention.
    - patch_scale: a float in [0.0, 1.0] controlling how much normal attention is used.
    - patch_layers: list of layer indices to patch; if None, defaults to all layers.
    """
    if logger is None:
        logger = logging.getLogger("polAIlogger")

    # If no specific layers are provided, patch all layers.
    if patch_layers is None:
        patch_layers = list(range(DEFAULT_NUM_LAYERS))
    # Also use these layers for extraction if not provided
    if extract_attention_layers is None:
        extract_attention_layers = patch_layers

    os.makedirs(output_dir, exist_ok=True)

    logger.info("Clearing CUDA cache before starting (typo patch).")
    torch.cuda.empty_cache()

    if generation_kwargs is None:
        generation_kwargs = DEFAULT_GENERATION_KWARGS

    # Set the model's config to output attentions (if not already set)
    model.config.output_attentions = True

    # We'll use a set for quick lookup of which layers to patch.
    attention_patch_layers = set(patch_layers)

    total_samples = len(data)
    total_batches = math.ceil(total_samples / batch_size)
    logger.warning(f"=== Starting MAIN (typo) inference with patched attention. #samples={total_samples} ===")

    # Process samples one by one to allow per-sample patching.
    results = []
    for batch_idx in range(total_batches):
        start_i = batch_idx * batch_size
        end_i = min((batch_idx + 1) * batch_size, total_samples)
        batch_items = data[start_i:end_i]
        batch_indices = [x[0] for x in batch_items]
        batch_texts = [x[1] for x in batch_items]

        if batch_idx % 20 == 0:
            logger.info(f"[Typo Patch] Processing batch {batch_idx+1}/{total_batches} (samples {start_i}-{end_i-1})")

        encodings = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt"
        )
        input_ids = encodings["input_ids"].cuda()
        attention_mask = encodings["attention_mask"].cuda()

        # For simplicity, we perform a single-sample forward pass for each sample in the batch.
        for i, sample_idx in enumerate(batch_indices):
            single_input_ids = input_ids[i].unsqueeze(0)
            single_attn_mask = attention_mask[i].unsqueeze(0)

            # Define a patched forward for a single sample.
            original_forward_inner = model.forward
            def single_sample_patched_forward(input_ids=None, attention_mask=None, **kwargs):
                # Remove explicit output_attentions argument (since model.config is already set)
                raw_outputs = original_forward_inner(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **kwargs
                )
                new_attns = []
                # Use the saved normal attention if available for this sample.
                if sample_idx in normal_attn_map_all:
                    normal_sample_map = normal_attn_map_all[sample_idx]  # dict: layer_key -> tensor
                else:
                    normal_sample_map = {}
                for layer_idx, attn_tensor in enumerate(raw_outputs.attentions):
                    # attn_tensor shape: [1, num_heads, seq_len, seq_len]
                    if layer_idx in attention_patch_layers:
                        layer_key = f"layer_{layer_idx}"
                        if layer_key in normal_sample_map:
                            normal_attn = normal_sample_map[layer_key].to(attn_tensor.device)
                            patched_layer = patch_attention(attn_tensor, normal_attn, scale=patch_scale)
                            new_attns.append(patched_layer)
                        else:
                            new_attns.append(attn_tensor)
                    else:
                        new_attns.append(attn_tensor)
                # Reconstruct the output with patched attentions.
                final_output = raw_outputs.__class__(
                    loss=raw_outputs.loss,
                    logits=raw_outputs.logits,
                    past_key_values=raw_outputs.past_key_values,
                    hidden_states=raw_outputs.hidden_states,
                    attentions=tuple(new_attns),
                )
                return final_output

            model.forward = single_sample_patched_forward
            with torch.no_grad():
                outputs = model(
                    input_ids=single_input_ids,
                    attention_mask=single_attn_mask
                )
                gen_out = model.generate(
                    single_input_ids,
                    attention_mask=single_attn_mask,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    **generation_kwargs
                )
            # Restore original forward.
            model.forward = original_forward_inner

            decoded_pred = tokenizer.decode(gen_out[0].cpu(), skip_special_tokens=True)
            results.append({
                "sample_idx": sample_idx,
                "final_prediction": decoded_pred,
                "attentions": [a.cpu() for a in outputs.attentions],
            })

    # Save all results in one file (you can modify this to save per batch if desired)
    save_name = "patched_main_results.pt"
    save_path = os.path.join(output_dir, save_name)
    torch.save(results, save_path)
    logger.debug(f"[Typo Patch] Saved patched results => {save_path}")
    logger.warning("=== Main (typo) patched Inference Complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on MAIN typos with patched attention from normal.")
    parser.add_argument("--normal_attention_dir", type=str, required=True,
                        help="Directory containing the .pt files from the normal run.")
    parser.add_argument("--patch_scale", type=float, default=1.0,
                        help="How much to replace attention with normal? 1.0 = full replacement, 0.5 = blend, etc.")
    parser.add_argument("--patch_layers", type=str, default="all",
                        help="Comma-separated list of layers to patch, or 'all' to patch every layer.")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max_seq_length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    parser.add_argument("--top_k_logits", type=int, default=DEFAULT_TOP_K_LOGITS)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("polAIlogger")

    # Process patch_layers argument.
    if args.patch_layers.lower() == "all":
        patch_layers = list(range(DEFAULT_NUM_LAYERS))
    else:
        # Parse comma-separated list into integers.
        patch_layers = [int(x.strip()) for x in args.patch_layers.split(",")]

    # EXAMPLE: dummy data (typo version)
    dummy_data = [(i, f"Solve this tasck: 4 + 93 = ??? [Typo version {i}]") for i in range(10)]

    model, tokenizer = load_model(logger=logger)
    
    # Load the normal attention from the specified directory.
    normal_attn_map_all = load_normal_attention(args.normal_attention_dir, logger=logger)

    run_inf_main_patched(
        model=model,
        tokenizer=tokenizer,
        data=dummy_data,
        normal_attn_map_all=normal_attn_map_all,
        patch_scale=args.patch_scale,
        patch_layers=patch_layers,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        top_k_logits=args.top_k_logits,
        logger=logger
    )
