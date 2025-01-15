#!/usr/bin/env python
import os
import glob
import re
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

def get_pt_files(directory):
    """Return sorted list of .pt files in a directory."""
    files = glob.glob(os.path.join(directory, "activations_*.pt"))
    return sorted(files)

def load_activations(pt_file):
    """Loads a single .pt file."""
    return torch.load(pt_file)

def identify_token_ranges(token_ids, tokenizer):
    """
    Identifies token ranges based on the first occurrence of a colon.
    
    Returns a dict with:
      - decoded_tokens: list of tokens.
      - nice_mask: a mask (all False) since no special detection is performed.
      - math_mask: a mask that is True for tokens after the first colon.
      - colon_pos: the index of the first colon (or None if not found).
    """
    decoded_tokens = tokenizer.convert_ids_to_tokens(token_ids, skip_special_tokens=False)
    colon_positions = [i for i, tok in enumerate(decoded_tokens) if ":" in tok]
    first_colon = colon_positions[0] if colon_positions else None
    nice_mask = [False] * len(decoded_tokens)
    math_mask = [False] * len(decoded_tokens)
    if first_colon is not None and first_colon < len(decoded_tokens) - 1:
        for idx in range(first_colon + 1, len(decoded_tokens)):
            math_mask[idx] = True
    return {
        "decoded_tokens": decoded_tokens,
        "nice_mask": nice_mask,
        "math_mask": math_mask,
        "colon_pos": first_colon
    }

def make_prompt_task_masks(colon_pos, seq_len):
    prompt_mask = np.zeros(seq_len, dtype=bool)
    task_mask = np.zeros(seq_len, dtype=bool)
    if colon_pos is not None and colon_pos < seq_len:
        prompt_mask[:colon_pos+1] = True
        if colon_pos+1 < seq_len:
            task_mask[colon_pos+1:] = True
    else:
        task_mask[:] = True
    return prompt_mask, task_mask

def extract_attention_stats(attentions, input_ids):
    stats_per_layer = {}
    for layer_name, attn_tensor in attentions.items():
        attn_sum = attn_tensor.sum(dim=2)
        attn_mean = attn_sum.mean(dim=1)
        stats_per_layer[layer_name] = attn_mean
    layer_list = list(stats_per_layer.values())
    all_layers = torch.stack(layer_list, dim=0)
    avg_across_layers = all_layers.mean(dim=0)
    return {
        "per_layer": stats_per_layer,
        "avg_layers": avg_across_layers,
        "input_ids": input_ids
    }

def extract_per_layer_head_stats(attentions, input_ids):
    results = []
    sorted_layers = sorted(attentions.keys(), key=lambda x: int(x.split('_')[1]) if '_' in x else 0)
    B, S = input_ids.shape
    for layer_name in sorted_layers:
        attn_tensor = attentions[layer_name]
        attn_sum = attn_tensor.sum(dim=3)
        attn_mean = attn_tensor.mean(dim=3)
        layer_idx = int(layer_name.split('_')[1]) if '_' in layer_name else layer_name
        b_size, h_size, s_len, s_len2 = attn_tensor.shape
        if b_size != B or s_len != S or s_len2 != S:
            raise ValueError("Mismatch shape...")
        for b_idx in range(B):
            for h_idx in range(h_size):
                row = {
                    "layer": layer_idx,
                    "head": h_idx,
                    "batch_idx": b_idx,
                    "attn_sum": attn_sum[b_idx, h_idx].cpu().tolist(),
                    "attn_mean": attn_mean[b_idx, h_idx].cpu().tolist()
                }
                results.append(row)
    return results

def run_attention_extraction(tokenizer,
                             extraction_base_dir="c_inference/extractions/",
                             prompt_types="normal main swap",
                             output_dir="output/attention"):
    """
    Runs the full attention extraction analysis.

    Parameters:
      - tokenizer: a preloaded Hugging Face tokenizer.
      - extraction_base_dir: Base directory containing extraction subdirectories.
           Default is "c_inference/extractions/".
      - prompt_types: Space-separated string of directory names to process.
           Default is "normal main swap".
      - output_dir: Directory to save attention analysis outputs.
           The final output directory will be: output_dir/<joined_prompt_types>
    
    The function processes all .pt files in each extraction subdirectory, computes attention stats,
    saves CSV summaries and boxplots, and prints summary statistics.
    """
    # Build list of prompt types and corresponding directories.
    prompt_types_list = [pt.strip() for pt in prompt_types.split() if pt.strip()]
    all_dirs_list = [os.path.join(extraction_base_dir, pt) for pt in prompt_types_list]
    # Build output directory: join output_dir with the prompt type names.
    joined_prompts = "_".join(prompt_types_list)
    final_output_dir = os.path.join(output_dir, joined_prompts)
    os.makedirs(final_output_dir, exist_ok=True)
    
    # Use the simplified detection function.
    identify_fn = lambda token_ids: identify_token_ranges(token_ids, tokenizer)
    
    all_attention_records = []
    all_plh_records = []
    for ddir in all_dirs_list:
        pt_files = get_pt_files(ddir)
        prompt_type = os.path.basename(os.path.normpath(ddir))
        print(f"Found {len(pt_files)} .pt files in: {ddir} (prompt_type='{prompt_type}')")
        
        for pt_file in pt_files:
            data = load_activations(pt_file)
            # Aggregate attention stats.
            batch_stats = extract_attention_stats(data["attentions"], data["input_ids"])
            B, S = batch_stats["avg_layers"].shape
            for i in range(B):
                attn_vals = batch_stats["avg_layers"][i].float().cpu().numpy()  # shape [S]
                masks_info = identify_fn(data["input_ids"][i].tolist())
                colon_pos = masks_info["colon_pos"]
                prompt_mask, task_mask = make_prompt_task_masks(colon_pos, S)
                total_sum = attn_vals.sum()
                prompt_sum = attn_vals[prompt_mask].sum() if prompt_mask.any() else 0.0
                task_sum = attn_vals[task_mask].sum() if task_mask.any() else 0.0
                frac_prompt = (prompt_sum / total_sum) if total_sum > 1e-10 else 0.0
                frac_task = (task_sum / total_sum) if total_sum > 1e-10 else 0.0
                all_attention_records.append({
                    "batch_file": pt_file,
                    "prompt_type": prompt_type,
                    "avg_attn_all": float(attn_vals.mean()),
                    "prompt_sum": float(prompt_sum),
                    "task_sum": float(task_sum),
                    "frac_prompt": float(frac_prompt),
                    "frac_task": float(frac_task),
                })
            # Per-layer-head stats.
            plh_data = extract_per_layer_head_stats(data["attentions"], data["input_ids"])
            for row in plh_data:
                b_idx = row["batch_idx"]
                colon_pos = identify_fn(data["input_ids"][b_idx].tolist())["colon_pos"]
                prompt_mask, task_mask = make_prompt_task_masks(colon_pos, S)
                attn_sum_array = np.array(row["attn_sum"], dtype=np.float32)
                total_sum = attn_sum_array.sum()
                prompt_sum = attn_sum_array[prompt_mask].sum() if prompt_mask.any() else 0.0
                task_sum = attn_sum_array[task_mask].sum() if task_mask.any() else 0.0
                frac_prompt = (prompt_sum / total_sum) if total_sum > 1e-10 else 0.0
                frac_task = (task_sum / total_sum) if total_sum > 1e-10 else 0.0
                all_plh_records.append({
                    "batch_file": pt_file,
                    "prompt_type": prompt_type,
                    "layer": row["layer"],
                    "head": row["head"],
                    "batch_idx": b_idx,
                    "sum_all": float(total_sum),
                    "sum_prompt": float(prompt_sum),
                    "sum_task": float(task_sum),
                    "frac_prompt": float(frac_prompt),
                    "frac_task": float(frac_task),
                })
    
    # Build aggregated DataFrame and save CSV.
    df_all = pd.DataFrame(all_attention_records)
    agg_mean = df_all.groupby("prompt_type")[["frac_prompt", "frac_task"]].mean()
    print("=== Average fraction of attention on prompt vs. task portion ===")
    print(agg_mean)
    agg_csv = os.path.join(final_output_dir, "prompt_task_fraction_aggregate.csv")
    agg_mean.to_csv(agg_csv)
    print(f"Saved fraction summary CSV to: {agg_csv}")
    
    # Boxplot for fraction on prompt.
    plt.figure()
    df_all.boxplot(column="frac_prompt", by="prompt_type", grid=False)
    plt.suptitle("")
    plt.title("Fraction of Attention on Prompt Portion")
    plt.ylabel("Fraction (0~1)")
    plt.xlabel("Prompt Type")
    boxplot_prompt = os.path.join(final_output_dir, "boxplot_frac_prompt.png")
    plt.savefig(boxplot_prompt)
    plt.show()  # To display inline in notebooks
    plt.close()
    
    # Boxplot for fraction on task.
    plt.figure()
    df_all.boxplot(column="frac_task", by="prompt_type", grid=False)
    plt.suptitle("")
    plt.title("Fraction of Attention on Task Portion")
    plt.ylabel("Fraction (0~1)")
    plt.xlabel("Prompt Type")
    boxplot_task = os.path.join(final_output_dir, "boxplot_frac_task.png")
    plt.savefig(boxplot_task)
    plt.show()  # Display inline
    plt.close()
    
    # Per-layer-head stats.
    df_plh = pd.DataFrame(all_plh_records).drop_duplicates()
    plh_mean = df_plh.groupby(["prompt_type", "layer"])[["frac_prompt", "frac_task"]].mean()
    print("=== Per-layer average fraction of attention on prompt vs. task ===")
    print(plh_mean)
    plh_csv = os.path.join(final_output_dir, "prompt_task_fraction_perlayer.csv")
    plh_mean.to_csv(plh_csv)
    print(f"Saved per-layer fraction CSV to: {plh_csv}")
    
    # For each prompt_type, create boxplots of frac_prompt by layer.
    for pt in df_plh["prompt_type"].unique():
        sub = df_plh[df_plh["prompt_type"] == pt]
        if not sub.empty:
            plt.figure()
            sub.boxplot(column="frac_prompt", by="layer", grid=False)
            plt.suptitle("")
            plt.title(f"Fraction of Attention on Prompt vs. Layer - {pt}")
            plt.ylabel("Fraction (0~1)")
            plt.xlabel("Layer")
            boxplot_pl = os.path.join(final_output_dir, f"boxplot_perlayer_frac_prompt_{pt}.png")
            plt.savefig(boxplot_pl)
            plt.show()  # Display inline
            plt.close()
    
    print("All plots and CSVs saved to:", final_output_dir)
    print("Done!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Run attention extraction analysis from .pt activation files."
    )
    parser.add_argument("--extraction_base_dir", type=str, default="c_inference/extractions/",
                        help="Base directory where extraction subdirectories are located")
    parser.add_argument("--prompt_types", type=str, default="normal main swap",
                        help="Space-separated list of prompt type directory names")
    parser.add_argument("--output_dir", type=str, default="output/attention",
                        help="Directory to save attention analysis outputs")
    args = parser.parse_args()

    # In SLURM or command-line usage, load a tokenizer externally and pass its name
    # For now, loading one:
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it", use_auth_token=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    run_attention_extraction(
        tokenizer=tokenizer,
        extraction_base_dir=args.extraction_base_dir,
        prompt_types=args.prompt_types,
        output_dir=args.output_dir
    )
