#!/usr/bin/env python
import os
import glob
import torch
import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

def load_features_from_dir(directory, layer, aggregation="mean"):
    """
    Loads activation features for a given layer from all .pt files in a directory.
    For each file, it averages the activations over the sequence dimension.

    Parameters:
      - directory: path to the folder containing .pt files.
      - layer: integer, the layer index (e.g., 0, 5, 10, 15).
      - aggregation: method to aggregate over sequence dimension; currently only "mean" is supported.

    Returns:
      - features: numpy array of shape (n_samples, hidden_dim)
    """
    pt_files = sorted(glob.glob(os.path.join(directory, "activations_*.pt")))
    all_features = []
    
    for pt_file in pt_files:
        try:
            data = torch.load(pt_file, map_location="cpu")
        except Exception as e:
            print(f"Could not load {pt_file}: {e}")
            continue

        # Expecting a dictionary "hidden_states" with keys like "layer_0", "layer_5", etc.
        key = f"layer_{layer}"
        if key not in data["hidden_states"]:
            print(f"File {pt_file} does not contain {key}. Skipping.")
            continue
        
        # data["hidden_states"][key] is assumed to be a tensor of shape [batch, seq_length, hidden_dim]
        activations = data["hidden_states"][key]
        # Aggregate over sequence dimension: result is [batch, hidden_dim]
        if aggregation == "mean":
            features = activations.mean(dim=1)  # average pooling over tokens
        else:
            raise ValueError("Unsupported aggregation method")
        
        # Convert to numpy array and append per sample
        #all_features.append(features.numpy())
        all_features.append(features.float().numpy())
    
    if len(all_features) == 0:
        raise ValueError(f"No features loaded from {directory} for layer {layer}")
    # Concatenate over batches
    return np.concatenate(all_features, axis=0)

def train_and_evaluate(features, labels, cv=5):
    """
    Trains a logistic regression classifier using cross-validation and returns the average accuracy.
    
    Parameters:
      - features: numpy array of shape (n_samples, hidden_dim)
      - labels: numpy array of shape (n_samples,)
      - cv: number of folds for cross-validation (default: 5)
    
    Returns:
      - mean_accuracy: float, the average cross-validation accuracy.
    """
    clf = LogisticRegression(max_iter=1000, solver="lbfgs")
    scores = cross_val_score(clf, features, labels, cv=cv, scoring="accuracy")
    return scores.mean()

def main():
    parser = argparse.ArgumentParser(
        description="Train linear classifiers on activation features to differentiate conditions."
    )
    parser.add_argument("--dir1", type=str, required=True,
                        help="Directory containing .pt files for condition 1 (e.g., normal prompts)")
    parser.add_argument("--dir2", type=str, required=True,
                        help="Directory containing .pt files for condition 2 (e.g., nicer prompts)")
    parser.add_argument("--layers", type=str, default="0,5,10,15",
                        help="Comma-separated list of layer indices to evaluate (default: 0,5,10,15)")
    parser.add_argument("--cv", type=int, default=5,
                        help="Number of cross-validation folds (default: 5)")
    args = parser.parse_args()

    layer_list = [int(x.strip()) for x in args.layers.split(",") if x.strip().isdigit()]
    
    # For each condition, load features for each layer.
    # We'll assign label 0 for dir1 and label 1 for dir2.
    results = {}
    for layer in layer_list:
        print(f"Processing layer {layer}...")
        try:
            features1 = load_features_from_dir(args.dir1, layer)
            features2 = load_features_from_dir(args.dir2, layer)
        except ValueError as e:
            print(f"Skipping layer {layer}: {e}")
            continue
        
        # Create labels for each condition
        labels1 = np.zeros(features1.shape[0], dtype=int)
        labels2 = np.ones(features2.shape[0], dtype=int)
        
        # Combine the features and labels
        X = np.concatenate([features1, features2], axis=0)
        y = np.concatenate([labels1, labels2], axis=0)
        
        mean_acc = train_and_evaluate(X, y, cv=args.cv)
        results[layer] = mean_acc
        print(f"Layer {layer}: Mean cross-validation accuracy = {mean_acc*100:.2f}%")
    
    print("Summary of results:")
    for layer, acc in results.items():
        print(f"Layer {layer}: {acc*100:.2f}% accuracy")

if __name__ == "__main__":
    main()
