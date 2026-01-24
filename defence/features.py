import os
import re

import torch as th
from torch.nn import functional

EPS = 1e-12

R_UPDATE_ROUND = re.compile(r"client_(\d+)_round_(\d+)\.pt")


def read_updates_from_round(dir_name: str, round_idx: int):
    """Return {filename: loaded_update} for a given round."""
    updates = os.listdir(dir_name)

    round_updates = []
    for u in updates:
        m = R_UPDATE_ROUND.match(u)
        if m and int(m.groups()[1]) == round_idx:
            round_updates.append(u)

    model_updates = {}
    for update_file in round_updates:
        model_updates[update_file] = th.load(
            os.path.join(dir_name, update_file), map_location="cpu"
        )

    return model_updates


def calculate_p_per_layer(delta_state_dict: dict, ignore_bias: bool = True):
    """Build a layer-profile vector p (shares per layer) using mean_abs per layer.

    Returns:
      layers: list[str]
      p: tensor shape (num_layers,) with sum(p)=1
    """
    layers = []
    vals = []

    for layer, data in delta_state_dict.items():
        if ignore_bias and "bias" in layer:
            continue
        vals.append(data.detach().norm(p=2))  # scalar tensor
        # vals.append(data.detach().abs().mean())  # scalar tensor
        layers.append(layer)

    v = th.stack(vals)  # shape (L,)
    p = v / (v.sum() + EPS)
    return layers, p


def entropy_from_probs(p: th.Tensor, normalized: bool = True) -> float:
    """p: probabilities that sum to 1 (non-negative)
    normalized=True -> divide by log(K) => entropy in [0,1]
    """
    h = -(p * (p + EPS).log()).sum()
    if not normalized:
        return float(h.item())
    h_max = th.log(th.tensor(p.numel(), dtype=p.dtype))
    return float((h / (h_max + EPS)).item())


def get_features_from_update(update: dict, mean_p: th.Tensor):
    """Extract features from one update.
    mean_p must be computed across the round first.
    """
    layers, p = calculate_p_per_layer(update["delta_state_dict"], ignore_bias=True)

    # global update strength (overall magnitude)
    global_l2 = float(th.norm(update["delta_vector"], 2).item())

    # entropy of the layer-profile (how concentrated is the update?)
    ent = entropy_from_probs(p, normalized=True)

    # head ratio: fc vs conv share
    head_sum = p[th.tensor([("fc" in l) for l in layers])].sum()
    conv_sum = p[th.tensor([("conv" in l) for l in layers])].sum()
    head_ratio = float((head_sum / (conv_sum + EPS)).item())

    # cosine similarity to round mean profile
    cos_to_mean = float(functional.cosine_similarity(p, mean_p, dim=0).item())

    return {
        "global_l2": global_l2,
        "entropy": ent,
        "head_ratio": head_ratio,
        "cos_to_mean": cos_to_mean,
        # Optional: direct share features are often very useful too:
        "share_fc2": float(p[layers.index("fc2.weight")].item())
        if "fc2.weight" in layers
        else None,
        "share_fc1": float(p[layers.index("fc1.weight")].item())
        if "fc1.weight" in layers
        else None,
        "share_conv1": float(p[layers.index("conv1.weight")].item())
        if "conv1.weight" in layers
        else None,
        "share_conv2": float(p[layers.index("conv2.weight")].item())
        if "conv2.weight" in layers
        else None,
    }


def analyze_client_update_round(round_idx: int, dir_name: str) -> float:
    """Analyze client updates for a given round index."""
    updates = read_updates_from_round(dir_name, round_idx)

    # --- compute mean profile for the round ---
    profiles = []
    for _, upd in updates.items():
        _, p = calculate_p_per_layer(upd["delta_state_dict"], ignore_bias=True)
        profiles.append(p)

    P = th.stack(profiles)  # (num_clients, num_layers)
    mean_p = P.mean(dim=0)
    mean_p /= mean_p.sum() + EPS  # re-normalize (safe)

    features = {}

    # --- extract features per client ---
    for client_file, upd in updates.items():
        feats = get_features_from_update(upd, mean_p)
        # print(f"{client_file} (is_drifted_client={upd['is_drifted_client']}): {feats}")
        features[client_file] = feats

    # INSERT DEFENSE STRATEGIES HERE
    return classify_malicious_drift_clients(features, updates)


def analyze_classification_performance(actual_labels: list[bool], drifted_clients: list[bool], classified_labels: list[bool]) -> None:
    """Analyze the performance of a classification system."""
    # Calculate accuracy
    correct = sum(1 for a, c in zip(actual_labels, classified_labels) if a == c)
    total_accuracy = correct / len(actual_labels) if actual_labels else 0

    drifted_client_accuracy = sum(
        1 for a, c, d in zip(actual_labels, classified_labels, drifted_clients) if d and a == c
    ) / (sum(drifted_clients) if sum(drifted_clients) > 0 else 1)

    benign_client_accuracy = sum(
        1 for a, c, d in zip(actual_labels, classified_labels, drifted_clients) if not d and a == c
    ) / (sum(not d for d in drifted_clients) if sum(not d for d in drifted_clients) > 0 else 1)

    print(f"Accuracy: {total_accuracy:.2%} | Drifted Client Accuracy: {drifted_client_accuracy:.2%} | Benign Client Accuracy: {benign_client_accuracy:.2%}")
    return total_accuracy, drifted_client_accuracy, benign_client_accuracy


def classify_malicious_drift_clients(features: dict, updates: dict) -> float:
    """Classify clients as malicious or benign based on a defence strategy."""
    clients = [(upd["is_drifted_client"] and upd["drift_applied"] and upd["malicious"]) for _, upd in updates.items()]
    drifted_clients = [(upd["is_drifted_client"] and upd["drift_applied"]) for _, upd in updates.items()]

    malicious = classify_by_share_fc2_threshold(features, updates, median_flag=True)

    return analyze_classification_performance(clients, drifted_clients, malicious)


def classify_by_share_fc2_threshold(features: dict, updates: dict, median_flag: bool = True) -> None:
    """Classify clients as malicious if their share_fc2 exceeds a threshold."""
    share_fc2_median = th.median(
        th.tensor([feat["share_fc2"] for _, feat in features.items()])
    ).item()
    share_fc2_mean = th.mean(th.tensor([feat["share_fc2"] for _, feat in features.items()])).item()

    threshold = (share_fc2_mean - 0.5 * (share_fc2_mean - share_fc2_median)) / 10
    # print(threshold)
    # print([(feat["share_fc2"] - share_fc2_median) for _, feat in features.items()])
    # print([(feat["share_fc2"] - share_fc2_mean) for _, feat in features.items()])
    # print([(feat["share_fc2"] - share_fc2_mean) > threshold for _, feat in features.items()])
    base = share_fc2_median if median_flag else share_fc2_mean
    malicious = [
        (feat["share_fc2"] - base) > threshold
        for _, feat in features.items()]

    return malicious


if __name__ == "__main__":
    dir_name_ls = "./fl_runs/MNIST/label_swapping/incremental/5-6_bidirectional/update_datasets_run_2026-01-18_22-44-53_fedavg_cb1f65f9-7fdf-4b9d-a762-718ab4f021d1/updates/client_updates"
    dir_name_rot = "./fl_runs/MNIST/rotation/incremental/all_classes_rot_65/update_datasets_run_2026-01-19_00-04-07_fedavg_cb1f65f9-7fdf-4b9d-a762-718ab4f021d1/updates/client_updates"

    round_accuracies = ([], [])

    for i in range(10, 30):
        print(f"--- Analyzing round {i} ---")
        print("Label-Swapping Attack:")
        round_accuracies[0].append(analyze_client_update_round(i, dir_name_ls))
        print("Rotation Benign Drift:")
        round_accuracies[1].append(analyze_client_update_round(i, dir_name_rot))

    import numpy as np
    averaged_accuracies = np.mean(round_accuracies, axis=1)
    print("Final Results:")
    print(f"Label-Swapping Attack Accuracies: Total Acc: {averaged_accuracies[0][0]:.2%} | Drifted Acc: {averaged_accuracies[0][1]:.2%} | Benign Acc: {averaged_accuracies[0][2]:.2%}")
    print(f"Rotation Benign Drift Accuracies: Total Acc: {averaged_accuracies[1][0]:.2%} | Drifted Acc: {averaged_accuracies[1][1]:.2%} | Benign Acc: {averaged_accuracies[1][2]:.2%}")
    pass
