import numpy as np
import torch as th
from features import (
    EPS,
    calculate_p_per_layer,
    get_features_from_update,
    read_updates_from_round,
)
from strategies.share_fc2 import classify_by_s_iqr, classify_by_s_kmeans, classify_by_s_lowerhalf_mad, classify_by_s_mad, classify_by_share_fc2_lowerhalf_mad, classify_by_share_fc2_threshold


def analyze_client_update_round(round_idx: int, dir_name: str) -> float:
    """Analyze client updates for a given round index."""
    updates = read_updates_from_round(dir_name, round_idx)

    profiles = []
    for _, upd in updates.items():
        _, p = calculate_p_per_layer(upd["delta_state_dict"], ignore_bias=True)
        profiles.append(p)

    p = th.stack(profiles)
    mean_p = p.mean(dim=0)
    median_p = th.median(p, dim=0).values
    mean_p /= mean_p.sum() + EPS
    median_p /= median_p.sum() + EPS

    features = {}

    # --- extract features per client ---
    for client_file, upd in updates.items():
        feats = get_features_from_update(upd, mean_p, median_p)
        features[client_file] = feats

    # INSERT DEFENSE STRATEGIES HERE
    return classify_malicious_drift_clients(features, updates)


def analyze_classification_performance(
    actual_labels: list[bool], drifted_clients: list[bool], classified_labels: list[bool]
) -> None:
    """Analyze the performance of a classification system."""
    # Calculate accuracy
    correct = sum(1 for a, c in zip(actual_labels, classified_labels) if a == c)
    total_accuracy = (correct, len(actual_labels) if actual_labels else 0)

    drifted_client_accuracy = (
        sum(
            1 for a, c, d in zip(actual_labels, classified_labels, drifted_clients) if d and a == c
        ),
        (sum(drifted_clients) if sum(drifted_clients) > 0 else 0),
    )

    benign_client_accuracy = (
        sum(
            1
            for a, c, d in zip(actual_labels, classified_labels, drifted_clients)
            if not d and a == c
        ),
        (sum(not d for d in drifted_clients) if sum(not d for d in drifted_clients) > 0 else 0),
    )

    return total_accuracy, drifted_client_accuracy, benign_client_accuracy


def classify_malicious_drift_clients(features: dict, updates: dict) -> float:
    """Classify clients as malicious or benign based on a defence strategy."""
    clients = [
        (upd["is_drifted_client"] and upd["drift_applied"] and upd["malicious"])
        for _, upd in updates.items()
    ]
    drifted_clients = [
        (upd["is_drifted_client"] and upd["drift_applied"]) for _, upd in updates.items()
    ]

    # malicious = classify_by_share_fc2_threshold(features, updates, median_flag=True)
    # malicious = classify_by_s_mad(features, k=1.5)
    # malicious = classify_by_s_kmeans(features)
    # malicious = classify_by_s_iqr(features)
    # malicious = classify_by_s_lowerhalf_mad(features)
    malicious = classify_by_share_fc2_lowerhalf_mad(features)

    return analyze_classification_performance(clients, drifted_clients, malicious)


def main():
    """Main function to analyze client updates across multiple rounds."""
    dir_name_ls = "./fl_runs/MNIST/label_swapping/incremental/5-6_bidirectional/update_datasets_run_2026-01-18_22-44-53_fedavg_cb1f65f9-7fdf-4b9d-a762-718ab4f021d1/updates/client_updates"  # noqa: E501
    dir_name_rot = "./fl_runs/MNIST/rotation/incremental/all_classes_rot_65/update_datasets_run_2026-01-19_00-04-07_fedavg_cb1f65f9-7fdf-4b9d-a762-718ab4f021d1/updates/client_updates"  # noqa: E501

    round_accuracies = ([], [])

    for i in range(40):
        print(f"--- Analyzing round {i} ---")
        round_accuracies[0].append(analyze_client_update_round(i, dir_name_ls))
        round_accuracies[1].append(analyze_client_update_round(i, dir_name_rot))

    averaged_accuracies = np.sum(round_accuracies, axis=1)
    averaged_accuracies_percentage = [
        averaged_accuracies[0][:, 0] / averaged_accuracies[0][:, 1],
        averaged_accuracies[1][:, 0] / averaged_accuracies[1][:, 1],
    ]
    print("Final Results:")
    print(
        "Label-Swapping Attack Accuracies: "
        + f"Total Acc: {(averaged_accuracies_percentage[0][0]):.2%} | "
        + f"Drifted Acc: {(averaged_accuracies_percentage[0][1]):.2%} | "
        + f"Benign Acc: {(averaged_accuracies_percentage[0][2]):.2%}"
    )
    print(
        "Rotation Benign Drift Accuracies: "
        + f"Total Acc: {(averaged_accuracies_percentage[1][0]):.2%} | "
        + f"Drifted Acc: {(averaged_accuracies_percentage[1][1]):.2%} | "
        + f"Benign Acc: {(averaged_accuracies_percentage[1][2]):.2%}"
    )
    return averaged_accuracies_percentage


if __name__ == "__main__":
    main()
