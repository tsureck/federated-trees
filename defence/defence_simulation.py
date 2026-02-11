from __future__ import annotations

import datetime
import os
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd
import torch as th
from features import (
    EPS,
    calculate_p_per_layer,
    get_features_from_update,
    read_updates_from_round,
)
from strategies.mlp import MLPDefense
from strategies.share_fc2 import (
    classify_by_s_iqr,
    classify_by_s_kmeans,
    classify_by_s_lowerhalf_mad,
    classify_by_s_mad,
    classify_by_share_fc2_lowerhalf_mad,
    classify_by_share_fc2_threshold,
)


@dataclass
class DefenceVariant:
    """Bundles a defence strategy with its parameters."""

    name: str
    classify_fn: Callable
    needs_updates: bool = False
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class RoundGroundTruth:
    """Pre-computed per-client ground truth aligned to sorted client file order."""

    client_files: list[str]
    is_benign: list[bool]
    is_concept_drifted: list[bool]
    is_ls_drifted: list[bool]
    is_malicious: list[bool]
    is_drifted: list[bool]


def _run_id_from_dir(dir_name: str) -> str:
    """Extract run identifier from a client_updates directory path.

    Expects: .../fl_runs/MNIST/<drift_type>/<drift_pattern>/<parameter>/...
    Returns: '<drift_type>/<drift_pattern>/<parameter>'
    """
    parts = dir_name.replace("\\", "/").split("/")
    mnist_idx = parts.index("MNIST")
    return "/".join(parts[mnist_idx + 1 : mnist_idx + 4])


MLP_CHECKPOINTS = [
    "defence/mlp_defense_cf-0.5_R10-30.pt",
    "defence/mlp_defense_cf-0.5_cf-0.375_R10-20.pt",
    "defence/mlp_defense_cf-0.5_cf-0.375_R10-30.pt",
]


def _mlp_name_from_path(path: str) -> str:
    """Derive a short variant name from an MLP checkpoint filename.

    e.g. 'defence/mlp_defense_cf-0.5_R10-30.pt' -> 'mlp_cf-0.5_R10-30'
    """
    stem = os.path.splitext(os.path.basename(path))[0]  # mlp_defense_cf-0.5_R10-30
    return stem.replace("mlp_defense", "mlp", 1)


def build_defence_registry() -> list[DefenceVariant]:
    """Return all defence variants to evaluate."""
    variants: list[DefenceVariant] = []

    # 1. fc2_threshold (median_flag=True and False)
    for median_flag in [True, False]:
        label = "median" if median_flag else "mean"
        variants.append(
            DefenceVariant(
                name=f"fc2_threshold_{label}",
                classify_fn=classify_by_share_fc2_threshold,
                needs_updates=True,
                kwargs={"median_flag": median_flag},
            )
        )

    # 2. s_mad sweep
    for k in [1.0, 1.5, 2.0, 2.5, 3.0]:
        variants.append(
            DefenceVariant(
                name=f"s_mad_k{k}",
                classify_fn=classify_by_s_mad,
                kwargs={"k": k},
            )
        )

    # 3. s_kmeans sweep
    for margin in [0.0, 0.02, 0.05]:
        variants.append(
            DefenceVariant(
                name=f"s_kmeans_m{margin}",
                classify_fn=classify_by_s_kmeans,
                kwargs={"margin": margin},
            )
        )

    # 4. s_iqr sweep
    for alpha in [1.0, 1.5, 2.0, 2.5]:
        variants.append(
            DefenceVariant(
                name=f"s_iqr_a{alpha}",
                classify_fn=classify_by_s_iqr,
                kwargs={"alpha": alpha},
            )
        )

    # 5. s_lowerhalf_mad sweep
    for k in [1.5, 2.0, 2.5, 3.0]:
        variants.append(
            DefenceVariant(
                name=f"s_lowerhalf_mad_k{k}",
                classify_fn=classify_by_s_lowerhalf_mad,
                kwargs={"k": k},
            )
        )

    # 6. fc2_lowerhalf_mad sweep
    for k in [1.5, 2.0, 2.5, 3.0]:
        variants.append(
            DefenceVariant(
                name=f"fc2_lowerhalf_mad_k{k}",
                classify_fn=classify_by_share_fc2_lowerhalf_mad,
                kwargs={"k": k},
            )
        )

    # 7. mlp_defence (all pretrained checkpoints)
    for ckpt_path in MLP_CHECKPOINTS:
        mlp = MLPDefense.load(ckpt_path)
        variants.append(
            DefenceVariant(
                name=_mlp_name_from_path(ckpt_path),
                classify_fn=mlp.classify_features,
            )
        )

    return variants


def extract_round_data(
    round_idx: int, dir_name: str
) -> tuple[dict, dict, RoundGroundTruth]:
    """Extract features and ground truth for one round.

    Returns features dict with sorted keys, the raw updates dict,
    and a RoundGroundTruth aligned to the sorted key order.
    """
    updates = read_updates_from_round(dir_name, round_idx)

    # Compute per-round profiles for mean_p and median_p
    profiles = []
    for upd in updates.values():
        _, p = calculate_p_per_layer(upd["delta_state_dict"], ignore_bias=True)
        profiles.append(p)

    p_stack = th.stack(profiles)
    mean_p = p_stack.mean(dim=0)
    median_p = th.median(p_stack, dim=0).values
    mean_p /= mean_p.sum() + EPS
    median_p /= median_p.sum() + EPS

    # Build features dict with sorted keys so defence output order matches ground truth
    sorted_keys = sorted(updates.keys())
    features = {}
    for client_file in sorted_keys:
        features[client_file] = get_features_from_update(
            updates[client_file], mean_p, median_p
        )

    # Build ground truth aligned to sorted keys
    gt_benign = []
    gt_concept_drifted = []
    gt_ls_drifted = []
    gt_malicious = []
    gt_drifted = []

    for client_file in sorted_keys:
        upd = updates[client_file]
        drifted = bool(upd["is_drifted_client"] and upd["drift_applied"])
        malicious = bool(drifted and upd["malicious"])
        concept_drifted = bool(drifted and not upd["malicious"])

        gt_benign.append(not drifted)
        gt_concept_drifted.append(concept_drifted)
        gt_ls_drifted.append(malicious)
        gt_malicious.append(malicious)
        gt_drifted.append(drifted)

    ground_truth = RoundGroundTruth(
        client_files=sorted_keys,
        is_benign=gt_benign,
        is_concept_drifted=gt_concept_drifted,
        is_ls_drifted=gt_ls_drifted,
        is_malicious=gt_malicious,
        is_drifted=gt_drifted,
    )

    return features, updates, ground_truth


def compute_round_stats(
    round_idx: int,
    attack_type: str,
    run_id: str,
    defence_name: str,
    classified_labels: list[bool],
    ground_truth: RoundGroundTruth,
) -> dict:
    """Compute per-round classification statistics."""
    n_benign = sum(ground_truth.is_benign)
    n_concept_drifted = sum(ground_truth.is_concept_drifted)
    n_ls_drifted = sum(ground_truth.is_ls_drifted)

    benign_correct = 0  # TN: benign and not flagged
    benign_flagged = 0  # FP: benign but flagged
    concept_drifted_correct = 0  # TN: concept drifted and not flagged
    concept_drifted_flagged = 0  # FP: concept drifted but flagged
    ls_drifted_flagged = 0  # TP: LS drifted and flagged
    ls_drifted_missed = 0  # FN: LS drifted but not flagged

    for i, flagged in enumerate(classified_labels):
        if ground_truth.is_benign[i]:
            if flagged:
                benign_flagged += 1
            else:
                benign_correct += 1
        elif ground_truth.is_concept_drifted[i]:
            if flagged:
                concept_drifted_flagged += 1
            else:
                concept_drifted_correct += 1
        elif ground_truth.is_ls_drifted[i]:
            if flagged:
                ls_drifted_flagged += 1
            else:
                ls_drifted_missed += 1

    return {
        "round": round_idx,
        "defence": defence_name,
        "attack_type": attack_type,
        "run_id": run_id,
        "n_benign": n_benign,
        "benign_correct": benign_correct,
        "benign_flagged": benign_flagged,
        "n_concept_drifted": n_concept_drifted,
        "concept_drifted_correct": concept_drifted_correct,
        "concept_drifted_flagged": concept_drifted_flagged,
        "n_ls_drifted": n_ls_drifted,
        "ls_drifted_flagged": ls_drifted_flagged,
        "ls_drifted_missed": ls_drifted_missed,
    }


def run_defence_on_round(
    variant: DefenceVariant,
    features: dict,
    updates: dict,
    ground_truth: RoundGroundTruth,
    round_idx: int,
    attack_type: str,
    run_id: str,
) -> dict:
    """Run a single defence variant on one round and return stats."""
    if variant.needs_updates:
        classified = variant.classify_fn(features, updates, **variant.kwargs)
    else:
        classified = variant.classify_fn(features, **variant.kwargs)

    return compute_round_stats(
        round_idx, attack_type, run_id, variant.name, classified, ground_truth
    )


def main():
    """Run all defence variants across all rounds and attack types, saving results to CSV."""
    # dir_name_ls = "./fl_runs/MNIST/label_swapping/incremental/5-6_bidirectional_cf-0.5/update_datasets_run_2026-01-26_09-27-18_fedavg_cb1f65f9-7fdf-4b9d-a762-718ab4f021d1/updates/client_updates"  # noqa: E501
    # dir_name_rot = "./fl_runs/MNIST/rotation/incremental/all_classes_rot_65_cf-0.5/update_datasets_run_2026-01-26_09-29-33_fedavg_cb1f65f9-7fdf-4b9d-a762-718ab4f021d1/updates/client_updates"  # noqa: E501
    dir_name_ls = "./fl_runs/MNIST/label_swapping/incremental/5-6_bidirectional_cf-0.5/update_datasets_run_2026-01-26_09-27-18_fedavg_cb1f65f9-7fdf-4b9d-a762-718ab4f021d1/updates/client_updates"  # noqa: E501
    dir_name_rot = "./fl_runs/MNIST/rotation/incremental/all_classes_rot_65_cf-0.5/update_datasets_run_2026-01-26_09-29-33_fedavg_cb1f65f9-7fdf-4b9d-a762-718ab4f021d1/updates/client_updates"  # noqa: E501

    registry = build_defence_registry()

    all_rows: list[dict] = []

    attack_configs = [
        ("label_swapping", dir_name_ls),
        ("rotation", dir_name_rot),
    ]

    for attack_type, dir_name in attack_configs:
        run_id = _run_id_from_dir(dir_name)
        for round_idx in range(40):
            print(f"--- {run_id} round {round_idx} ---")
            features, updates, gt = extract_round_data(round_idx, dir_name)

            for variant in registry:
                row = run_defence_on_round(
                    variant, features, updates, gt, round_idx, attack_type, run_id
                )
                all_rows.append(row)

    df = pd.DataFrame(all_rows)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = f"defence/defence_stats_{timestamp}.csv"
    df.to_csv(out_path, index=False, sep=";")
    print(f"Saved {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
