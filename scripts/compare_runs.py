#!/usr/bin/env python3
"""
Compare federated client updates across two methods.

Expected filenames:
  client_<x>_round_<y>.pt

Each .pt is expected to contain something like:
  {"delta_state_dict": {param_name: tensor, ...}, ...}
but the loader tries a few common fallbacks.

Key idea for signed metrics (e.g. "mean"):
- DO NOT normalize by sum(v) because it can be ~0 due to sign cancellations.
- Instead normalize by sum(abs(v)) to keep sign + remain stable:
    v_rel = v / (sum(abs(v)) + eps)

Outputs:
  - matched_pairs.csv
  - layer_metrics_long.csv
  - layer_metrics_wide.csv
  - Heatmaps per round (shared color scale per round):
      heatmap_round_<y>_{A,B,diff}.png
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


# ----------------------------
# Config / layer names
# ----------------------------

FNAME_RE = re.compile(r"client_(\d+)_round_(\d+)\.pt$")

LAYER_ORDER = [
    "conv1.weight", "conv1.bias",
    "conv2.weight", "conv2.bias",
    "fc1.weight", "fc1.bias",
    "fc2.weight", "fc2.bias",
]

EARLY_PARAMS = {"conv1.weight", "conv1.bias", "conv2.weight", "conv2.bias"}
LATE_PARAMS  = {"fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"}

EPS = 1e-12


# ----------------------------
# Helpers
# ----------------------------

@dataclass(frozen=True)
class Key:
    client: int
    round: int


def parse_filename(p: Path) -> Optional[Key]:
    m = FNAME_RE.search(p.name)
    if not m:
        return None
    return Key(client=int(m.group(1)), round=int(m.group(2)))


def index_dir(folder: Path) -> Dict[Key, Path]:
    idx: Dict[Key, Path] = {}
    for p in folder.glob("*.pt"):
        k = parse_filename(p)
        if k is None:
            continue
        idx[k] = p  # last wins on duplicates
    return idx


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def extract_state_dict(obj) -> Dict[str, torch.Tensor]:
    """
    Try to find the dict of parameter tensors in a loaded .pt.
    Priority: delta_state_dict -> state_dict -> model_state_dict -> object itself if dict[str, Tensor].
    """
    if isinstance(obj, dict):
        for key in ("delta_state_dict", "state_dict", "model_state_dict"):
            if key in obj and isinstance(obj[key], dict):
                if all(isinstance(v, torch.Tensor) for v in obj[key].values()):
                    return obj[key]
        if all(isinstance(k, str) for k in obj.keys()) and all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj  # type: ignore
    raise ValueError("Could not locate a tensor state dict (tried delta_state_dict/state_dict/model_state_dict).")


def layer_metric_value(t: torch.Tensor, metric: str) -> float:
    x = t.detach().float().cpu()

    if metric == "l2":
        return torch.norm(x).item()
    if metric == "l1":
        return torch.norm(x, p=1).item()
    if metric == "mean_abs":
        return x.abs().mean().item()
    if metric == "mean":
        return x.mean().item()  # signed
    if metric == "rms":
        return torch.sqrt((x * x).mean()).item()

    raise ValueError(f"Unknown metric: {metric}")


def compute_layer_metrics(state: Dict[str, torch.Tensor], metric: str) -> Dict[str, float]:
    return {name: layer_metric_value(t, metric) for name, t in state.items()}


def to_fixed_layer_vector(metrics: Dict[str, float], layers: List[str]) -> np.ndarray:
    vec = np.zeros(len(layers), dtype=np.float64)
    for i, l in enumerate(layers):
        vec[i] = float(metrics.get(l, 0.0))
    return vec


def normalize_profile(v: np.ndarray, signed: bool) -> Tuple[np.ndarray, float]:
    """
    Returns (v_rel, strength).
    - strength: scalar magnitude of the update profile over layers.
    - v_rel: normalized profile

    signed=False -> normalize by sum(v) (v is non-negative metrics like l2/mean_abs)
    signed=True  -> normalize by sum(abs(v)) to avoid cancellation
    """
    if signed:
        strength = float(np.sum(np.abs(v)))
        denom = strength + EPS
        return v / denom, strength
    else:
        strength = float(np.sum(v))
        denom = strength + EPS
        return v / denom, strength


def plot_heatmap_clients_layers(
    mat: np.ndarray,
    clients: List[int],
    layers: List[str],
    title: str,
    out_path: Path,
    diverging: bool,
    vmax: Optional[float] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(layers)), max(4, 0.35 * len(clients))))

    if diverging:
        if vmax is None:
            vmax = float(np.max(np.abs(mat))) if mat.size else 1.0
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        im = ax.imshow(mat, aspect="auto", interpolation="nearest", norm=norm, cmap="RdBu_r")
    else:
        im = ax.imshow(mat, aspect="auto", interpolation="nearest")

    ax.set_title(title)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Client")

    ax.set_xticks(np.arange(len(layers)))
    ax.set_xticklabels(layers, rotation=45, ha="right")

    ax.set_yticks(np.arange(len(clients)))
    ax.set_yticklabels(clients)

    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    plt.tight_layout()
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


# ----------------------------
# Main
# ----------------------------

def main():
    # ap = argparse.ArgumentParser()
    # ap.add_argument("--dir_a", type=str, required=True, help="Folder with method A updates")
    # ap.add_argument("--dir_b", type=str, required=True, help="Folder with method B updates")
    # ap.add_argument("--out", type=str, default="compare_out", help="Output folder")
    # ap.add_argument("--metric", type=str, default="l2", choices=["l2", "l1", "mean_abs", "rms"],
    #                 help="Magnitude metric per parameter tensor")
    # ap.add_argument("--round", type=int, default=None, help="If set, only process this round")
    # ap.add_argument("--weights_only", action="store_true", help="If set, ignore bias parameters")
    # args = ap.parse_args()
    import datetime
    dir_a = "./fl_runs/MNIST/label_swapping/incremental/1-2_5-6_bidirectional/update_datasets_run_2026-01-06_18-14-38_fedavg_25c10661-6755-4adb-aebc-ad56c9e78e86/updates/client_updates"
    dir_b = "./fl_runs/MNIST/rotation/incremental/all_classes_rot_45/update_datasets_run_2026-01-06_22-24-35_fedavg_f405cb53-c990-4946-95a0-0f3499d97d89/updates/client_updates"
    out_dir = f"./scripts/output/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    metric = "l2"
    round_arg = None
    weights_only = False

    dir_a = Path(dir_a)
    dir_b = Path(dir_b)
    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    layers = [l for l in LAYER_ORDER if (not weights_only or l.endswith(".weight"))]
    signed_metric = (metric == "mean")  # only mean is signed here

    idx_a = index_dir(dir_a)
    idx_b = index_dir(dir_b)

    keys = sorted(set(idx_a.keys()) & set(idx_b.keys()), key=lambda k: (k.round, k.client))
    if round_arg is not None:
        keys = [k for k in keys if k.round == round_arg]

    if not keys:
        raise SystemExit("No matching (client, round) pairs found between the two folders (or after filtering by --round).")

    # Collect records for CSV
    records: List[dict] = []

    # Per-round matrices for plotting (store normalized profiles)
    by_round: Dict[int, Dict[str, List[Tuple[int, np.ndarray]]]] = {}
    # Also store strengths (absolute scale) if you want later
    by_round_strength: Dict[int, Dict[str, List[Tuple[int, float]]]] = {}

    for k in keys:
        p_a = idx_a[k]
        p_b = idx_b[k]

        obj_a = torch.load(p_a, map_location="cpu")
        obj_b = torch.load(p_b, map_location="cpu")

        sd_a = extract_state_dict(obj_a)
        sd_b = extract_state_dict(obj_b)

        m_a = compute_layer_metrics(sd_a, metric=metric)
        m_b = compute_layer_metrics(sd_b, metric=metric)

        v_a = to_fixed_layer_vector(m_a, layers)
        v_b = to_fixed_layer_vector(m_b, layers)

        v_a_rel, strength_a = normalize_profile(v_a, signed=signed_metric)
        v_b_rel, strength_b = normalize_profile(v_b, signed=signed_metric)
        v_diff_rel = v_a_rel - v_b_rel

        # Early vs late score on normalized profiles
        early_idx = [i for i, l in enumerate(layers) if l in EARLY_PARAMS]
        late_idx  = [i for i, l in enumerate(layers) if l in LATE_PARAMS]
        early_a = float(v_a_rel[early_idx].sum()) if early_idx else 0.0
        late_a  = float(v_a_rel[late_idx].sum())  if late_idx else 0.0
        early_b = float(v_b_rel[early_idx].sum()) if early_idx else 0.0
        late_b  = float(v_b_rel[late_idx].sum())  if late_idx else 0.0

        early_late_score_a = early_a - late_a
        early_late_score_b = early_b - late_b

        # store long rows
        for i, layer in enumerate(layers):
            records.append({
                "round": k.round,
                "client": k.client,
                "layer": layer,
                "A_value": float(v_a[i]),
                "B_value": float(v_b[i]),
                "A_rel": float(v_a_rel[i]),
                "B_rel": float(v_b_rel[i]),
                "diff_rel": float(v_diff_rel[i]),
                "A_strength": strength_a,
                "B_strength": strength_b,
                "metric": metric,
                "file_a": str(p_a),
                "file_b": str(p_b),
                "early_late_score_A": early_late_score_a,
                "early_late_score_B": early_late_score_b,
            })

        by_round.setdefault(k.round, {"A": [], "B": [], "diff": []})
        by_round[k.round]["A"].append((k.client, v_a_rel))
        by_round[k.round]["B"].append((k.client, v_b_rel))
        by_round[k.round]["diff"].append((k.client, v_diff_rel))

        by_round_strength.setdefault(k.round, {"A": [], "B": []})
        by_round_strength[k.round]["A"].append((k.client, strength_a))
        by_round_strength[k.round]["B"].append((k.client, strength_b))

    # ----------------------------
    # Write CSVs (no pandas)
    # ----------------------------
    import csv

    # matched pairs
    with open(out_dir / "matched_pairs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["round", "client", "file_a", "file_b"])
        w.writeheader()
        for k in keys:
            w.writerow({"round": k.round, "client": k.client, "file_a": str(idx_a[k]), "file_b": str(idx_b[k])})

    # long metrics
    fieldnames = list(records[0].keys())
    with open(out_dir / "layer_metrics_long.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(records)

    # wide metrics (one row per (round, client))
    wide_rows: Dict[Tuple[int, int], dict] = {}
    for r in records:
        key = (r["round"], r["client"])
        wide_rows.setdefault(key, {
            "round": r["round"],
            "client": r["client"],
            "metric": r["metric"],
            "A_strength": r["A_strength"],
            "B_strength": r["B_strength"],
            "early_late_score_A": r["early_late_score_A"],
            "early_late_score_B": r["early_late_score_B"],
        })
        layer = r["layer"]
        wide_rows[key][f"A_value.{layer}"] = r["A_value"]
        wide_rows[key][f"B_value.{layer}"] = r["B_value"]
        wide_rows[key][f"A_rel.{layer}"] = r["A_rel"]
        wide_rows[key][f"B_rel.{layer}"] = r["B_rel"]
        wide_rows[key][f"diff_rel.{layer}"] = r["diff_rel"]

    wide_fieldnames = sorted({k for row in wide_rows.values() for k in row.keys()})
    with open(out_dir / "layer_metrics_wide.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=wide_fieldnames)
        w.writeheader()
        for key in sorted(wide_rows.keys()):
            w.writerow(wide_rows[key])

    # ----------------------------
    # Plot per-round heatmaps
    # ----------------------------
    # For signed metric, use diverging for A/B/diff.
    # For non-signed metric, use non-diverging for A/B and diverging for diff (still meaningful).
    for rnd, payload in sorted(by_round.items()):
        payload["A"].sort(key=lambda x: x[0])
        payload["B"].sort(key=lambda x: x[0])
        payload["diff"].sort(key=lambda x: x[0])

        clients = [c for c, _ in payload["A"]]
        mat_a = np.stack([v for _, v in payload["A"]], axis=0)
        mat_b = np.stack([v for _, v in payload["B"]], axis=0)
        mat_d = np.stack([v for _, v in payload["diff"]], axis=0)

        # Shared color scaling per round so colors are comparable between A/B/diff
        if signed_metric:
            vmax_round = float(np.max(np.abs(np.concatenate([mat_a.ravel(), mat_b.ravel(), mat_d.ravel()])))) or 1.0
        else:
            vmax_round = None  # not used for non-diverging

        plot_heatmap_clients_layers(
            mat_a, clients, layers,
            title=f"Round {rnd} — Method A ({metric}; normalized profile)",
            out_path=out_dir / f"heatmap_round_{rnd}_A.png",
            diverging=signed_metric,
            vmax=vmax_round,
        )
        plot_heatmap_clients_layers(
            mat_b, clients, layers,
            title=f"Round {rnd} — Method B ({metric}; normalized profile)",
            out_path=out_dir / f"heatmap_round_{rnd}_B.png",
            diverging=signed_metric,
            vmax=vmax_round,
        )
        plot_heatmap_clients_layers(
            mat_d, clients, layers,
            title=f"Round {rnd} — Diff (A − B) in normalized profile",
            out_path=out_dir / f"heatmap_round_{rnd}_diff.png",
            diverging=True,
            vmax=(vmax_round if signed_metric else None),
        )

    print(f"Done. Wrote outputs to: {out_dir.resolve()}")
    print("Key files:")
    print(f"  - {out_dir / 'matched_pairs.csv'}")
    print(f"  - {out_dir / 'layer_metrics_long.csv'}")
    print(f"  - {out_dir / 'layer_metrics_wide.csv'}")
    print("  - heatmap_round_<r>_{A,B,diff}.png")

if __name__ == "__main__":
    main()