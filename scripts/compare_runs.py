#!/usr/bin/env python3
"""
Compare federated client updates across two methods.

Expected filenames:
  client_<x>_round_<y>.pt

Each .pt is expected to contain something like:
  {"delta_state_dict": {param_name: tensor, ...}, ...}
but the loader tries a few common fallbacks.

Outputs:
  - matched_pairs.csv
  - layer_metrics_long.csv
  - layer_metrics_wide.csv
  - Heatmaps per round: heatmap_round_<y>_{A,B,diff}.png
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


FNAME_RE = re.compile(r"client_(\d+)_round_(\d+)\.pt$")


LAYER_ORDER = [
    "conv1.weight", "conv1.bias",
    "conv2.weight", "conv2.bias",
    "fc1.weight", "fc1.bias",
    "fc2.weight", "fc2.bias",
]


EARLY_PARAMS = {"conv1.weight", "conv1.bias", "conv2.weight", "conv2.bias"}
LATE_PARAMS = {"fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"}


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
        # last one wins if duplicates exist
        idx[k] = p
    return idx


def extract_state_dict(obj) -> Dict[str, torch.Tensor]:
    """
    Try to find the dict of parameter tensors in a loaded .pt.
    Priority: delta_state_dict -> state_dict -> model_state_dict -> the object itself if dict[str, Tensor].
    """
    if isinstance(obj, dict):
        for key in ("delta_state_dict", "state_dict", "model_state_dict"):
            if key in obj and isinstance(obj[key], dict):
                # ensure tensors
                if all(isinstance(v, torch.Tensor) for v in obj[key].values()):
                    return obj[key]
        # if already a dict[str, Tensor]
        if all(isinstance(k, str) for k in obj.keys()) and all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj  # type: ignore
    raise ValueError("Could not locate a tensor state dict (tried delta_state_dict/state_dict/model_state_dict).")


def layer_norm(t: torch.Tensor, metric: str = "l2") -> float:
    x = t.detach().float().cpu()
    if metric == "l2":
        return torch.norm(x).item()
    if metric == "l1":
        return torch.norm(x, p=1).item()
    if metric == "mean_abs":
        return x.abs().mean().item()
    if metric == "rms":
        return torch.sqrt((x * x).mean()).item()
    raise ValueError(f"Unknown metric: {metric}")


def compute_layer_metrics(state: Dict[str, torch.Tensor], metric: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for name, t in state.items():
        # you can filter here if you only want weights:
        # if not name.endswith(".weight"): continue
        out[name] = layer_norm(t, metric=metric)
    return out


def to_fixed_layer_vector(metrics: Dict[str, float], layers: List[str]) -> np.ndarray:
    vec = np.zeros(len(layers), dtype=np.float64)
    for i, l in enumerate(layers):
        vec[i] = float(metrics.get(l, 0.0))
    return vec


def safe_row_normalize(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denom = mat.sum(axis=1, keepdims=True)
    return mat / (denom + eps)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def plot_heatmap_clients_layers(
    mat: np.ndarray,
    clients: List[int],
    layers: List[str],
    title: str,
    out_path: Path,
    diverging: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(layers)), max(4, 0.35 * len(clients))))

    if diverging:
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

    idx_a = index_dir(dir_a)
    idx_b = index_dir(dir_b)

    keys = sorted(set(idx_a.keys()) & set(idx_b.keys()), key=lambda k: (k.round, k.client))
    if round_arg is not None:
        keys = [k for k in keys if k.round == round_arg]

    if not keys:
        raise SystemExit("No matching (client, round) pairs found between the two folders (or after filtering by --round).")

    # Collect records (long format) for CSV
    records = []  # dict rows

    # Also keep per-round matrices for plotting
    by_round: Dict[int, Dict[str, List[Tuple[int, np.ndarray]]]] = {}  # round -> {"A":[(client, vec)], "B":[...], "diff":[...]}

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

        # Relative contribution per update (helps compare “where” the update went, independent of total magnitude)
        v_a_rel = v_a / (v_a.sum() + 1e-12)
        v_b_rel = v_b / (v_b.sum() + 1e-12)
        v_diff_rel = v_a_rel - v_b_rel

        # Early vs late score (simple): early_fraction - late_fraction
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
                "A_abs": float(v_a[i]),
                "B_abs": float(v_b[i]),
                "A_rel": float(v_a_rel[i]),
                "B_rel": float(v_b_rel[i]),
                "diff_rel": float(v_diff_rel[i]),
                "file_a": str(p_a),
                "file_b": str(p_b),
                "metric": metric,
                "early_late_score_A": early_late_score_a,
                "early_late_score_B": early_late_score_b,
            })

        by_round.setdefault(k.round, {"A": [], "B": [], "diff": []})
        by_round[k.round]["A"].append((k.client, v_a_rel))
        by_round[k.round]["B"].append((k.client, v_b_rel))
        by_round[k.round]["diff"].append((k.client, v_diff_rel))

    # Write CSVs (no pandas dependency)
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
        for r in records:
            w.writerow(r)

    # wide metrics (one row per (round, client), columns like A_rel.conv1.weight, ...)
    wide_rows = {}
    for r in records:
        key = (r["round"], r["client"])
        wide_rows.setdefault(key, {"round": r["round"], "client": r["client"], "metric": r["metric"],
                                   "early_late_score_A": r["early_late_score_A"],
                                   "early_late_score_B": r["early_late_score_B"]})
        layer = r["layer"]
        wide_rows[key][f"A_abs.{layer}"] = r["A_abs"]
        wide_rows[key][f"B_abs.{layer}"] = r["B_abs"]
        wide_rows[key][f"A_rel.{layer}"] = r["A_rel"]
        wide_rows[key][f"B_rel.{layer}"] = r["B_rel"]
        wide_rows[key][f"diff_rel.{layer}"] = r["diff_rel"]

    wide_fieldnames = sorted({k for row in wide_rows.values() for k in row.keys()})
    with open(out_dir / "layer_metrics_wide.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=wide_fieldnames)
        w.writeheader()
        for key in sorted(wide_rows.keys()):
            w.writerow(wide_rows[key])

    # Plot per-round heatmaps
    for rnd, payload in sorted(by_round.items()):
        # sort clients
        payload["A"].sort(key=lambda x: x[0])
        payload["B"].sort(key=lambda x: x[0])
        payload["diff"].sort(key=lambda x: x[0])

        clients = [c for c, _ in payload["A"]]
        mat_a = np.stack([v for _, v in payload["A"]], axis=0)
        mat_b = np.stack([v for _, v in payload["B"]], axis=0)
        mat_d = np.stack([v for _, v in payload["diff"]], axis=0)

        plot_heatmap_clients_layers(
            mat_a, clients, layers,
            title=f"Round {rnd} — Method A (relative layer contribution)",
            out_path=out_dir / f"heatmap_round_{rnd}_A.png",
            diverging=False,
        )
        plot_heatmap_clients_layers(
            mat_b, clients, layers,
            title=f"Round {rnd} — Method B (relative layer contribution)",
            out_path=out_dir / f"heatmap_round_{rnd}_B.png",
            diverging=False,
        )
        plot_heatmap_clients_layers(
            mat_d, clients, layers,
            title=f"Round {rnd} — Diff (A − B) in relative contribution",
            out_path=out_dir / f"heatmap_round_{rnd}_diff.png",
            diverging=True,
        )

    print(f"Done. Wrote outputs to: {out_dir.resolve()}")
    print("Key files:")
    print(f"  - {out_dir / 'matched_pairs.csv'}")
    print(f"  - {out_dir / 'layer_metrics_long.csv'}")
    print(f"  - {out_dir / 'layer_metrics_wide.csv'}")
    print("  - heatmap_round_<r>_{A,B,diff}.png")


if __name__ == "__main__":
    main()
