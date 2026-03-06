"""Print per-defence classification metrics from a defence simulation CSV.

Usage:
    python defence/print_defence_metrics.py <csv_path> [--out <output.csv>] [--plot <out_dir>]

Prints total_accuracy, F1, precision, recall, FNR, FPR for each defence,
broken down per attack type and combined.  Optionally writes to a CSV and/or
generates metric comparison plots (PNG + PDF) in <out_dir>.

Defence selection:
  - All mlp_* except mlp_defence are kept.
  - fc2_threshold_* variants are all kept.
  - For each statistical family (s_mad, s_iqr, etc.) only the best variant
    by combined F1 is kept.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Statistical families: all variants within each family compete; only the
# best by combined F1 is kept.  Matched by prefix before the parameter suffix.
_STAT_FAMILY_PREFIXES = [
    "fc2_lowerhalf_mad",
    "s_iqr",
    "s_kmeans",
    "s_lowerhalf_mad",
    "s_mad",
]


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def aggregate_metrics(df: pd.DataFrame) -> dict:
    """Compute TP/FP/TN/FN by summing counts, then derive rates.

    Positive = ls_drifted (malicious).  Negative = benign + concept_drifted.
    """
    tp = df["ls_drifted_flagged"].sum()
    fn = df["ls_drifted_missed"].sum()
    fp = df["benign_flagged"].sum() + df["concept_drifted_flagged"].sum()
    tn = df["benign_correct"].sum() + df["concept_drifted_correct"].sum()

    total = tp + fn + fp + tn
    total_accuracy = (tp + tn) / total if total else 0.0

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fnr = 1.0 - recall
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    return {
        "total_accuracy": total_accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "FNR": fnr,
        "FPR": fpr,
    }


def select_defences(df: pd.DataFrame) -> list[str]:
    """Return the filtered list of defence names to use.

    Rules:
      - mlp_*            → keep all except mlp_defence
      - fc2_threshold_*  → keep all variants
      - stat families    → keep only the best variant by combined F1
    """
    all_defences = sorted(df["defence"].unique())
    selected = []

    for d in all_defences:
        if d.startswith("mlp_"):
            if d != "mlp_defence":
                selected.append(d)
            continue
        if d.startswith("fc2_threshold_"):
            selected.append(d)
            continue
        # Statistical family variants are handled below

    for prefix in _STAT_FAMILY_PREFIXES:
        variants = [d for d in all_defences if d.startswith(prefix + "_")]
        if not variants:
            continue
        best = max(variants, key=lambda v: aggregate_metrics(df[df["defence"] == v])["f1"])
        selected.append(best)
        others = [v for v in variants if v != best]
        if others:
            print(f"  [{prefix}] best: {best}  (dropped: {', '.join(others)})")

    return sorted(selected)


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def print_table(rows: list[dict], title: str) -> None:
    """Pretty-print a table of metrics to the console."""
    cols = ["defence", "attack_type", "total_accuracy", "f1", "precision", "recall", "FNR", "FPR"]
    widths = {
        "defence": 30,
        "attack_type": 16,
        "total_accuracy": 10,
        "f1": 8,
        "precision": 10,
        "recall": 8,
        "FNR": 8,
        "FPR": 8,
    }

    header = "  ".join(c.rjust(widths[c]) for c in cols)
    sep = "-" * len(header)

    print(f"\n{title}")
    print(sep)
    print(header)
    print(sep)

    for row in rows:
        parts = []
        for c in cols:
            v = row[c]
            w = widths[c]
            if isinstance(v, float):
                parts.append(f"{v:.4f}".replace(".", ",").rjust(w))
            else:
                parts.append(str(v).rjust(w))
        print("  ".join(parts))

    print(sep)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_METRIC_META = {
    "total_accuracy": {"label": "Total Accuracy",  "higher_better": True,  "color": "#4C72B0"},
    "f1":             {"label": "F1 Score",         "higher_better": True,  "color": "#55A868"},
    "precision":      {"label": "Precision",        "higher_better": True,  "color": "#4C72B0"},
    "recall":         {"label": "Recall",           "higher_better": True,  "color": "#55A868"},
    "FNR":            {"label": "FNR",              "higher_better": False, "color": "#DD8452"},
    "FPR":            {"label": "FPR",              "higher_better": False, "color": "#C44E52"},
}


def _save_plot(fig: plt.Figure, out_dir: str, name: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"{name}.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, f"{name}.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {name}.png / .pdf")


def plot_single_metric(rows: list[dict], metric: str, out_dir: str) -> None:
    """Vertical bar chart for one metric, all defences, combined only."""
    meta = _METRIC_META[metric]
    sorted_rows = sorted(rows, key=lambda r: r[metric], reverse=meta["higher_better"])

    defences = [r["defence"] for r in sorted_rows]
    values = [r[metric] * 100 for r in sorted_rows]
    x = np.arange(len(defences))

    fig, ax = plt.subplots(figsize=(max(10, len(defences) * 1.2), 6))
    bars = ax.bar(x, values, color=meta["color"], alpha=0.85, edgecolor="white")

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{val:.1f}", ha="center", va="bottom", fontsize=8,
        )

    ax.set_ylabel(f"{meta['label']} (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(defences, rotation=35, ha="right", fontsize=9)
    ax.set_xlim(x[0] - 0.6, x[-1] + 0.6)
    ax.set_ylim(0, 110)
    ax.set_title(f"{meta['label']} per Defence (combined)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save_plot(fig, out_dir, f"defence_{metric.lower()}")


def plot_fpr_fnr_grouped(rows: list[dict], out_dir: str) -> None:
    """Grouped bar chart: FPR and FNR side by side per defence, sorted by F1."""
    sorted_rows = sorted(rows, key=lambda r: r["f1"], reverse=True)

    defences = [r["defence"] for r in sorted_rows]
    fprs = [r["FPR"] * 100 for r in sorted_rows]
    fnrs = [r["FNR"] * 100 for r in sorted_rows]

    x = np.arange(len(defences))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(defences) * 1.2), 6))
    bars1 = ax.bar(x - width / 2, fprs, width, label="FPR (false alarms)", color="#C44E52", alpha=0.85)
    bars2 = ax.bar(x + width / 2, fnrs, width, label="FNR (missed attacks)", color="#DD8452", alpha=0.85)

    for bar, val in zip(bars1, fprs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}", ha="center", fontsize=7)
    for bar, val in zip(bars2, fnrs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}", ha="center", fontsize=7)

    ax.set_ylabel("Rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(defences, rotation=35, ha="right", fontsize=9)
    ax.set_xlim(x[0] - 0.6, x[-1] + 0.6)
    ax.legend(fontsize=10)
    ax.set_title("FPR vs FNR per Defence (combined, sorted by F1)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save_plot(fig, out_dir, "defence_fpr_fnr_grouped")


def plot_fpr_fnr_scatter(rows: list[dict], out_dir: str) -> None:
    """Scatter plot: FPR vs FNR, each defence as a labelled point."""
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(rows)))

    for i, row in enumerate(rows):
        fpr = row["FPR"] * 100
        fnr = row["FNR"] * 100
        ax.scatter(fpr, fnr, s=120, color=colors[i], zorder=3, edgecolors="black", linewidth=0.8)
        ax.annotate(row["defence"], (fpr, fnr),
                    textcoords="offset points", xytext=(6, 4), fontsize=7)

    ax.set_xlabel("FPR — False Positive Rate (%)")
    ax.set_ylabel("FNR — False Negative Rate (%)")
    ax.set_title("FPR vs FNR Tradeoff per Defence (combined)")
    ax.grid(alpha=0.3)
    ax.axhline(y=10, color="gray", linestyle="--", alpha=0.35)
    ax.axvline(x=10, color="gray", linestyle="--", alpha=0.35)
    ax.text(10.5, 10.5, "ideal region", fontsize=8, color="gray", alpha=0.6)
    fig.tight_layout()
    _save_plot(fig, out_dir, "defence_fpr_fnr_scatter")


def generate_plots(combined_rows: list[dict], out_dir: str) -> None:
    """Generate all metric plots from the combined rows."""
    print(f"\nGenerating plots -> {out_dir}")
    for metric in _METRIC_META:
        plot_single_metric(combined_rows, metric, out_dir)
    plot_fpr_fnr_grouped(combined_rows, out_dir)
    plot_fpr_fnr_scatter(combined_rows, out_dir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Defence classification metrics")
    parser.add_argument("csv_path", help="Path to defence_stats CSV")
    parser.add_argument("--out", help="Optional output CSV path (semicolon-separated)")
    parser.add_argument("--plot", metavar="OUT_DIR",
                        help="Generate metric plots (PNG + PDF) and save to OUT_DIR")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path, sep=";")

    # Strip leading apostrophes that Excel may add when saving CSV
    for col in df.columns:
        if df[col].dtype == object or str(df[col].dtype) == "string":
            df[col] = df[col].astype(str).str.lstrip("'")

    # Ensure count columns are numeric (needed after stripping)
    count_cols = [
        "benign_correct", "benign_flagged", "concept_drifted_correct",
        "concept_drifted_flagged", "ls_drifted_flagged", "ls_drifted_missed",
    ]
    for col in count_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    print("\nSelecting defences...")
    defences = select_defences(df)
    attack_types = sorted(df["attack_type"].unique())
    print(f"  Selected {len(defences)} defences: {defences}")

    # Per attack type
    per_attack_rows = []
    for defence in defences:
        for attack in attack_types:
            sub = df[(df["defence"] == defence) & (df["attack_type"] == attack)]
            if sub.empty:
                continue
            m = aggregate_metrics(sub)
            per_attack_rows.append({"defence": defence, "attack_type": attack, **m})

    print_table(per_attack_rows, "Per Attack Type")

    # Combined
    combined_rows = []
    for defence in defences:
        sub = df[df["defence"] == defence]
        m = aggregate_metrics(sub)
        combined_rows.append({"defence": defence, "attack_type": "combined", **m})

    print_table(combined_rows, "Combined (LS + Rotation)")

    # Write to CSV if requested
    if args.out:
        all_rows = per_attack_rows + combined_rows
        out_df = pd.DataFrame(all_rows)
        out_df.to_csv(
            args.out,
            index=False,
            sep=";",
            decimal=",",
            quoting=csv.QUOTE_ALL,
        )
        print(f"\nWritten {len(out_df)} rows to {args.out}")

    # Generate plots if requested
    if args.plot:
        generate_plots(combined_rows, args.plot)


if __name__ == "__main__":
    main()
