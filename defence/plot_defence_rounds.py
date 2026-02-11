"""Plot per-round accuracy curves from a defence simulation CSV.

Usage:
    python defence/plot_defence_rounds.py defence/defence_stats_2026-02-11_12-00-00.csv
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def compute_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """Add an 'accuracy' column: fraction of all clients correctly classified."""
    total_clients = df["n_benign"] + df["n_concept_drifted"] + df["n_ls_drifted"]
    total_correct = (
        df["benign_correct"] + df["concept_drifted_correct"] + df["ls_drifted_flagged"]
    )
    df = df.copy()
    df["accuracy"] = total_correct / total_clients
    return df


HIGHLIGHT_COLORS = {
    "fc2_threshold_median": "#e41a1c",
    "fc2_threshold_mean": "#ff7f00",
    "mlp_cf-0.5_R10-30": "#4daf4a",
    "mlp_cf-0.5_cf-0.375_R10-20": "#377eb8",
    "mlp_cf-0.5_cf-0.375_R10-30": "#984ea3",
}
GRAY = "#b0b0b0"


def _style_for(name: str) -> dict:
    """Return plot kwargs: highlighted methods get colour + thicker line,
    everything else is gray and thin."""
    if name in HIGHLIGHT_COLORS:
        return {"color": HIGHLIGHT_COLORS[name], "linewidth": 1.8, "alpha": 1.0}
    return {"color": GRAY, "linewidth": 0.8, "alpha": 0.5}


def plot_accuracy_over_rounds(df: pd.DataFrame, out_path: str) -> None:
    """Create a two-subplot figure (LS top, rotation bottom) with one accuracy
    curve per defence method."""
    fig, (ax_ls, ax_rot) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    attack_types = df["attack_type"].unique()
    ls_key = [a for a in attack_types if "label" in a.lower() or "ls" in a.lower()]
    rot_key = [a for a in attack_types if "rot" in a.lower()]

    panels = []
    if ls_key:
        panels.append((ax_ls, ls_key[0]))
    if rot_key:
        panels.append((ax_rot, rot_key[0]))

    for ax, attack in panels:
        sub = df[df["attack_type"] == attack]
        run_id = sub["run_id"].iloc[0] if "run_id" in sub.columns else attack

        # Draw gray (background) lines first, then highlighted on top
        defences = sorted(sub["defence"].unique(),
                          key=lambda d: d not in HIGHLIGHT_COLORS)
        for defence_name in defences:
            grp = sub[sub["defence"] == defence_name].sort_values("round")
            style = _style_for(defence_name)
            ax.plot(grp["round"], grp["accuracy"], label=defence_name, **style)

        ax.set_ylabel("Accuracy")
        ax.set_title(run_id)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)

    ax_rot.set_xlabel("Round")

    # Legend: only highlighted methods + one gray "other" entry
    handles, labels = ax_ls.get_legend_handles_labels()
    highlight_h, highlight_l = [], []
    has_gray = False
    for h, l in zip(handles, labels):
        if l in HIGHLIGHT_COLORS:
            highlight_h.append(h)
            highlight_l.append(l)
        else:
            has_gray = True
    if has_gray:
        gray_line = plt.Line2D([], [], color=GRAY, linewidth=0.8, alpha=0.5)
        highlight_h.append(gray_line)
        highlight_l.append("other defences")
    fig.legend(highlight_h, highlight_l, loc="lower center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.06, 1, 1])

    base, _ = os.path.splitext(out_path)
    fig.savefig(f"{base}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{base}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {base}.png / .pdf")


def main():
    if len(sys.argv) < 2:
        print("Usage: python defence/plot_defence_rounds.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    df = pd.read_csv(csv_path, sep=";")
    df = compute_accuracy(df)

    out_path = os.path.splitext(csv_path)[0] + "_accuracy_rounds.png"
    plot_accuracy_over_rounds(df, out_path)


if __name__ == "__main__":
    main()
