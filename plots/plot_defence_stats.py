"""
Plot defence detection algorithm statistics from the CSV spreadsheet.

Reads 'Federated Learning Defence Statistics - Sheet1.csv' and generates
comparison plots for all detection methods across simulation scenarios,
with extensive focus on FC2 Threshold and MLP methods with full coverage.

Author: Auto-generated
Date: 2026-02-10
"""

import sys
sys.path.append(".")

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SAVE_DIR = "plots/defence_stats/"
CSV_PATH = "Federated Learning Defence Statistics - Sheet1.csv"

# Short display names for methods
SHORT_NAMES = {
    "classify_by_share_fc2_threshold": "FC2 Threshold",
    'classify_by_s_mad(features, k=1.5)': "S-MAD k=1.5",
    'classify_by_s_mad(features, k=1)': "S-MAD k=1.0",
    'classify_by_s_mad(features, k=0.5)': "S-MAD k=0.5",
    "classify_by_s_kmeans(features)": "S-KMeans",
    "classify_by_s_iqr(features)": "S-IQR",
    "classify_by_s_lowerhalf_mad(features)": "S-LowerHalf MAD",
    "classify_by_share_fc2_lowerhalf_mad(features)": "FC2 LowerHalf MAD",
    "mlp trained on cf_0.5 and rounds 10 to 20": "MLP (cf0.5, r10-20)",
    "mlp trained on cf_0.5/0.375 and rounds 10 to 20": "MLP (cf0.5/0.375, r10-20)",
    "mlp trained on cf_0.5/0.375 and rounds 10 to 30": "MLP (cf0.5/0.375, r10-30)",
    "mlp trained on cf_0.5 and rounds 10 to 30 (1-2bi)": "MLP (cf0.5, r10-30, 1-2bi)",
}

# Methods that have data across ALL 12 scenarios (both sims x 3 CFs x 2 drifts)
FOCUS_METHODS = ["FC2 Threshold", "MLP (cf0.5, r10-20)", "MLP (cf0.5/0.375, r10-30)"]

STATISTICAL_METHODS = [
    "FC2 Threshold", "S-MAD k=1.5", "S-MAD k=1.0", "S-MAD k=0.5",
    "S-KMeans", "S-IQR", "S-LowerHalf MAD", "FC2 LowerHalf MAD",
]
MLP_METHODS = [
    "MLP (cf0.5, r10-20)", "MLP (cf0.5/0.375, r10-20)",
    "MLP (cf0.5/0.375, r10-30)", "MLP (cf0.5, r10-30, 1-2bi)",
]


def parse_csv():
    """Parse the multi-header CSV into a structured dict.
    Stops at the 80-round section to avoid overwriting 40-round data."""
    raw = pd.read_csv(CSV_PATH, header=None)

    # Find the boundary where 80-round section starts by scanning for it
    # in any column (it appears in column 1, not column 0)
    stop_row = raw.shape[0]
    for i in range(3, raw.shape[0]):
        for j in range(raw.shape[1]):
            cell = str(raw.iloc[i, j]).strip()
            if "80 ROUND" in cell.upper():
                stop_row = i
                break
        if stop_row != raw.shape[0]:
            break

    print(f"Parsing 40-round data from rows 3 to {stop_row - 1}")

    methods = []
    data_rows = []

    for i in range(3, stop_row):
        name = str(raw.iloc[i, 0]).strip()
        if name == "" or name == "nan" or name == " ":
            continue
        methods.append(name)
        vals = []
        for j in range(1, raw.shape[1]):
            v = raw.iloc[i, j]
            try:
                vals.append(float(v))
            except (ValueError, TypeError):
                vals.append(np.nan)
        data_rows.append(vals)

    # Build column labels from rows 0-2
    sim_types = []
    client_fracs = []
    metrics = []
    for j in range(1, raw.shape[1]):
        sim_types.append(str(raw.iloc[0, j]).strip())
        client_fracs.append(str(raw.iloc[1, j]).strip())
        metrics.append(str(raw.iloc[2, j]).strip())

    return methods, sim_types, client_fracs, metrics, np.array(data_rows, dtype=float)


def build_structured_data(methods, sim_types, client_fracs, metrics, data):
    """Build structured dict: method -> {(sim, cf, drift): {total, drifted, benign}, "summary": {...}}"""
    n_cols = data.shape[1]

    # Find summary columns start
    summary_start = None
    for j in range(n_cols):
        if "Total Accuracy" in metrics[j]:
            summary_start = j
            break

    # Propagate simulation type headers forward
    current_sim = ""
    for j in range(len(sim_types)):
        if sim_types[j] != "" and sim_types[j] != "nan":
            current_sim = sim_types[j]
        sim_types[j] = current_sim

    # Propagate client fractions forward
    current_cf = ""
    for j in range(len(client_fracs)):
        if client_fracs[j] != "" and client_fracs[j] != "nan":
            current_cf = client_fracs[j]
        client_fracs[j] = current_cf

    # Shorten simulation labels
    sim_labels = {}
    for j in range(n_cols):
        s = sim_types[j]
        if "Incremental" in s or "65" in s:
            sim_labels[j] = "Incr. 65\u00b0"
        elif "Gradual" in s or "45" in s:
            sim_labels[j] = "Grad. 45\u00b0"
        else:
            sim_labels[j] = ""

    result = {}
    for i, method_name in enumerate(methods):
        short = SHORT_NAMES.get(method_name, method_name)
        entry = {}

        end = summary_start if summary_start else n_cols
        for j in range(end):
            m = metrics[j]
            if m == "nan" or m == "":
                continue
            sim = sim_labels[j]
            cf = client_fracs[j]

            if "LS" in m:
                drift = "LS"
            elif "Rot" in m:
                drift = "Rot"
            else:
                continue

            key = (sim, cf, drift)
            if key not in entry:
                entry[key] = {"total": np.nan, "drifted": np.nan, "benign": np.nan}

            if "Total" in m:
                entry[key]["total"] = data[i, j]
            elif "Malicious" in m or "Drifted" in m:
                entry[key]["drifted"] = data[i, j]
            elif "Benign" in m:
                entry[key]["benign"] = data[i, j]

        if summary_start is not None:
            entry["summary"] = {
                "total": data[i, summary_start],
                "drifted_mal": data[i, summary_start + 1],
                "drifted_concept": data[i, summary_start + 2],
                "benign": data[i, summary_start + 3],
            }

        result[short] = entry

    # Debug: print scenario counts per method
    for m, entry in result.items():
        scenario_keys = [k for k in entry if isinstance(k, tuple)]
        filled = sum(1 for k in scenario_keys if not np.isnan(entry[k]["total"]))
        print(f"  {m}: {filled}/{len(scenario_keys)} scenarios with total acc data")

    return result


# ---------------------------------------------------------------------------
# GENERAL OVERVIEW PLOTS
# ---------------------------------------------------------------------------

def plot_overall_summary(structured, save_dir):
    """Bar chart: overall Total Acc, Malicious/Drifted Acc, Benign Acc for each method."""
    methods_with_summary = [m for m in structured if "summary" in structured[m]
                            and not np.isnan(structured[m]["summary"]["total"])]
    methods_with_summary.sort(key=lambda m: structured[m]["summary"]["total"], reverse=True)

    totals = [structured[m]["summary"]["total"] for m in methods_with_summary]
    drifted_mal = [structured[m]["summary"]["drifted_mal"] for m in methods_with_summary]
    drifted_concept = [structured[m]["summary"]["drifted_concept"] for m in methods_with_summary]
    benign = [structured[m]["summary"]["benign"] for m in methods_with_summary]

    x = np.arange(len(methods_with_summary))
    width = 0.2

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - 1.5 * width, totals, width, label="Total Accuracy", color="#4C72B0")
    ax.bar(x - 0.5 * width, drifted_mal, width, label="Malicious/Drifted Acc", color="#DD8452")
    ax.bar(x + 0.5 * width, drifted_concept, width, label="Concept Drifted Acc", color="#55A868")
    ax.bar(x + 1.5 * width, benign, width, label="Benign Client Acc", color="#C44E52")

    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(methods_with_summary, rotation=35, ha="right", fontsize=9)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, save_dir, "overall_summary")


def plot_total_accuracy_ranking(structured, save_dir):
    """Horizontal bar chart ranking methods by overall total accuracy."""
    methods_with_summary = [m for m in structured if "summary" in structured[m]
                            and not np.isnan(structured[m]["summary"]["total"])]
    methods_with_summary.sort(key=lambda m: structured[m]["summary"]["total"])

    totals = [structured[m]["summary"]["total"] for m in methods_with_summary]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(methods_with_summary)))
    bars = ax.barh(range(len(methods_with_summary)), totals, color=colors)
    ax.set_yticks(range(len(methods_with_summary)))
    ax.set_yticklabels(methods_with_summary, fontsize=9)
    ax.set_xlabel("Overall Total Accuracy (%)")
    ax.set_xlim(0, 105)
    ax.grid(axis="x", alpha=0.3)

    for bar, val in zip(bars, totals):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=8)

    fig.tight_layout()
    _save(fig, save_dir, "total_accuracy_ranking")


def plot_heatmap(structured, save_dir):
    """Heatmap: methods x scenarios showing total accuracy."""
    all_keys = set()
    for m in structured:
        for k in structured[m]:
            if k != "summary" and isinstance(k, tuple):
                all_keys.add(k)

    sorted_keys = sorted(all_keys, key=lambda k: (k[0], float(k[1]), k[2]))
    all_methods = sorted(structured.keys(), key=lambda m: _sort_key(structured, m), reverse=True)

    matrix = np.full((len(all_methods), len(sorted_keys)), np.nan)
    for i, m in enumerate(all_methods):
        for j, k in enumerate(sorted_keys):
            if k in structured[m]:
                matrix[i, j] = structured[m][k]["total"]

    col_labels = [f"{k[0]}\nCF={k[1]}, {k[2]}" for k in sorted_keys]

    fig, ax = plt.subplots(figsize=(16, 8))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=30, vmax=100)

    ax.set_xticks(range(len(sorted_keys)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(all_methods)))
    ax.set_yticklabels(all_methods, fontsize=8)

    for i in range(len(all_methods)):
        for j in range(len(sorted_keys)):
            val = matrix[i, j]
            if not np.isnan(val):
                color = "white" if val < 60 else "black"
                ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=6, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Total Accuracy (%)")
    fig.tight_layout()
    _save(fig, save_dir, "heatmap_scenarios")


def _sort_key(structured, m):
    """Sort key: summary total accuracy (descending), fallback to avg scenario total."""
    if "summary" in structured[m] and not np.isnan(structured[m]["summary"]["total"]):
        return structured[m]["summary"]["total"]
    vals = [v["total"] for k, v in structured[m].items()
            if isinstance(k, tuple) and not np.isnan(v["total"])]
    return np.nanmean(vals) if vals else 0


def plot_ls_vs_rot(structured, save_dir):
    """Grouped bar chart: average LS total acc vs average Rot total acc per method."""
    methods = list(structured.keys())
    method_data = {}

    for m in methods:
        ls_vals, rot_vals = [], []
        for k, v in structured[m].items():
            if k == "summary" or not isinstance(k, tuple):
                continue
            if k[2] == "LS" and not np.isnan(v["total"]):
                ls_vals.append(v["total"])
            elif k[2] == "Rot" and not np.isnan(v["total"]):
                rot_vals.append(v["total"])
        if ls_vals or rot_vals:
            method_data[m] = (
                np.nanmean(ls_vals) if ls_vals else np.nan,
                np.nanmean(rot_vals) if rot_vals else np.nan,
            )

    valid_methods = sorted(method_data.keys(), key=lambda m: _sort_key(structured, m), reverse=True)
    ls_avgs = [method_data[m][0] for m in valid_methods]
    rot_avgs = [method_data[m][1] for m in valid_methods]

    x = np.arange(len(valid_methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width / 2, ls_avgs, width, label="Label Swap (LS)", color="#4C72B0")
    ax.bar(x + width / 2, rot_avgs, width, label="Rotation (Rot)", color="#DD8452")
    ax.set_ylabel("Avg Total Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(valid_methods, rotation=35, ha="right", fontsize=9)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, save_dir, "ls_vs_rot")


def plot_client_fraction_impact(structured, save_dir):
    """For each client fraction, show average total accuracy per method."""
    methods = list(structured.keys())
    cfs = ["0.5", "0.375", "0.2"]
    method_data = {}

    for m in methods:
        has_data = False
        row = {}
        for cf in cfs:
            vals = []
            for k, v in structured[m].items():
                if k == "summary" or not isinstance(k, tuple):
                    continue
                if k[1] == cf and not np.isnan(v["total"]):
                    vals.append(v["total"])
            row[cf] = np.nanmean(vals) if vals else np.nan
            if vals:
                has_data = True
        if has_data:
            method_data[m] = row

    valid_methods = sorted(method_data.keys(), key=lambda m: _sort_key(structured, m), reverse=True)
    cf_data = {cf: [method_data[m][cf] for m in valid_methods] for cf in cfs}

    x = np.arange(len(valid_methods))
    width = 0.25
    colors = ["#4C72B0", "#55A868", "#DD8452"]

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, cf in enumerate(cfs):
        ax.bar(x + (i - 1) * width, cf_data[cf], width, label=f"CF = {cf}", color=colors[i])

    ax.set_ylabel("Avg Total Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(valid_methods, rotation=35, ha="right", fontsize=9)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, save_dir, "client_fraction_impact")


def plot_incremental_vs_gradual(structured, save_dir):
    """Compare average total accuracy under incremental vs gradual drift per method."""
    methods = list(structured.keys())
    method_data = {}

    for m in methods:
        incr_vals, grad_vals = [], []
        for k, v in structured[m].items():
            if k == "summary" or not isinstance(k, tuple):
                continue
            if np.isnan(v["total"]):
                continue
            if "Incr" in k[0]:
                incr_vals.append(v["total"])
            elif "Grad" in k[0]:
                grad_vals.append(v["total"])
        if incr_vals or grad_vals:
            method_data[m] = (
                np.nanmean(incr_vals) if incr_vals else np.nan,
                np.nanmean(grad_vals) if grad_vals else np.nan,
            )

    valid_methods = sorted(method_data.keys(), key=lambda m: _sort_key(structured, m), reverse=True)
    incr_avgs = [method_data[m][0] for m in valid_methods]
    grad_avgs = [method_data[m][1] for m in valid_methods]

    x = np.arange(len(valid_methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width / 2, incr_avgs, width, label="Incremental 65\u00b0", color="#4C72B0")
    ax.bar(x + width / 2, grad_avgs, width, label="Gradual 45\u00b0", color="#55A868")
    ax.set_ylabel("Avg Total Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(valid_methods, rotation=35, ha="right", fontsize=9)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, save_dir, "incremental_vs_gradual")


def plot_drifted_vs_benign_accuracy(structured, save_dir):
    """Scatter plot: avg drifted detection acc vs avg benign preservation acc."""
    methods = list(structured.keys())
    method_data = {}

    for m in methods:
        d_vals, b_vals = [], []
        for k, v in structured[m].items():
            if k == "summary" or not isinstance(k, tuple):
                continue
            if not np.isnan(v["drifted"]):
                d_vals.append(v["drifted"])
            if not np.isnan(v["benign"]):
                b_vals.append(v["benign"])
        if d_vals and b_vals:
            method_data[m] = (np.nanmean(d_vals), np.nanmean(b_vals))

    valid_methods = sorted(method_data.keys(), key=lambda m: _sort_key(structured, m), reverse=True)
    drifted_accs = [method_data[m][0] for m in valid_methods]
    benign_accs = [method_data[m][1] for m in valid_methods]

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(valid_methods)))

    for i, (m, d, b) in enumerate(zip(valid_methods, drifted_accs, benign_accs)):
        marker = "s" if m in FOCUS_METHODS else "o"
        size = 200 if m in FOCUS_METHODS else 100
        ax.scatter(d, b, s=size, color=colors[i], zorder=3, edgecolors="black",
                   linewidth=1.5 if m in FOCUS_METHODS else 0.5, marker=marker)
        ax.annotate(m, (d, b), textcoords="offset points", xytext=(8, 5), fontsize=7,
                    fontweight="bold" if m in FOCUS_METHODS else "normal")

    ax.set_xlabel("Avg Drifted/Malicious Detection Accuracy (%)")
    ax.set_ylabel("Avg Benign Client Accuracy (%)")
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)
    ax.axhline(y=90, color="gray", linestyle="--", alpha=0.4)
    ax.axvline(x=90, color="gray", linestyle="--", alpha=0.4)
    ax.text(92, 92, "Ideal", fontsize=9, color="gray", alpha=0.6)
    fig.tight_layout()
    _save(fig, save_dir, "drifted_vs_benign_tradeoff")


def plot_statistical_vs_mlp(structured, save_dir):
    """Compare statistical methods vs MLP methods on key metrics."""
    stat_totals, stat_drifted, stat_benign = [], [], []
    mlp_totals, mlp_drifted, mlp_benign = [], [], []

    for m, entry in structured.items():
        if "summary" not in entry or np.isnan(entry["summary"]["total"]):
            continue
        s = entry["summary"]
        if m in STATISTICAL_METHODS:
            stat_totals.append(s["total"])
            stat_drifted.append(s["drifted_mal"])
            stat_benign.append(s["benign"])
        elif m in MLP_METHODS:
            mlp_totals.append(s["total"])
            mlp_drifted.append(s["drifted_mal"])
            mlp_benign.append(s["benign"])

    categories = ["Total Accuracy", "Malicious/Drifted Acc", "Benign Client Acc"]
    stat_means = [np.nanmean(stat_totals), np.nanmean(stat_drifted), np.nanmean(stat_benign)]
    mlp_means = [np.nanmean(mlp_totals), np.nanmean(mlp_drifted), np.nanmean(mlp_benign)]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, stat_means, width, label="Statistical Methods (avg)", color="#4C72B0")
    bars2 = ax.bar(x + width / 2, mlp_means, width, label="MLP Methods (avg)", color="#DD8452")

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{bar.get_height():.1f}", ha="center", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{bar.get_height():.1f}", ha="center", fontsize=9)

    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, save_dir, "statistical_vs_mlp")


# ---------------------------------------------------------------------------
# FOCUS PLOTS: FC2 Threshold and complete-data MLPs
# ---------------------------------------------------------------------------

def plot_focus_full_breakdown(structured, save_dir):
    """Detailed per-scenario breakdown for FC2 Threshold and complete MLPs.
    Shows total accuracy for every (sim, CF, drift) combination side by side."""
    focus = [m for m in FOCUS_METHODS if m in structured]

    # Collect all scenario keys from focus methods
    all_keys = set()
    for m in focus:
        for k in structured[m]:
            if isinstance(k, tuple) and not np.isnan(structured[m][k]["total"]):
                all_keys.add(k)
    sorted_keys = sorted(all_keys, key=lambda k: (k[0], float(k[1]), k[2]))

    x = np.arange(len(sorted_keys))
    width = 0.25
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    fig, ax = plt.subplots(figsize=(18, 7))
    for i, m in enumerate(focus):
        vals = []
        for k in sorted_keys:
            if k in structured[m] and not np.isnan(structured[m][k]["total"]):
                vals.append(structured[m][k]["total"])
            else:
                vals.append(0)
        offset = (i - (len(focus) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=m, color=colors[i])
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{val:.0f}", ha="center", fontsize=6, rotation=90)

    col_labels = [f"{k[0]}\nCF={k[1]}, {k[2]}" for k in sorted_keys]
    ax.set_ylabel("Total Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=8)
    ax.legend(fontsize=10, loc="lower left")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, save_dir, "focus_full_scenario_breakdown")


def plot_focus_drifted_benign_breakdown(structured, save_dir):
    """For each focus method: side-by-side drifted acc vs benign acc across all scenarios."""
    focus = [m for m in FOCUS_METHODS if m in structured]

    all_keys = set()
    for m in focus:
        for k in structured[m]:
            if isinstance(k, tuple) and not np.isnan(structured[m][k]["total"]):
                all_keys.add(k)
    sorted_keys = sorted(all_keys, key=lambda k: (k[0], float(k[1]), k[2]))

    fig, axes = plt.subplots(len(focus), 1, figsize=(16, 5 * len(focus)), sharex=True)
    if len(focus) == 1:
        axes = [axes]

    for ax, m in zip(axes, focus):
        drifted_vals = []
        benign_vals = []
        for k in sorted_keys:
            if k in structured[m]:
                drifted_vals.append(structured[m][k]["drifted"])
                benign_vals.append(structured[m][k]["benign"])
            else:
                drifted_vals.append(np.nan)
                benign_vals.append(np.nan)

        x = np.arange(len(sorted_keys))
        width = 0.35
        ax.bar(x - width / 2, drifted_vals, width, label="Drifted/Malicious Acc", color="#DD8452")
        ax.bar(x + width / 2, benign_vals, width, label="Benign Acc", color="#55A868")

        # Annotate values
        for xi, (d, b) in enumerate(zip(drifted_vals, benign_vals)):
            if not np.isnan(d):
                ax.text(xi - width / 2, d + 1, f"{d:.0f}", ha="center", fontsize=7)
            if not np.isnan(b):
                ax.text(xi + width / 2, b + 1, f"{b:.0f}", ha="center", fontsize=7)

        ax.set_ylabel("Accuracy (%)")
        ax.set_title(m, fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.set_ylim(0, 110)
        ax.grid(axis="y", alpha=0.3)

    col_labels = [f"{k[0]}\nCF={k[1]}, {k[2]}" for k in sorted_keys]
    axes[-1].set_xticks(np.arange(len(sorted_keys)))
    axes[-1].set_xticklabels(col_labels, rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    _save(fig, save_dir, "focus_drifted_vs_benign_breakdown")


def plot_focus_radar(structured, save_dir):
    """Radar chart comparing FC2 Threshold and MLPs across aggregated metrics."""
    focus = [m for m in FOCUS_METHODS if m in structured]

    # Compute metrics for each focus method
    labels = [
        "Avg Total\n(all scenarios)",
        "Avg Drifted\nDetection",
        "Avg Benign\nPreservation",
        "LS Total\nAvg",
        "Rot Total\nAvg",
        "Worst-Case\nScenario",
    ]

    data = {}
    for m in focus:
        all_totals, all_drifted, all_benign = [], [], []
        ls_totals, rot_totals = [], []
        for k, v in structured[m].items():
            if k == "summary" or not isinstance(k, tuple):
                continue
            if not np.isnan(v["total"]):
                all_totals.append(v["total"])
            if not np.isnan(v["drifted"]):
                all_drifted.append(v["drifted"])
            if not np.isnan(v["benign"]):
                all_benign.append(v["benign"])
            if k[2] == "LS" and not np.isnan(v["total"]):
                ls_totals.append(v["total"])
            elif k[2] == "Rot" and not np.isnan(v["total"]):
                rot_totals.append(v["total"])

        data[m] = [
            np.nanmean(all_totals) if all_totals else 0,
            np.nanmean(all_drifted) if all_drifted else 0,
            np.nanmean(all_benign) if all_benign else 0,
            np.nanmean(ls_totals) if ls_totals else 0,
            np.nanmean(rot_totals) if rot_totals else 0,
            np.nanmin(all_totals) if all_totals else 0,
        ]

    n_metrics = len(labels)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    for i, m in enumerate(focus):
        values = data[m] + data[m][:1]
        ax.plot(angles, values, "o-", linewidth=2, label=m, color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 105)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=7)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, save_dir, "focus_radar_comparison")


def plot_focus_cf_detailed(structured, save_dir):
    """Detailed client fraction impact for focus methods, split by sim type and drift type."""
    focus = [m for m in FOCUS_METHODS if m in structured]
    cfs = ["0.5", "0.375", "0.2"]
    sims = ["Incr. 65\u00b0", "Grad. 45\u00b0"]
    drifts = ["LS", "Rot"]
    colors_method = ["#4C72B0", "#DD8452", "#55A868"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=True)

    for row, sim in enumerate(sims):
        for col, drift in enumerate(drifts):
            ax = axes[row][col]
            x = np.arange(len(cfs))
            width = 0.25

            for i, m in enumerate(focus):
                vals = []
                for cf in cfs:
                    key = (sim, cf, drift)
                    if key in structured[m] and not np.isnan(structured[m][key]["total"]):
                        vals.append(structured[m][key]["total"])
                    else:
                        vals.append(0)
                offset = (i - (len(focus) - 1) / 2) * width
                bars = ax.bar(x + offset, vals, width, label=m, color=colors_method[i])
                for bar, val in zip(bars, vals):
                    if val > 0:
                        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                                f"{val:.1f}", ha="center", fontsize=8)

            ax.set_title(f"{sim} / {drift}", fontsize=11, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels([f"CF={cf}" for cf in cfs], fontsize=10)
            ax.set_ylim(0, 110)
            ax.grid(axis="y", alpha=0.3)
            if col == 0:
                ax.set_ylabel("Total Accuracy (%)")

    axes[0][0].legend(fontsize=9, loc="lower left")
    fig.tight_layout()
    _save(fig, save_dir, "focus_cf_by_sim_and_drift")


def plot_focus_strengths_weaknesses(structured, save_dir):
    """Bar chart showing best and worst scenarios for each focus method."""
    focus = [m for m in FOCUS_METHODS if m in structured]
    colors_method = ["#4C72B0", "#DD8452", "#55A868"]

    fig, axes = plt.subplots(1, len(focus), figsize=(6 * len(focus), 7), sharey=True)
    if len(focus) == 1:
        axes = [axes]

    for ax, m, color in zip(axes, focus, colors_method):
        scenarios = []
        for k, v in structured[m].items():
            if k == "summary" or not isinstance(k, tuple):
                continue
            if not np.isnan(v["total"]):
                label = f"{k[0]}\nCF={k[1]}, {k[2]}"
                scenarios.append((label, v["total"]))

        scenarios.sort(key=lambda x: x[1])
        labels = [s[0] for s in scenarios]
        vals = [s[1] for s in scenarios]

        # Color bars: red-ish for lowest, green-ish for highest
        bar_colors = plt.cm.RdYlGn(np.linspace(0.15, 0.85, len(vals)))
        bars = ax.barh(range(len(vals)), vals, color=bar_colors)

        for bar, val in zip(bars, vals):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", fontsize=8)

        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("Total Accuracy (%)")
        ax.set_title(m, fontsize=11, fontweight="bold")
        ax.set_xlim(0, 110)
        ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    _save(fig, save_dir, "focus_strengths_weaknesses")


def plot_focus_heatmap(structured, save_dir):
    """Dedicated heatmap for focus methods only, showing all 3 metric types."""
    focus = [m for m in FOCUS_METHODS if m in structured]

    all_keys = set()
    for m in focus:
        for k in structured[m]:
            if isinstance(k, tuple) and not np.isnan(structured[m][k]["total"]):
                all_keys.add(k)
    sorted_keys = sorted(all_keys, key=lambda k: (k[0], float(k[1]), k[2]))

    metric_types = ["total", "drifted", "benign"]
    metric_labels = ["Total Acc", "Drifted/Mal Acc", "Benign Acc"]
    n_methods = len(focus)
    n_scenarios = len(sorted_keys)

    # Build a matrix: rows = method×metric, cols = scenarios
    n_rows = n_methods * len(metric_types)
    matrix = np.full((n_rows, n_scenarios), np.nan)
    row_labels = []

    for mi, m in enumerate(focus):
        for ti, mt in enumerate(metric_types):
            row_idx = mi * len(metric_types) + ti
            row_labels.append(f"{m}\n({metric_labels[ti]})")
            for j, k in enumerate(sorted_keys):
                if k in structured[m]:
                    matrix[row_idx, j] = structured[m][k][mt]

    col_labels = [f"{k[0]}\nCF={k[1]}, {k[2]}" for k in sorted_keys]

    fig, ax = plt.subplots(figsize=(16, 2.5 + n_rows * 0.7))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=20, vmax=100)

    ax.set_xticks(range(n_scenarios))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=7)

    # Draw horizontal separators between methods
    for mi in range(1, n_methods):
        ax.axhline(y=mi * len(metric_types) - 0.5, color="white", linewidth=2)

    for i in range(n_rows):
        for j in range(n_scenarios):
            val = matrix[i, j]
            if not np.isnan(val):
                color = "white" if val < 55 else "black"
                ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=7, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label("Accuracy (%)")
    fig.tight_layout()
    _save(fig, save_dir, "focus_detailed_heatmap")


def plot_focus_summary_comparison(structured, save_dir):
    """Summary metrics comparison for focus methods from the CSV summary columns."""
    focus = [m for m in FOCUS_METHODS if m in structured and "summary" in structured[m]]

    categories = ["Total\nAccuracy", "Malicious/\nDrifted Acc", "Concept\nDrifted Acc", "Benign\nClient Acc"]
    colors_method = ["#4C72B0", "#DD8452", "#55A868"]

    x = np.arange(len(categories))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, m in enumerate(focus):
        s = structured[m]["summary"]
        vals = [s["total"], s["drifted_mal"], s["drifted_concept"], s["benign"]]
        offset = (i - (len(focus) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=m, color=colors_method[i])
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                        f"{val:.1f}", ha="center", fontsize=8)

    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, save_dir, "focus_summary_comparison")


# ---------------------------------------------------------------------------
# CLASSIFICATION METRICS: FPR, FNR, Precision, Recall, F1
# ---------------------------------------------------------------------------

def _compute_group_metrics(ls_mal_acc, ls_ben_acc, rot_dri_acc, rot_ben_acc, cf):
    """Compute classification metrics for a (sim, CF) group.

    Only LS drift is malicious (positive). Rot drift is NOT malicious — flagging
    a Rot-drifted client is a false positive.

    Combines LS and Rot runs (each with N clients) into a joint evaluation:
      - Positives:  CF*N  LS-drifted clients  (from LS run)
      - Negatives:  (1-CF)*N benign (LS run) + CF*N rot-drifted + (1-CF)*N benign (Rot run)
                  = (2-CF)*N total negatives

    All accuracies on 0-100 scale, cf on 0-1 scale.
    """
    ls_mal = ls_mal_acc / 100.0   # TPR: P(flag | LS malicious)
    ls_ben = ls_ben_acc / 100.0   # P(pass | benign in LS run)
    rot_dri = rot_dri_acc / 100.0 # P(flag | Rot drifted) — this is a FP!
    rot_ben = rot_ben_acc / 100.0 # P(pass | benign in Rot run)

    # Counts proportional (per N)
    tp = ls_mal * cf
    fn = (1 - ls_mal) * cf
    fp = (1 - ls_ben) * (1 - cf) + rot_dri * cf + (1 - rot_ben) * (1 - cf)
    tn = ls_ben * (1 - cf) + (1 - rot_dri) * cf + rot_ben * (1 - cf)

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0       # = ls_mal
    fnr = 1.0 - recall
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"fpr": fpr, "fnr": fnr, "precision": precision, "recall": recall, "f1": f1}


def _get_method_avg_metrics(structured, method):
    """Compute averaged classification metrics across all (sim, CF) groups for a method.

    For each (sim, CF) group, takes the LS and Rot data and computes joint metrics
    where only LS is malicious (positive) and Rot is non-malicious (negative).
    """
    entry = structured[method]
    # Group scenarios by (sim, cf)
    groups = {}  # (sim, cf) -> {"LS": {...}, "Rot": {...}}
    for k, v in entry.items():
        if not isinstance(k, tuple):
            continue
        sim, cf_str, drift = k
        group_key = (sim, cf_str)
        if group_key not in groups:
            groups[group_key] = {}
        groups[group_key][drift] = v

    all_metrics = []
    for (sim, cf_str), g in groups.items():
        # Need both LS and Rot data for this group
        if "LS" not in g or "Rot" not in g:
            continue
        ls = g["LS"]
        rot = g["Rot"]
        if np.isnan(ls["drifted"]) or np.isnan(ls["benign"]) or \
           np.isnan(rot["drifted"]) or np.isnan(rot["benign"]):
            continue
        cf = float(cf_str)
        all_metrics.append(_compute_group_metrics(
            ls["drifted"], ls["benign"], rot["drifted"], rot["benign"], cf))

    if not all_metrics:
        return None
    # Average across all (sim, cf) groups
    return {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}


def _get_method_per_group_metrics(structured, method):
    """Compute classification metrics per (sim, CF) group for a method."""
    entry = structured[method]
    groups = {}
    for k, v in entry.items():
        if not isinstance(k, tuple):
            continue
        sim, cf_str, drift = k
        group_key = (sim, cf_str)
        if group_key not in groups:
            groups[group_key] = {}
        groups[group_key][drift] = v

    results = {}
    for (sim, cf_str), g in groups.items():
        if "LS" not in g or "Rot" not in g:
            continue
        ls = g["LS"]
        rot = g["Rot"]
        if np.isnan(ls["drifted"]) or np.isnan(ls["benign"]) or \
           np.isnan(rot["drifted"]) or np.isnan(rot["benign"]):
            continue
        cf = float(cf_str)
        results[(sim, cf_str)] = _compute_group_metrics(
            ls["drifted"], ls["benign"], rot["drifted"], rot["benign"], cf)
    return results


def plot_fpr_fnr(structured, save_dir):
    """Bar chart: average FPR and FNR per method, sorted by total accuracy."""
    method_metrics = {}
    for m in structured:
        metrics = _get_method_avg_metrics(structured, m)
        if metrics:
            method_metrics[m] = metrics

    valid_methods = sorted(method_metrics.keys(), key=lambda m: _sort_key(structured, m), reverse=True)
    fprs = [method_metrics[m]["fpr"] * 100 for m in valid_methods]
    fnrs = [method_metrics[m]["fnr"] * 100 for m in valid_methods]

    x = np.arange(len(valid_methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width / 2, fprs, width, label="FPR (false alarm on benign)", color="#C44E52")
    bars2 = ax.bar(x + width / 2, fnrs, width, label="FNR (missed drifted)", color="#DD8452")

    for bar, val in zip(bars1, fprs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", fontsize=7)
    for bar, val in zip(bars2, fnrs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", fontsize=7)

    ax.set_ylabel("Rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(valid_methods, rotation=35, ha="right", fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, save_dir, "fpr_fnr")


def plot_precision_recall_f1(structured, save_dir):
    """Bar chart: Precision, Recall, F1 per method, sorted by total accuracy."""
    method_metrics = {}
    for m in structured:
        metrics = _get_method_avg_metrics(structured, m)
        if metrics:
            method_metrics[m] = metrics

    valid_methods = sorted(method_metrics.keys(), key=lambda m: _sort_key(structured, m), reverse=True)
    precisions = [method_metrics[m]["precision"] * 100 for m in valid_methods]
    recalls = [method_metrics[m]["recall"] * 100 for m in valid_methods]
    f1s = [method_metrics[m]["f1"] * 100 for m in valid_methods]

    x = np.arange(len(valid_methods))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width, precisions, width, label="Precision", color="#4C72B0")
    ax.bar(x, recalls, width, label="Recall", color="#55A868")
    ax.bar(x + width, f1s, width, label="F1 Score", color="#DD8452")

    ax.set_ylabel("Score (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(valid_methods, rotation=35, ha="right", fontsize=9)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, save_dir, "precision_recall_f1")


def plot_f1_ranking(structured, save_dir):
    """Horizontal bar chart ranking methods by F1 score."""
    method_metrics = {}
    for m in structured:
        metrics = _get_method_avg_metrics(structured, m)
        if metrics:
            method_metrics[m] = metrics

    valid_methods = sorted(method_metrics.keys(), key=lambda m: method_metrics[m]["f1"])
    f1s = [method_metrics[m]["f1"] * 100 for m in valid_methods]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(valid_methods)))
    bars = ax.barh(range(len(valid_methods)), f1s, color=colors)
    ax.set_yticks(range(len(valid_methods)))
    ax.set_yticklabels(valid_methods, fontsize=9)
    ax.set_xlabel("F1 Score (%)")
    ax.set_xlim(0, 105)
    ax.grid(axis="x", alpha=0.3)

    for bar, val in zip(bars, f1s):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=8)

    fig.tight_layout()
    _save(fig, save_dir, "f1_ranking")


def plot_focus_fpr_fnr_per_group(structured, save_dir):
    """Detailed FPR/FNR per (sim, CF) group for focus methods.
    Each group combines LS (positives) and Rot (negatives) data."""
    focus = [m for m in FOCUS_METHODS if m in structured]

    all_keys = set()
    for m in focus:
        per_group = _get_method_per_group_metrics(structured, m)
        all_keys.update(per_group.keys())
    sorted_keys = sorted(all_keys, key=lambda k: (k[0], float(k[1])))

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    colors_method = ["#4C72B0", "#DD8452", "#55A868"]
    x = np.arange(len(sorted_keys))
    width = 0.25

    for rate_idx, (rate_name, rate_label) in enumerate([("fpr", "FPR (%)"), ("fnr", "FNR (%)")]):
        ax = axes[rate_idx]
        for i, m in enumerate(focus):
            per_group = _get_method_per_group_metrics(structured, m)
            vals = []
            for k in sorted_keys:
                if k in per_group:
                    vals.append(per_group[k][rate_name] * 100)
                else:
                    vals.append(np.nan)
            offset = (i - (len(focus) - 1) / 2) * width
            bars = ax.bar(x + offset, vals, width, label=m, color=colors_method[i])
            for bar, val in zip(bars, vals):
                if not np.isnan(val):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                            f"{val:.1f}", ha="center", fontsize=7, rotation=90)
        ax.set_ylabel(rate_label)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    col_labels = [f"{k[0]}\nCF={k[1]}" for k in sorted_keys]
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(col_labels, rotation=45, ha="right", fontsize=9)
    fig.tight_layout()
    _save(fig, save_dir, "focus_fpr_fnr_per_group")


def plot_focus_precision_recall_f1_per_group(structured, save_dir):
    """Detailed Precision/Recall/F1 per (sim, CF) group for focus methods."""
    focus = [m for m in FOCUS_METHODS if m in structured]

    all_keys = set()
    for m in focus:
        per_group = _get_method_per_group_metrics(structured, m)
        all_keys.update(per_group.keys())
    sorted_keys = sorted(all_keys, key=lambda k: (k[0], float(k[1])))

    fig, axes = plt.subplots(len(focus), 1, figsize=(14, 5 * len(focus)), sharex=True)
    if len(focus) == 1:
        axes = [axes]

    for ax, m in zip(axes, focus):
        per_group = _get_method_per_group_metrics(structured, m)
        precs, recs, f1s = [], [], []
        for k in sorted_keys:
            if k in per_group:
                precs.append(per_group[k]["precision"] * 100)
                recs.append(per_group[k]["recall"] * 100)
                f1s.append(per_group[k]["f1"] * 100)
            else:
                precs.append(np.nan)
                recs.append(np.nan)
                f1s.append(np.nan)

        x = np.arange(len(sorted_keys))
        width = 0.25
        ax.bar(x - width, precs, width, label="Precision", color="#4C72B0")
        ax.bar(x, recs, width, label="Recall", color="#55A868")
        ax.bar(x + width, f1s, width, label="F1 Score", color="#DD8452")

        for xi, (p, r, f) in enumerate(zip(precs, recs, f1s)):
            for val, off in [(p, -width), (r, 0), (f, width)]:
                if not np.isnan(val):
                    ax.text(xi + off, val + 0.5, f"{val:.0f}", ha="center", fontsize=8)

        ax.set_ylabel("Score (%)")
        ax.set_title(m, fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.set_ylim(0, 110)
        ax.grid(axis="y", alpha=0.3)

    col_labels = [f"{k[0]}\nCF={k[1]}" for k in sorted_keys]
    axes[-1].set_xticks(np.arange(len(sorted_keys)))
    axes[-1].set_xticklabels(col_labels, rotation=45, ha="right", fontsize=9)
    fig.tight_layout()
    _save(fig, save_dir, "focus_precision_recall_f1_per_group")


def plot_focus_metrics_summary(structured, save_dir):
    """Summary bar chart comparing FPR, FNR, Precision, Recall, F1 for focus methods."""
    focus = [m for m in FOCUS_METHODS if m in structured]
    colors_method = ["#4C72B0", "#DD8452", "#55A868"]

    categories = ["FPR", "FNR", "Precision", "Recall", "F1"]
    x = np.arange(len(categories))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, m in enumerate(focus):
        metrics = _get_method_avg_metrics(structured, m)
        if not metrics:
            continue
        vals = [
            metrics["fpr"] * 100,
            metrics["fnr"] * 100,
            metrics["precision"] * 100,
            metrics["recall"] * 100,
            metrics["f1"] * 100,
        ]
        offset = (i - (len(focus) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=m, color=colors_method[i])
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                    f"{val:.1f}", ha="center", fontsize=8)

    ax.set_ylabel("Score / Rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, save_dir, "focus_classification_metrics_summary")


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _save(fig, save_dir, name):
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, f"{name}.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(save_dir, f"{name}.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {name}")


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    methods, sim_types, client_fracs, metrics, data = parse_csv()
    structured = build_structured_data(methods, sim_types, client_fracs, metrics, data)

    print(f"\nParsed {len(structured)} detection methods.")
    print(f"Methods: {list(structured.keys())}\n")

    # --- General overview plots ---
    plot_overall_summary(structured, SAVE_DIR)
    plot_total_accuracy_ranking(structured, SAVE_DIR)
    plot_heatmap(structured, SAVE_DIR)
    plot_ls_vs_rot(structured, SAVE_DIR)
    plot_client_fraction_impact(structured, SAVE_DIR)
    plot_incremental_vs_gradual(structured, SAVE_DIR)
    plot_drifted_vs_benign_accuracy(structured, SAVE_DIR)
    plot_statistical_vs_mlp(structured, SAVE_DIR)

    # --- Classification metrics: FPR, FNR, Precision, Recall, F1 ---
    print("\n--- Classification metrics plots ---")
    plot_fpr_fnr(structured, SAVE_DIR)
    plot_precision_recall_f1(structured, SAVE_DIR)
    plot_f1_ranking(structured, SAVE_DIR)

    # --- Focus plots: FC2 Threshold + complete MLPs ---
    print("\n--- Focus plots (FC2 Threshold + complete MLPs) ---")
    plot_focus_full_breakdown(structured, SAVE_DIR)
    plot_focus_drifted_benign_breakdown(structured, SAVE_DIR)
    plot_focus_radar(structured, SAVE_DIR)
    plot_focus_cf_detailed(structured, SAVE_DIR)
    plot_focus_strengths_weaknesses(structured, SAVE_DIR)
    plot_focus_heatmap(structured, SAVE_DIR)
    plot_focus_summary_comparison(structured, SAVE_DIR)
    plot_focus_fpr_fnr_per_group(structured, SAVE_DIR)
    plot_focus_precision_recall_f1_per_group(structured, SAVE_DIR)
    plot_focus_metrics_summary(structured, SAVE_DIR)

    print(f"\nAll plots saved to {SAVE_DIR}")


if __name__ == "__main__":
    main()
