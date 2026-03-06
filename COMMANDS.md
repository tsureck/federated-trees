# Commands Reference

All commands are run from the project root unless otherwise noted.

---

## Scripts Overview

| Script | Purpose |
|---|---|
| `main.py` | Run a federated learning simulation (basic, without defence) |
| `main_defence.py` | Run a federated learning simulation saving client updates for defence analysis |
| `constants.py` | Central configuration (paths, dataset, drift settings, training hyperparameters) |
| `plots/plot_from_logs.py` | Generate performance plots from a completed simulation's log files |
| `plots/plot_defence_stats.py` | Generate comparison plots from a defence statistics spreadsheet CSV |
| `plots/plotting.py` | Library — plotting helpers (not a standalone script) |
| `defence/defence_simulation.py` | Run all defence variants over a simulation's client updates, saves results to CSV |
| `defence/train_mlp.py` | Train an MLP defence model on client update data and save a checkpoint |
| `defence/plot_defence_rounds.py` | Plot per-round accuracy curves for each defence method from a defence CSV |
| `defence/print_defence_metrics.py` | Print/export precision, recall, F1, FPR, FNR per defence from a defence CSV |
| `defence/features.py` | Library — feature extraction from model updates (not a standalone script) |
| `defence/strategies/mlp.py` | Library — MLP defence strategy implementation (not a standalone script) |
| `defence/strategies/share_fc2.py` | Library — statistical defence strategies (not a standalone script) |
| `scripts/compare_runs.py` | Compare client updates from two simulation runs, generates per-round heatmaps |
| `scripts/update_comparison.py` | Compare two individual `.pt` update files at the convolutional filter level |
| `logs/analysis_functions.py` | Library — log analysis helpers (not a standalone script) |
| `logs/logging.py` | Library — log read/write utilities (not a standalone script) |

---

## Flow 1: Basic FL Simulation + Performance Plots

Run a federated learning simulation, then plot accuracy/loss curves from the saved logs.

**Step 1 — Configure the simulation**

Edit `constants.py` to set dataset, drift pattern, saving behaviour, etc.
Key settings:
```python
UpdateDataSettings.SAVE_DATASET_UPDATES = False  # no update files needed here
ModelSettings.LOAD_MODEL_STATE = True/False
AnalysisSettings.PLOTTING_FORMAT = "png"  # or "pgf" for PDF output
```

**Step 2 — Run the simulation**

```bash
python main.py
```

Output is written to `fl_runs/<dataset>/<drift_method>/<pattern>/<params>/<run_timestamp>/`:
- `logs/` — pickle log files (`client_log.pkl`, `server_log.pkl`, etc.)
- `plots/` — auto-generated drift transition plot
- `updates/` — client update `.pt` files (if `SAVE_DATASET_UPDATES = True`)

**Step 3 — Plot performance from logs**

Edit the `dir_name` variable at the bottom of `plots/plot_from_logs.py` to point to the run directory, then:

```bash
python plots/plot_from_logs.py
```

Outputs PNG/PDF plots into the run's `plots/` directory:
- Client loss and accuracy per round
- Server loss and accuracy per round
- Drifted vs non-drifted client average performance
- Server level and overall average performance

---

## Flow 2: Defence Simulation + Analysis

Run a simulation that saves client updates, evaluate defence detection algorithms, then visualize results.

**Step 1 — Configure for update saving**

Edit `constants.py`:
```python
UpdateDataSettings.SAVE_DATASET_UPDATES = True
```

**Step 2 — Run the simulation**

```bash
python main_defence.py
```

This saves client update `.pt` files to `fl_runs/.../updates/client_updates/`.

**Step 3 — Run defence evaluation**

Edit the `dir_name_ls` and `dir_name_rot` paths in `defence/defence_simulation.py` to point to the label-swapping and rotation run directories respectively, then:

```bash
python defence/defence_simulation.py
```

Outputs `defence/defence_stats_<timestamp>.csv` with per-round classification counts for all defence variants.

**Step 4a — Plot per-round accuracy curves**

```bash
python defence/plot_defence_rounds.py defence/defence_stats_<timestamp>.csv
```

Outputs `defence/defence_stats_<timestamp>_accuracy_rounds.png` and `.pdf` — two-panel plot (label swapping / rotation) showing accuracy over rounds for all defence methods.

**Step 4b — Print and export aggregate metrics**

```bash
# Print metrics table to console
python defence/print_defence_metrics.py defence/defence_stats_<timestamp>.csv

# Also export metrics to CSV
python defence/print_defence_metrics.py defence/defence_stats_<timestamp>.csv --out defence/metrics.csv

# Also generate metric bar charts and scatter plots
python defence/print_defence_metrics.py defence/defence_stats_<timestamp>.csv --plot defence/plots/
```

Prints total accuracy, F1, precision, recall, FNR, FPR per defence — broken down by attack type and combined. The `--plot` flag generates bar charts and FPR/FNR scatter plots.

---

## Flow 3: MLP Defence Training

Train a custom MLP detector on client update data before running the defence simulation.

**Step 1 — Run simulations to generate training data**

Complete Flow 2 Steps 1–2 for each simulation configuration you want to train on.

**Step 2 — Train the MLP**

Edit the training paths in `defence/train_mlp.py` to point to your simulation directories and the desired training round range, then:

```bash
python defence/train_mlp.py
```

Saves `defence/mlp_defense.pt`.

**Step 3 — Use the checkpoint in defence evaluation**

Add the checkpoint path to the `MLP_CHECKPOINTS` list in `defence/defence_simulation.py`, then run Flow 2 Step 3 as normal.

---

## Flow 4: Compare Two Simulation Runs

Visualize differences in client update profiles (layer weight distributions) between two runs.

**Step 1 — Run two simulations**

Complete Flow 2 Steps 1–2 for two different configurations (e.g. label swapping vs rotation).

**Step 2 — Run the comparison**

Edit `dir_a` and `dir_b` in `scripts/compare_runs.py` to point to the two `client_updates` directories, then:

```bash
python scripts/compare_runs.py
```

Outputs per-round heatmaps and drift comparison plots (drifted vs non-drifted clients, method A vs B) to `scripts/output/<timestamp>/`.

---

## Flow 5: Compare Individual Update Files

Inspect convolutional filter differences between two specific `.pt` update files.

**Step 1 — Identify the files**

Locate two `.pt` client update files (e.g. `client_0_round_20.pt` from two different runs).

**Step 2 — Run the comparison**

Edit the file paths in `scripts/update_comparison.py`, then:

```bash
python scripts/update_comparison.py
```

Outputs a filter mosaic image (update 1 / update 2 / difference) for the selected convolutional layer.

---

## Flow 6: Defence Statistics Spreadsheet Plots

Generate method comparison plots from a manually curated spreadsheet CSV (e.g. after aggregating results from multiple runs into the CSV manually).

**Step 1 — Place the CSV**

Ensure `Federated Learning Defence Statistics - Sheet1_v2.csv` is at the project root.

**Step 2 — Run the plotting script**

```bash
python plots/plot_defence_stats.py
```

Outputs plots to `plots/defence_stats_c-8/`.

---

## Configuration Reference (`constants.py`)

| Class | Key Setting | Description |
|---|---|---|
| `UpdateDataSettings` | `SAVE_DATASET_UPDATES` | Set `True` to save client updates for defence analysis |
| `UpdateDataSettings` | `DATA_SAVE_PATH` | Root directory for simulation output (default: `./fl_runs`) |
| `ModelSettings` | `LOAD_MODEL_STATE` | Load a pre-trained model to continue from |
| `ModelSettings` | `SAVE_MODEL_STATE` | Save the root server model after training |
| `AnalysisSettings` | `PLOTTING_FORMAT` | `"png"` for PNG output, `"pgf"` for PDF via LaTeX backend |
| `TrainingSettings` | `MAX_BATCHES_PER_EPOCH` | Limit batches per epoch (lower = faster, less accurate) |
| `DriftPatterns` | — | `ABRUPT`, `GRADUAL`, `INCREMENTAL`, `GRADUAL_REOCCURRING` |
| `DriftCreationMethods` | — | `LABEL_SWAPPING`, `ROTATION` |
