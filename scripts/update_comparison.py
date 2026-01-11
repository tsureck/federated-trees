import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

def reduce_over_in_channels(w: np.ndarray, mode: str = "mean") -> np.ndarray:
    """
    w: (out_ch, in_ch, 3, 3)  -> returns (out_ch, 3, 3)
    mode:
      - "mean": mean over in_ch (signed, can cancel)
      - "mean_abs": mean(abs(w)) over in_ch (magnitude)
      - "rms": sqrt(mean(w^2)) over in_ch (magnitude)
    """
    assert w.ndim == 4 and w.shape[2:] == (3, 3), f"Expected (O,I,3,3), got {w.shape}"

    if mode == "mean":
        return w.mean(axis=1)
    if mode == "mean_abs":
        return np.abs(w).mean(axis=1)
    if mode == "rms":
        return np.sqrt((w ** 2).mean(axis=1))

    raise ValueError(f"Unknown mode: {mode}")
def stacked_filter_mosaic_3rows(w1: np.ndarray, w2: np.ndarray, pad: int = 1) -> np.ndarray:
    """
    w1, w2: (N, 1, 3, 3) or (N, 3, 3)
    returns mosaic with 3 rows of tiles: [w1; w2; (w1-w2)]
    """
    if w1.ndim == 4:  # (N, 1, 3, 3)
        w1 = w1[:, 0]
    if w2.ndim == 4:
        w2 = w2[:, 0]

    assert w1.shape == w2.shape and w1.shape[1:] == (3, 3), f"Got {w1.shape} vs {w2.shape}"
    diff = w1 - w2

    n = w1.shape[0]
    kh, kw = 3, 3

    H = 3 * kh + 2 * pad
    W = n * kw + (n - 1) * pad
    mosaic = np.full((H, W), np.nan, dtype=np.float32)

    for i in range(n):
        x0 = i * (kw + pad)
        # row 0
        mosaic[0:kh, x0:x0 + kw] = w1[i]
        # row 1
        y1 = kh + pad
        mosaic[y1:y1 + kh, x0:x0 + kw] = w2[i]
        # row 2
        y2 = 2 * kh + 2 * pad
        mosaic[y2:y2 + kh, x0:x0 + kw] = diff[i]

    return mosaic

def plot_stacked_filters_3rows(w1: np.ndarray, w2: np.ndarray, out_path="conv1_stacked_3rows.png", pad: int = 1, layer: str = "conv1.weight"):
    mosaic = stacked_filter_mosaic_3rows(w1, w2, pad=pad)

    # Use shared normalization across w1/w2/diff
    if w1.ndim == 4: w1_ = w1[:, 0]
    else: w1_ = w1
    if w2.ndim == 4: w2_ = w2[:, 0]
    else: w2_ = w2
    diff_ = w1_ - w2_

    vmax = float(np.nanmax(np.abs([w1_.min(), w1_.max(), w2_.min(), w2_.max(), diff_.min(), diff_.max()])))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad(color="lightgray")  # padding (NaNs)

    fig, ax = plt.subplots(figsize=(18, 4))
    im = ax.imshow(mosaic, cmap=cmap, norm=norm, interpolation="nearest", aspect="auto")

    n = (w1.shape[0] if w1.ndim == 4 else w1_.shape[0])
    kw = 3

    centers = [i * (kw + pad) + (kw - 1) / 2 for i in range(n)]
    ax.set_xticks(centers[::2])
    ax.set_xticklabels(list(range(n))[::2])
    ax.set_xlabel("Filter index")

    # y tick centers for each 3x3 row
    y0 = (3 - 1) / 2
    y1 = (3 + pad) + (3 - 1) / 2
    y2 = (2 * (3 + pad)) + (3 - 1) / 2
    ax.set_yticks([y0, y1, y2])
    ax.set_yticklabels(["Update 1", "Update 2", "Diff (u1-u2)"])

    ax.set_title(f"{layer} (3×3 kernels) — stacked: update1 / update2 / diff")
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()

def compare_pth_files(file1: str, file2: str, layer: str):
    update_1 = torch.load(file1, map_location="cpu")
    update_2 = torch.load(file2, map_location="cpu")

    w1 = update_1["delta_state_dict"]["conv2.weight"].detach().cpu().numpy()  # (64,32,3,3)
    w2 = update_2["delta_state_dict"]["conv2.weight"].detach().cpu().numpy()
    mode = "rms"
    w1_red = reduce_over_in_channels(w1, mode=mode)       # (64,3,3)
    w2_red = reduce_over_in_channels(w2, mode=mode)

    plot_stacked_filters_3rows(w1_red, w2_red, out_path=f"conv2_avgIn_stacked_3rows_{mode}.png", pad=1)

if __name__ == "__main__":
    compare_pth_files(
        "scripts/rotation_incremental_client_0_round_20.pt",
        "scripts/label_swapping_incremental_client_0_round_20.pt",
        "conv2.weight"
    )