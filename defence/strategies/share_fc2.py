import torch as th

EPS = 1e-12


def classify_by_share_fc2_threshold(features: dict, updates: dict, median_flag: bool = True) -> None:
    """Classify clients as malicious if their share_fc2 exceeds a threshold."""
    share_fc2_median = th.median(
        th.tensor([feat["share_fc2"] for _, feat in features.items()])
    ).item()
    share_fc2_mean = th.mean(th.tensor([feat["share_fc2"] for _, feat in features.items()])).item()

    threshold = (share_fc2_mean - 0.5 * (share_fc2_mean - share_fc2_median)) / 10
    base = share_fc2_median if median_flag else share_fc2_mean
    malicious = [
        (feat["share_fc2"] - base) > threshold
        for _, feat in features.items()]

    return malicious


def _score_s(features: dict, inverse: bool = False) -> th.Tensor:
    """Build the score vector s for each client:
    s = share_fc2 - share_fc1   (default)
    s = share_fc1 - share_fc2   (inverse=True)
    """
    s_list = []
    for feat in features.values():
        fc2 = float(feat["share_fc2"])
        fc1 = float(feat["share_fc1"])
        s_list.append((fc1 - fc2) if inverse else (fc2 - fc1))
    return th.tensor(s_list, dtype=th.float32)


def classify_by_s_mad(features: dict, k: float = 2.0, inverse: bool = False) -> list[bool]:
    """Robust thresholding using MAD:
      threshold = median(s) + k * MAD(s)

    Flags: s > threshold

    Note:
    - Assumes benign majority; if ~50/50 split, median may shift.
    """
    s = _score_s(features, inverse=inverse)

    med = th.median(s)
    mad = th.median(th.abs(s - med)) + EPS

    threshold = med + k * mad
    return [(val.item() > threshold.item()) for val in s]


def _kmeans_1d_two_clusters(x: th.Tensor, iters: int = 30) -> tuple[float, float, th.Tensor]:
    """1D 2-means. Returns sorted centers (c0 < c1) and labels.
    """
    x = x.flatten()

    # init centers
    c0 = x.min()
    c1 = x.max()

    for _ in range(iters):
        d0 = (x - c0).abs()
        d1 = (x - c1).abs()
        labels = (d1 < d0).long()

        c0_new = x[labels == 0].mean() if (labels == 0).any() else c0
        c1_new = x[labels == 1].mean() if (labels == 1).any() else c1

        if th.allclose(c0, c0_new) and th.allclose(c1, c1_new):
            break
        c0, c1 = c0_new, c1_new

    if c0 > c1:
        c0, c1 = c1, c0
        labels = 1 - labels

    return c0.item(), c1.item(), labels


def classify_by_s_kmeans(features: dict, margin: float = 0.0, inverse: bool = False) -> list[bool]:
    """2-cluster k-means threshold:
      threshold = midpoint between centers (+ optional margin)

    Flags: s > threshold
    """
    s = _score_s(features, inverse=inverse)
    c0, c1, _ = _kmeans_1d_two_clusters(s)

    threshold = (c0 + c1) / 2.0 + margin
    return [(val.item() > threshold) for val in s]


def classify_by_s_iqr(features: dict, alpha: float = 1.5, inverse: bool = False) -> list[bool]:
    """IQR-based thresholding:
      threshold = Q3 + alpha * IQR
      IQR = Q3 - Q1

    Flags: s > threshold

    This is a typical outlier rule, but may be too conservative depending on your separation.
    """
    s = _score_s(features, inverse=inverse)

    q1 = th.quantile(s, 0.25).item()
    q3 = th.quantile(s, 0.75).item()
    iqr = (q3 - q1) + EPS

    threshold = q3 + alpha * iqr
    return [(val.item() > threshold) for val in s]


def classify_by_s_lowerhalf_mad(features: dict, k: float = 2.0, inverse: bool = False) -> list[bool]:
    """Robust one-sided threshold on:
      s = share_fc2 - share_fc1   (default)
      s = share_fc1 - share_fc2   (inverse=True)

    Uses ONLY the lower half of s-values to estimate benign baseline:
      base = median(lower_half)
      spread = MAD(lower_half)
      threshold = base + k * spread

    Flags: s > threshold
    """
    s_list = []
    for feat in features.values():
        fc2 = float(feat["share_fc2"])
        fc1 = float(feat["share_fc1"])
        s_list.append((fc1 - fc2) if inverse else (fc2 - fc1))

    s = th.tensor(s_list, dtype=th.float32)

    med_all = th.median(s)

    # lower half (benign-biased)
    lower = s[s <= med_all]

    # fallback if something goes wrong
    if lower.numel() < 2:
        lower = s

    base = th.median(lower)
    mad = th.median(th.abs(lower - base)) + EPS

    threshold = base + k * mad

    return [(val.item() > threshold.item()) for val in s]


def classify_by_share_fc2_lowerhalf_mad(features: dict, k: float = 2.0) -> list[bool]:
    """Robust one-sided threshold on share_fc2:
    threshold = median(lower_half) + k * MAD(lower_half)
    """
    x = th.tensor([float(f["share_fc2"]) for f in features.values()], dtype=th.float32)

    med_all = th.median(x)
    lower = x[x <= med_all]
    if lower.numel() < 2:
        lower = x

    base = th.median(lower)
    mad = th.median(th.abs(lower - base)) + EPS
    threshold = base + k * mad

    return [(val.item() > threshold.item()) for val in x]
