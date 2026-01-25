import torch as th


def classify_by_share_fc2_threshold(features: dict, updates: dict, median_flag: bool = True) -> None:
    """Classify clients as malicious if their share_fc2 exceeds a threshold."""
    share_fc2_median = th.median(
        th.tensor([feat["share_fc2"] for _, feat in features.items()])
    ).item()
    share_fc2_mean = th.mean(th.tensor([feat["share_fc2"] for _, feat in features.items()])).item()

    threshold = (share_fc2_mean - 0.5 * (share_fc2_mean - share_fc2_median)) / 10
    # print(threshold)
    # print([(feat["share_fc2"] - share_fc2_median) for _, feat in features.items()])
    # print([(feat["share_fc2"] - share_fc2_mean) for _, feat in features.items()])
    # print([(feat["share_fc2"] - share_fc2_mean) > threshold for _, feat in features.items()])
    base = share_fc2_median if median_flag else share_fc2_mean
    malicious = [
        (feat["share_fc2"] - base) > threshold
        for _, feat in features.items()]

    return malicious
