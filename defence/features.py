import os
import re

import torch as th
from torch.nn import functional

EPS = 1e-12

R_UPDATE_ROUND = re.compile(r"client_(\d+)_round_(\d+)\.pt")


def read_updates_from_round(dir_name: str, round_idx: int):
    """Return {filename: loaded_update} for a given round."""
    updates = os.listdir(dir_name)

    round_updates = []
    for u in updates:
        m = R_UPDATE_ROUND.match(u)
        if m and int(m.groups()[1]) == round_idx:
            round_updates.append(u)

    model_updates = {}
    for update_file in round_updates:
        model_updates[update_file] = th.load(
            os.path.join(dir_name, update_file), map_location="cpu"
        )

    return model_updates


def calculate_p_per_layer(delta_state_dict: dict, ignore_bias: bool = True):
    """Build a layer-profile vector p (shares per layer) using mean_abs per layer.

    Returns:
      layers: list[str]
      p: tensor shape (num_layers,) with sum(p)=1
    """
    layers = []
    vals = []

    for layer, data in delta_state_dict.items():
        if ignore_bias and "bias" in layer:
            continue
        vals.append(data.detach().norm(p=2))  # scalar tensor
        # vals.append(data.detach().abs().mean())  # scalar tensor
        layers.append(layer)

    v = th.stack(vals)  # shape (L,)
    p = v / (v.sum() + EPS)
    return layers, p


def entropy_from_probs(p: th.Tensor, normalized: bool = True) -> float:
    """p: probabilities that sum to 1 (non-negative)
    normalized=True -> divide by log(K) => entropy in [0,1]
    """
    h = -(p * (p + EPS).log()).sum()
    if not normalized:
        return float(h.item())
    h_max = th.log(th.tensor(p.numel(), dtype=p.dtype))
    return float((h / (h_max + EPS)).item())


def get_features_from_update(update: dict, mean_p: th.Tensor):
    """Extract features from one update.
    mean_p must be computed across the round first.
    """
    layers, p = calculate_p_per_layer(update["delta_state_dict"], ignore_bias=True)

    # global update strength (overall magnitude)
    global_l2 = float(th.norm(update["delta_vector"], 2).item())

    # entropy of the layer-profile (how concentrated is the update?)
    ent = entropy_from_probs(p, normalized=True)

    # head ratio: fc vs conv share
    head_sum = p[th.tensor([("fc" in l) for l in layers])].sum()
    conv_sum = p[th.tensor([("conv" in l) for l in layers])].sum()
    head_ratio = float((head_sum / (conv_sum + EPS)).item())

    # cosine similarity to round mean profile
    cos_to_mean = float(functional.cosine_similarity(p, mean_p, dim=0).item())

    return {
        "global_l2": global_l2,
        "entropy": ent,
        "head_ratio": head_ratio,
        "cos_to_mean": cos_to_mean,
        # Optional: direct share features are often very useful too:
        "share_fc2": float(p[layers.index("fc2.weight")].item())
        if "fc2.weight" in layers
        else None,
        "share_fc1": float(p[layers.index("fc1.weight")].item())
        if "fc1.weight" in layers
        else None,
        "share_conv1": float(p[layers.index("conv1.weight")].item())
        if "conv1.weight" in layers
        else None,
        "share_conv2": float(p[layers.index("conv2.weight")].item())
        if "conv2.weight" in layers
        else None,
    }
