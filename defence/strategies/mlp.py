from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch as th
from features import EPS, calculate_p_per_layer, get_features_from_update, read_updates_from_round
from torch import nn

# ---- Feature columns used by the MLP ----
FEATURE_COLS = [
    "s_fc2_fc1",
    "share_fc2",
    "share_fc1",
    "entropy",
    "head_ratio",
    "robust_dist",
    "global_l2",
    "share_conv1",
    "share_conv2",
    # "cos_to_mean",  # optional, usually close to 1.0
]


def _features_dict_to_matrix(features: Dict[str, dict]) -> Tuple[List[str], np.ndarray]:
    """Convert features dict {client_file -> {feature_name -> value}} into X matrix.

    Returns:
      client_files_sorted: List[str]
      X: np.ndarray shape (N, D)
    """
    client_files = sorted(features.keys())
    X = []
    for cf in client_files:
        row = []
        for c in FEATURE_COLS:
            v = features[cf].get(c, 0.0)
            if v is None:
                v = 0.0
            row.append(float(v))
        X.append(row)
    return client_files, np.asarray(X, dtype=np.float32)


class StandardScaler:
    """Simple numpy scaler (mean/std)."""

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X: np.ndarray):
        self.mean_ = X.mean(axis=0, keepdims=True)
        self.std_ = X.std(axis=0, keepdims=True) + 1e-12
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None and self.std_ is not None
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class MLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),  # logits
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class MLPDefenseConfig:
    lr: float = 1e-3
    epochs: int = 250
    batch_size: int = 32
    threshold: float = 0.5  # probability threshold at inference
    auto_pos_weight: bool = True  # compute imbalance weight from training data
    pos_weight: float = 1.0  # used if auto_pos_weight=False
    seed: int = 42


class MLPDefense:
    """Offline MLP defense with on-the-fly feature extraction from .pt updates.

    Usage pattern:
      defense = MLPDefense()
      defense.add_training_key(dir_name, round_idx)
      defense.add_training_key(...)
      defense.fit()  # loads pt files, extracts features, trains model

      preds = defense.classify_round(dir_name, round_idx)
    """

    def __init__(self, config: Optional[MLPDefenseConfig] = None):
        self.cfg = config or MLPDefenseConfig()
        self.scaler = StandardScaler()
        self.model = MLP(in_dim=len(FEATURE_COLS))
        self._is_fitted = False

        # list of (dir_name, round_idx) that form the training set
        self.train_keys: List[Tuple[str, int]] = []

        # for reproducibility
        th.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

    def add_training_key(self, dir_name: str, round_idx: int) -> None:
        """Add one (directory, round) pair to the training pool."""
        self.train_keys.append((dir_name, round_idx))

    def _extract_round_features_and_labels(
        self,
        dir_name: str,
        round_idx: int,
        label_mode: str = "malicious_only",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load the round updates, compute per-client features and labels.

        label_mode:
          - "malicious_only": label = is_drifted_client & drift_applied & malicious  (your current)
          - "drifted_any":    label = is_drifted_client & drift_applied
        """
        updates = read_updates_from_round(dir_name, round_idx)

        if not updates:
            raise ValueError(f"No updates found in {dir_name} for round {round_idx}")

        # --- build per-round profiles to get mean_p & median_p ---
        profiles = []
        for upd in updates.values():
            _, p = calculate_p_per_layer(upd["delta_state_dict"], ignore_bias=True)
            profiles.append(p)

        p_stack = th.stack(profiles)
        mean_p = p_stack.mean(dim=0)
        median_p = th.median(p_stack, dim=0).values
        mean_p /= mean_p.sum() + EPS
        median_p /= median_p.sum() + EPS

        # --- per-client features ---
        features: Dict[str, dict] = {}
        for client_file, upd in updates.items():
            features[client_file] = get_features_from_update(upd, mean_p, median_p)

        # --- labels ---
        client_files_sorted, X = _features_dict_to_matrix(features)

        if label_mode == "malicious_only":
            y = np.array(
                [
                    bool(
                        updates[cf]["is_drifted_client"]
                        and updates[cf]["drift_applied"]
                        and updates[cf]["malicious"]
                    )
                    for cf in client_files_sorted
                ],
                dtype=np.float32,
            )
        elif label_mode == "drifted_any":
            y = np.array(
                [
                    bool(updates[cf]["is_drifted_client"] and updates[cf]["drift_applied"])
                    for cf in client_files_sorted
                ],
                dtype=np.float32,
            )
        else:
            raise ValueError(f"Unknown label_mode: {label_mode}")

        return X, y

    def fit(self, label_mode: str = "malicious_only") -> None:
        """Extract features for all added train_keys and train the MLP.
        Feature extraction happens inside this function (no preprocessing step).
        """
        if not self.train_keys:
            raise RuntimeError("No training keys added. Call add_training_key(...) first.")

        # --- build training dataset (X, y) by extracting from pt files ---
        X_list = []
        y_list = []

        for dir_name, round_idx in self.train_keys:
            X_round, y_round = self._extract_round_features_and_labels(
                dir_name, round_idx, label_mode=label_mode
            )
            X_list.append(X_round)
            y_list.append(y_round)

        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)

        # --- scale ---
        Xs = self.scaler.fit_transform(X)

        xb = th.tensor(Xs, dtype=th.float32)
        yb = th.tensor(y, dtype=th.float32)

        ds = th.utils.data.TensorDataset(xb, yb)
        dl = th.utils.data.DataLoader(ds, batch_size=self.cfg.batch_size, shuffle=True)

        # --- imbalance handling ---
        if self.cfg.auto_pos_weight:
            pos = float(y.sum())
            neg = float(len(y) - pos)
            # avoid div-by-zero
            pos_weight_value = (neg / (pos + 1e-12)) if pos > 0 else 1.0
        else:
            pos_weight_value = float(self.cfg.pos_weight)

        loss_fn = nn.BCEWithLogitsLoss(pos_weight=th.tensor(pos_weight_value, dtype=th.float32))

        opt = th.optim.Adam(self.model.parameters(), lr=self.cfg.lr)

        # --- train ---
        self.model.train()
        for _ in range(self.cfg.epochs):
            for X_batch, y_batch in dl:
                logits = self.model(X_batch)
                loss = loss_fn(logits, y_batch)
                opt.zero_grad()
                loss.backward()
                opt.step()

        self._is_fitted = True
        # import datetime
        # th.save(self.model.state_dict(), f"defence/mlp_defense_model_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pt")

    @th.no_grad()
    def predict_proba_features(self, features: Dict[str, dict]) -> Tuple[List[str], np.ndarray]:
        """Predict probabilities from already computed features dict."""
        if not self._is_fitted:
            raise RuntimeError("MLPDefense not trained. Call fit() first.")

        client_files, X = _features_dict_to_matrix(features)
        Xs = self.scaler.transform(X)

        xb = th.tensor(Xs, dtype=th.float32)
        self.model.eval()
        probs = th.sigmoid(self.model(xb)).cpu().numpy()
        return client_files, probs

    @th.no_grad()
    def classify_features(self, features: Dict[str, dict]) -> List[bool]:
        """Classify from features dict (sorted client_file order)."""
        _, probs = self.predict_proba_features(features)
        return [float(p) >= self.cfg.threshold for p in probs]

    @th.no_grad()
    def classify_round(self, dir_name: str, round_idx: int) -> List[bool]:
        """Convenience wrapper:
        Load updates from round, compute mean/median p, extract features, predict.
        """
        updates = read_updates_from_round(dir_name, round_idx)

        profiles = []
        for upd in updates.values():
            _, p = calculate_p_per_layer(upd["delta_state_dict"], ignore_bias=True)
            profiles.append(p)

        p_stack = th.stack(profiles)
        mean_p = p_stack.mean(dim=0)
        median_p = th.median(p_stack, dim=0).values
        mean_p /= mean_p.sum() + EPS
        median_p /= median_p.sum() + EPS

        features: Dict[str, dict] = {}
        for client_file, upd in updates.items():
            features[client_file] = get_features_from_update(upd, mean_p, median_p)

        return self.classify_features(features)

    def save(self, path: str) -> None:
        """Save model + scaler + config to a single .pt file.
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save an unfitted model. Train (fit) it first.")

        payload = {
            "model_state_dict": self.model.state_dict(),
            "scaler_mean": th.tensor(self.scaler.mean_, dtype=th.float32),
            "scaler_std": th.tensor(self.scaler.std_, dtype=th.float32),
            "feature_cols": FEATURE_COLS,
            "config": {
                "lr": self.cfg.lr,
                "epochs": self.cfg.epochs,
                "batch_size": self.cfg.batch_size,
                "threshold": self.cfg.threshold,
                "auto_pos_weight": self.cfg.auto_pos_weight,
                "pos_weight": self.cfg.pos_weight,
                "seed": self.cfg.seed,
            },
        }
        th.save(payload, path)

    @classmethod
    def load(cls, path: str) -> "MLPDefense":
        try:
            ckpt = th.load(path, map_location="cpu")  # default weights_only=True
        except Exception:
            ckpt = th.load(path, map_location="cpu", weights_only=False)  # fallback

        cfg_dict = ckpt.get("config", {})
        cfg = MLPDefenseConfig(
            lr=float(cfg_dict.get("lr", 1e-3)),
            epochs=int(cfg_dict.get("epochs", 250)),
            batch_size=int(cfg_dict.get("batch_size", 32)),
            threshold=float(cfg_dict.get("threshold", 0.5)),
            auto_pos_weight=bool(cfg_dict.get("auto_pos_weight", True)),
            pos_weight=float(cfg_dict.get("pos_weight", 1.0)),
            seed=int(cfg_dict.get("seed", 42)),
        )

        obj = cls(cfg)

        # mean/std may be numpy arrays or tensors depending on how you saved
        mean = ckpt["scaler_mean"]
        std = ckpt["scaler_std"]

        if isinstance(mean, th.Tensor):
            obj.scaler.mean_ = mean.cpu().numpy()
        else:
            obj.scaler.mean_ = mean

        if isinstance(std, th.Tensor):
            obj.scaler.std_ = std.cpu().numpy()
        else:
            obj.scaler.std_ = std

        obj.model.load_state_dict(ckpt["model_state_dict"])
        obj._is_fitted = True
        return obj
