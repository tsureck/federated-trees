from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
from collections import OrderedDict
import hashlib
import time

import torch
import torch.nn as nn


StateDict = Dict[str, torch.Tensor]


def _stable_tensor_meta(t: torch.Tensor) -> str:
    # Shapes + dtype are enough for a stable "spec" signature
    return f"{tuple(t.shape)}|{str(t.dtype)}"


@dataclass(frozen=True)
class ModelSpec:
    """A stable description of a model's state_dict layout.
    Use this to ensure deltas from different models aren't mixed accidentally.
    """
    model_name: str
    param_names: Tuple[str, ...]
    param_meta: Tuple[str, ...]          # e.g. "(32,1,3,3)|torch.float32"
    num_params_total: int
    signature: str                       # hash over (names + meta)

    @staticmethod
    def from_model(model: nn.Module, model_name: Optional[str] = None) -> "ModelSpec":
        sd = model.state_dict()
        names: List[str] = []
        meta: List[str] = []
        total = 0

        # state_dict() is ordered in modern PyTorch; we preserve that order
        for k, v in sd.items():
            names.append(k)
            meta.append(_stable_tensor_meta(v))
            total += v.numel()

        sig_input = (model_name or model.__class__.__name__) + "||" + "||".join(
            f"{n}:{m}" for n, m in zip(names, meta)
        )
        signature = hashlib.sha256(sig_input.encode("utf-8")).hexdigest()[:16]

        return ModelSpec(
            model_name=model_name or model.__class__.__name__,
            param_names=tuple(names),
            param_meta=tuple(meta),
            num_params_total=total,
            signature=signature,
        )


def compute_delta_state_dict(global_sd: StateDict, client_sd: StateDict, spec: ModelSpec,
                             dtype: torch.dtype = torch.float32) -> "OrderedDict[str, torch.Tensor]":
    """Compute delta = client - global in the exact parameter order of the spec.
    Stores tensors detached on CPU with chosen dtype (default float32).
    """
    delta = OrderedDict()
    for name in spec.param_names:
        if name not in global_sd or name not in client_sd:
            raise KeyError(f"Missing key '{name}' in global_sd or client_sd.")
        d = (client_sd[name].detach() - global_sd[name].detach()).to("cpu", dtype=dtype)
        delta[name] = d
    return delta


def flatten_delta(delta_sd: "OrderedDict[str, torch.Tensor]") -> torch.Tensor:
    """Flatten ordered tensors to one 1D vector."""
    return torch.cat([t.reshape(-1) for t in delta_sd.values()], dim=0)


def delta_features(delta_vec: torch.Tensor) -> Dict[str, float]:
    """A few cheap, useful scalar features."""
    with torch.no_grad():
        abs_v = delta_vec.abs()
        return {
            "l2": float(torch.linalg.vector_norm(delta_vec, ord=2)),
            "l1": float(torch.linalg.vector_norm(delta_vec, ord=1)),
            "max_abs": float(abs_v.max()),
            "mean_abs": float(abs_v.mean()),
            "std": float(delta_vec.float().std(unbiased=False)),
            "sparsity_1e-6": float((abs_v < 1e-6).float().mean()),
        }

@dataclass
class ClientUpdateRecord:
    """One client update at one round.
    Stores:
      - metadata
      - delta_state_dict (client - global)
      - optional delta_vector (flattened)
      - derived features (norms etc.)
    """
    # identity / ordering
    client_id: str
    round_id: int
    received_ts: float

    # model identity (prevents mixing MNIST/CIFAR, or changed architectures)
    model_signature: str
    model_name: str

    # base reference info for correctness / reproducibility
    base_global_round: int
    base_global_hash: str

    # weighting / training context
    num_examples: int
    local_steps: Optional[int] = None
    local_epochs: Optional[int] = None
    hyper: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)

    # payload
    delta_state_dict: "OrderedDict[str, torch.Tensor]" = field(default_factory=OrderedDict)
    delta_vector: Optional[torch.Tensor] = None

    # derived
    features: Dict[str, float] = field(default_factory=dict)
    malicious: bool = False # whether this update was generated under malicious drift

    def ensure_vector_and_features(self) -> None:
        if self.delta_vector is None:
            self.delta_vector = flatten_delta(self.delta_state_dict)
        if not self.features:
            self.features = delta_features(self.delta_vector)

    def save_torch(self, path: str) -> None:
        """Saves everything (including tensors) via torch.save."""
        self.ensure_vector_and_features()
        payload = {
            "meta": {
                "client_id": self.client_id,
                "round_id": self.round_id,
                "received_ts": self.received_ts,
                "model_signature": self.model_signature,
                "model_name": self.model_name,
                "base_global_round": self.base_global_round,
                "base_global_hash": self.base_global_hash,
                "num_examples": self.num_examples,
                "local_steps": self.local_steps,
                "local_epochs": self.local_epochs,
                "hyper": self.hyper,
                "metrics": self.metrics,
                "features": self.features,
            },
            "delta_state_dict": self.delta_state_dict,
            "delta_vector": self.delta_vector,
            "malicious": self.malicious
        }
        torch.save(payload, path)

    @staticmethod
    def load_torch(path: str) -> "ClientUpdateRecord":
        payload = torch.load(path, map_location="cpu")
        meta = payload["meta"]
        rec = ClientUpdateRecord(
            client_id=meta["client_id"],
            round_id=meta["round_id"],
            received_ts=meta["received_ts"],
            model_signature=meta["model_signature"],
            model_name=meta["model_name"],
            base_global_round=meta["base_global_round"],
            base_global_hash=meta["base_global_hash"],
            num_examples=meta["num_examples"],
            local_steps=meta.get("local_steps"),
            local_epochs=meta.get("local_epochs"),
            hyper=meta.get("hyper", {}),
            metrics=meta.get("metrics", {}),
            features=meta.get("features", {}),
            delta_state_dict=payload["delta_state_dict"],
            delta_vector=payload.get("delta_vector"),
            malicious=meta.get("malicious", False)
        )
        return rec

    @classmethod
    def build_update_record(
        cls,
        *,
        client_id: str,
        round_id: int,
        global_sd: StateDict,
        client_sd: StateDict,
        spec: ModelSpec,
        base_global_round: int,
        base_global_hash: str,
        num_examples: int,
        dtype_for_storage: torch.dtype = torch.float32,
        local_steps: Optional[int] = None,
        local_epochs: Optional[int] = None,
        hyper: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        store_flat_vector: bool = True,
        malicious: bool = False,
    ) -> ClientUpdateRecord:
        delta_sd = compute_delta_state_dict(global_sd, client_sd, spec, dtype=dtype_for_storage)
        vec = flatten_delta(delta_sd) if store_flat_vector else None

        rec = cls(
            client_id=client_id,
            round_id=round_id,
            received_ts=time.time(),
            model_signature=spec.signature,
            model_name=spec.model_name,
            base_global_round=base_global_round,
            base_global_hash=base_global_hash,
            num_examples=num_examples,
            local_steps=local_steps,
            local_epochs=local_epochs,
            hyper=hyper or {},
            metrics=metrics or {},
            delta_state_dict=delta_sd,
            delta_vector=vec,
            malicious=malicious
        )
        rec.ensure_vector_and_features()
        return rec
