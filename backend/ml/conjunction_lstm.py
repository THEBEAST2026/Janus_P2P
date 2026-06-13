"""
ml/conjunction_lstm.py
PyTorch LSTM model for conjunction prediction.
Input:  20-step window of 9 orbital features per satellite pair
Output: risk_prob, miss_dist, tca_secs, who_evades
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass


# ─── Model ────────────────────────────────────────────────────────────────────
class ConjunctionLSTM(nn.Module):
    """
    LSTM that takes a (batch, 20, 9) tensor and outputs 4 predictions.
    """

    def __init__(
        self,
        input_size: int = 9,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_size)

        # Output heads
        self.head_risk    = nn.Sequential(nn.Linear(hidden_size, 1), nn.Sigmoid())
        self.head_dist    = nn.Sequential(nn.Linear(hidden_size, 1), nn.ReLU())
        self.head_tca     = nn.Sequential(nn.Linear(hidden_size, 1), nn.ReLU())
        self.head_evader  = nn.Sequential(nn.Linear(hidden_size, 3), nn.Softmax(dim=-1))

    def forward(self, x: torch.Tensor):
        # x: (batch, seq=20, features=9)
        out, _ = self.lstm(x)
        last = self.norm(out[:, -1, :])  # take last timestep
        return {
            "risk_prob":  self.head_risk(last).squeeze(-1),    # (batch,)
            "miss_dist":  self.head_dist(last).squeeze(-1),
            "tca_secs":   self.head_tca(last).squeeze(-1),
            "who_evades": self.head_evader(last),              # (batch, 3)
        }


# ─── Inference wrapper ────────────────────────────────────────────────────────
@dataclass
class ConjunctionPrediction:
    risk_prob: float       # 0–1
    miss_dist_km: float    # km
    tca_secs: float        # seconds to closest approach
    who_evades: str        # "A" | "B" | "SPLIT"
    raw_evader_probs: list[float]


EVADER_LABELS = ["A", "B", "SPLIT"]


class LSTMPredictor:
    """
    Wraps the LSTM model for single-pair inference.
    Loaded once at startup, shared across all SatelliteAgent instances.
    """

    def __init__(self, model_path: str | None = None):
        self.model = ConjunctionLSTM()
        if model_path:
            try:
                state = torch.load(model_path, map_location="cpu", weights_only=True)
                self.model.load_state_dict(state)
                print(f"✅ LSTM weights loaded from {model_path}")
            except Exception as e:
                print(f"⚠️  Could not load weights ({e}) — using random init")
        self.model.eval()

    def predict(self, feature_window: list[list[float]]) -> ConjunctionPrediction:
        """
        feature_window: list of 20 steps, each step = 9 floats
        [miss_distance, rel_vx, rel_vy, rel_vz, tca_est,
         alt_A, alt_B, fuel_A, fuel_B]
        """
        if len(feature_window) < 20:
            # Pad with zeros if buffer not full yet
            pad = [[0.0] * 9] * (20 - len(feature_window))
            feature_window = pad + feature_window

        x = torch.tensor([feature_window], dtype=torch.float32)  # (1, 20, 9)

        with torch.no_grad():
            out = self.model(x)

        risk_prob   = float(out["risk_prob"][0])
        miss_dist   = float(out["miss_dist"][0])
        tca_secs    = float(out["tca_secs"][0])
        evader_probs = out["who_evades"][0].tolist()
        who_evades  = EVADER_LABELS[int(torch.argmax(out["who_evades"][0]))]

        return ConjunctionPrediction(
            risk_prob=round(risk_prob, 4),
            miss_dist_km=round(miss_dist, 3),
            tca_secs=round(tca_secs, 1),
            who_evades=who_evades,
            raw_evader_probs=[round(p, 3) for p in evader_probs],
        )

    def batch_predict(self, windows: list[list[list[float]]]) -> list[ConjunctionPrediction]:
        """Batch inference for multiple pairs."""
        padded = []
        for w in windows:
            if len(w) < 20:
                pad = [[0.0] * 9] * (20 - len(w))
                w = pad + w
            padded.append(w)

        x = torch.tensor(padded, dtype=torch.float32)
        with torch.no_grad():
            out = self.model(x)

        results = []
        for i in range(len(padded)):
            evader_probs = out["who_evades"][i].tolist()
            results.append(ConjunctionPrediction(
                risk_prob=round(float(out["risk_prob"][i]), 4),
                miss_dist_km=round(float(out["miss_dist"][i]), 3),
                tca_secs=round(float(out["tca_secs"][i]), 1),
                who_evades=EVADER_LABELS[int(torch.argmax(out["who_evades"][i]))],
                raw_evader_probs=[round(p, 3) for p in evader_probs],
            ))
        return results


# ─── Feature extractor ────────────────────────────────────────────────────────
def extract_features(sat_a, sat_b) -> list[float]:
    """
    Extract 9 features from two satellites for one timestep.
    sat_a, sat_b: Satellite objects from core/state.py
    """
    import math
    dx = sat_a.x - sat_b.x
    dy = sat_a.y - sat_b.y
    dz = sat_a.z - sat_b.z
    miss_dist = math.sqrt(dx**2 + dy**2 + dz**2)

    dvx = sat_a.vx - sat_b.vx
    dvy = sat_a.vy - sat_b.vy
    dvz = sat_a.vz - sat_b.vz

    # Simple TCA estimate
    dv2 = dvx**2 + dvy**2 + dvz**2
    if dv2 > 1e-10:
        tca = -(dx * dvx + dy * dvy + dz * dvz) / dv2
        tca = max(0.0, tca)
    else:
        tca = 0.0

    return [
        miss_dist,
        dvx, dvy, dvz,
        tca,
        sat_a.alt_km,
        sat_b.alt_km,
        sat_a.fuel_kg,
        sat_b.fuel_kg,
    ]


# Global predictor (loaded once at startup)
predictor: LSTMPredictor | None = None


def load_predictor(model_path: str | None = None) -> LSTMPredictor:
    global predictor
    predictor = LSTMPredictor(model_path)
    return predictor
