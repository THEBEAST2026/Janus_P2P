"""
p2p/satellite_agent.py
SatelliteAgent — handles LSTM inference, negotiation protocol, and burn execution.
One agent per satellite. LSTM model is injected at startup.
"""

import math
import time
import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Literal

from core.cache import cache_set, cache_get, cache_lpush, publish
from ml.conjunction_lstm import LSTMPredictor, extract_features, ConjunctionPrediction

logger = logging.getLogger("janus.agent")

NegotiationStatus = Literal["PENDING", "ACCEPTED", "REJECTED", "COUNTER_PROPOSED", "EXECUTED"]

RISK_THRESHOLD = 0.75         # trigger negotiation
ESCALATION_THRESHOLD = 0.95   # escalate to ground
MAX_BURN = 0.015              # km/s max autonomous burn
WINDOW_SIZE = 20


@dataclass
class NegotiationRecord:
    record_id: str
    timestamp: float
    sat_a_id: str
    sat_b_id: str
    lstm_risk: float
    lstm_miss_dist: float
    lstm_tca_secs: float
    who_evades: str
    proposed_dv_a: float
    proposed_dv_b: float
    status: NegotiationStatus
    actual_dv_a: float = 0.0
    actual_dv_b: float = 0.0
    fuel_a_after: float = 0.0
    fuel_b_after: float = 0.0
    new_miss_dist: float = 0.0
    ground_escalated: bool = False
    ground_approved: bool = False


class SatelliteAgent:
    """
    Autonomous agent for one satellite.
    Manages LSTM feature buffer, negotiation, and burn execution.
    """

    def __init__(self, sat_id: str, predictor: LSTMPredictor):
        self.sat_id = sat_id
        self.predictor = predictor
        # Rolling feature deques per partner satellite
        self._buffers: dict[str, deque] = {}
        self._active_negotiations: set[str] = set()
        self._negotiation_log: list[NegotiationRecord] = []

    def _get_buffer(self, partner_id: str) -> deque:
        if partner_id not in self._buffers:
            self._buffers[partner_id] = deque(maxlen=WINDOW_SIZE)
        return self._buffers[partner_id]

    def update_buffer(self, partner_sat, my_sat) -> None:
        """Add current timestep features to the buffer for this pair."""
        features = extract_features(my_sat, partner_sat)
        buf = self._get_buffer(partner_sat.sat_id)
        buf.append(features)

    def get_prediction(self, partner_id: str) -> ConjunctionPrediction | None:
        buf = self._get_buffer(partner_id)
        if len(buf) < 5:  # need at least 5 steps
            return None
        return self.predictor.predict(list(buf))

    async def evaluate_and_negotiate(
        self,
        my_sat,
        partner_sat,
        all_agents: dict[str, "SatelliteAgent"],
    ) -> NegotiationRecord | None:
        """
        Main agent loop step:
        1. Update feature buffer
        2. Run LSTM
        3. If risk > threshold → initiate negotiation
        4. Execute burns
        """
        self.update_buffer(partner_sat, my_sat)
        pred = self.get_prediction(partner_sat.sat_id)
        if pred is None:
            return None

        pair_key = f"{min(self.sat_id, partner_sat.sat_id)}:{max(self.sat_id, partner_sat.sat_id)}"

        # Avoid duplicate negotiation for same pair
        if pair_key in self._active_negotiations:
            return None

        if pred.risk_prob < RISK_THRESHOLD:
            return None

        # Escalate to ground if very high risk
        escalate = pred.risk_prob >= ESCALATION_THRESHOLD

        logger.info(
            f"[{self.sat_id}] Risk {pred.risk_prob:.2f} vs {partner_sat.sat_id} "
            f"— initiating negotiation (escalate={escalate})"
        )

        self._active_negotiations.add(pair_key)

        record = await self._negotiate(
            my_sat, partner_sat, pred, escalate, all_agents
        )

        self._active_negotiations.discard(pair_key)
        if record:
            self._negotiation_log.append(record)
            # Persist to cache
            await cache_lpush("acm:p2p:negotiation_log", _record_to_dict(record), maxlen=500)
            # Notify ground
            await publish("acm:updates", {
                "type": "NEGOTIATION_COMPLETE",
                "record": _record_to_dict(record)
            })

        return record

    async def _negotiate(
        self,
        my_sat,
        partner_sat,
        pred: ConjunctionPrediction,
        escalate: bool,
        all_agents: dict[str, "SatelliteAgent"],
    ) -> NegotiationRecord:
        record_id = f"NEG-{self.sat_id}-{partner_sat.sat_id}-{int(time.time())}"

        # Compute proposed burns based on who_evades
        if pred.who_evades == "A":
            dv_a = min(pred.miss_dist_km * 0.001 + 0.002, MAX_BURN)
            dv_b = 0.0
        elif pred.who_evades == "B":
            dv_a = 0.0
            dv_b = min(pred.miss_dist_km * 0.001 + 0.002, MAX_BURN)
        else:  # SPLIT
            half = min(pred.miss_dist_km * 0.0005 + 0.001, MAX_BURN / 2)
            dv_a = half
            dv_b = half

        # Publish negotiation proposal via Redis pub/sub
        proposal = {
            "type": "CONJUNCTION_NEGOTIATION",
            "proposer": self.sat_id,
            "acceptor": partner_sat.sat_id,
            "record_id": record_id,
            "lstm_risk": pred.risk_prob,
            "proposed_dv_a": dv_a,
            "proposed_dv_b": dv_b,
            "who_evades": pred.who_evades,
        }
        await publish(f"acm:isl:{partner_sat.sat_id}", proposal)

        # Check if partner can execute the burn
        partner_agent = all_agents.get(partner_sat.sat_id)
        accepted, actual_dv_a, actual_dv_b = self._evaluate_response(
            my_sat, partner_sat, dv_a, dv_b
        )

        if not accepted:
            # Counter-propose: I take full burn
            actual_dv_a = min(dv_a + dv_b, MAX_BURN)
            actual_dv_b = 0.0
            status = "COUNTER_PROPOSED"
        else:
            status = "ACCEPTED"

        # Execute burns (unless ground escalation pending)
        if escalate:
            status = "PENDING"
            my_sat.status = "MANEUVERING"
            # Ground must approve before burn
            return NegotiationRecord(
                record_id=record_id,
                timestamp=time.time(),
                sat_a_id=self.sat_id,
                sat_b_id=partner_sat.sat_id,
                lstm_risk=pred.risk_prob,
                lstm_miss_dist=pred.miss_dist_km,
                lstm_tca_secs=pred.tca_secs,
                who_evades=pred.who_evades,
                proposed_dv_a=dv_a,
                proposed_dv_b=dv_b,
                status=status,
                ground_escalated=True,
            )

        # Execute burns
        actual_a = my_sat.apply_burn(actual_dv_a) if actual_dv_a > 0 else 0.0
        actual_b = partner_sat.apply_burn(actual_dv_b) if actual_dv_b > 0 else 0.0
        my_sat.status = "NOMINAL" if my_sat.fuel_pct > 20 else "FUEL_LOW"
        partner_sat.status = "NOMINAL" if partner_sat.fuel_pct > 20 else "FUEL_LOW"

        # New miss distance (simplified: delta-V → altitude change → more separation)
        new_miss = pred.miss_dist_km + (actual_a + actual_b) * 500

        return NegotiationRecord(
            record_id=record_id,
            timestamp=time.time(),
            sat_a_id=self.sat_id,
            sat_b_id=partner_sat.sat_id,
            lstm_risk=pred.risk_prob,
            lstm_miss_dist=pred.miss_dist_km,
            lstm_tca_secs=pred.tca_secs,
            who_evades=pred.who_evades,
            proposed_dv_a=dv_a,
            proposed_dv_b=dv_b,
            status="EXECUTED",
            actual_dv_a=round(actual_a, 6),
            actual_dv_b=round(actual_b, 6),
            fuel_a_after=round(my_sat.fuel_kg, 2),
            fuel_b_after=round(partner_sat.fuel_kg, 2),
            new_miss_dist=round(new_miss, 3),
            ground_escalated=escalate,
        )

    def _evaluate_response(
        self, my_sat, partner_sat, dv_a: float, dv_b: float
    ) -> tuple[bool, float, float]:
        """Check if partner has enough fuel. Returns (accepted, actual_dv_a, actual_dv_b)."""
        # Partner accepts if it has enough fuel for its burn
        if dv_b > 0 and partner_sat.fuel_kg < 5.0:
            return False, dv_a, dv_b
        return True, dv_a, dv_b


def _record_to_dict(r: NegotiationRecord) -> dict:
    return {
        "record_id": r.record_id,
        "timestamp": r.timestamp,
        "sat_a_id": r.sat_a_id,
        "sat_b_id": r.sat_b_id,
        "lstm_risk": r.lstm_risk,
        "lstm_miss_dist_km": r.lstm_miss_dist,
        "lstm_tca_secs": r.lstm_tca_secs,
        "who_evades": r.who_evades,
        "proposed_dv_a": r.proposed_dv_a,
        "proposed_dv_b": r.proposed_dv_b,
        "status": r.status,
        "actual_dv_a": r.actual_dv_a,
        "actual_dv_b": r.actual_dv_b,
        "fuel_a_after_kg": r.fuel_a_after,
        "fuel_b_after_kg": r.fuel_b_after,
        "new_miss_dist_km": r.new_miss_dist,
        "ground_escalated": r.ground_escalated,
        "ground_approved": r.ground_approved,
    }
