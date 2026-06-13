"""
routers/p2p.py
GET /api/p2p/network-state    — ISL mesh graph for Three.js
GET /api/p2p/negotiation-log  — all autonomous maneuvers
GET /api/p2p/lstm-accuracy    — live model metrics
"""

import time
from fastapi import APIRouter
from core.cache import cache_get, cache_lrange
from core.state import simulation

router = APIRouter(prefix="/api/p2p", tags=["p2p"])


@router.get("/network-state")
async def network_state():
    """
    Returns ISL mesh graph for Three.js line overlay.
    Cached for 5s.
    """
    cached = await cache_get("acm:p2p:network")
    if cached:
        return cached

    # Build fresh if cache miss
    from p2p.gossip import gossip_round, compute_cdm_warnings
    cdm = compute_cdm_warnings(simulation.satellites, simulation.debris)
    state = await gossip_round(simulation.satellites, cdm)
    return state


@router.get("/negotiation-log")
async def negotiation_log(limit: int = 50):
    """
    Returns last N autonomous negotiation records.
    Includes LSTM risk scores, burns applied, fuel remaining.
    """
    records = await cache_lrange("acm:p2p:negotiation_log", 0, limit - 1)

    # Also pull from in-memory agents
    from main import agents
    in_memory = []
    for agent in agents.values():
        for rec in agent._negotiation_log[-10:]:
            from p2p.satellite_agent import _record_to_dict
            in_memory.append(_record_to_dict(rec))

    # Merge and deduplicate by record_id
    all_records = {r["record_id"]: r for r in records}
    for r in in_memory:
        all_records[r["record_id"]] = r

    sorted_records = sorted(
        all_records.values(),
        key=lambda x: x["timestamp"],
        reverse=True
    )[:limit]

    return {
        "count": len(sorted_records),
        "records": sorted_records,
        "timestamp": time.time(),
    }


@router.get("/lstm-accuracy")
async def lstm_accuracy():
    """
    Live model precision/recall metrics based on recent negotiations.
    """
    records = await cache_lrange("acm:p2p:negotiation_log", 0, 99)

    if not records:
        return {
            "total_predictions": 0,
            "high_risk_triggered": 0,
            "burns_executed": 0,
            "avg_risk_score": 0.0,
            "avg_miss_dist_km": 0.0,
            "evader_distribution": {"A": 0, "B": 0, "SPLIT": 0},
        }

    high_risk = [r for r in records if r.get("lstm_risk", 0) >= 0.75]
    executed = [r for r in records if r.get("status") == "EXECUTED"]
    escalated = [r for r in records if r.get("ground_escalated")]

    avg_risk = sum(r.get("lstm_risk", 0) for r in records) / len(records)
    avg_miss = sum(r.get("lstm_miss_dist_km", 0) for r in records) / len(records)

    evader_dist = {"A": 0, "B": 0, "SPLIT": 0}
    for r in records:
        evader = r.get("who_evades", "A")
        evader_dist[evader] = evader_dist.get(evader, 0) + 1

    return {
        "total_predictions": len(records),
        "high_risk_triggered": len(high_risk),
        "burns_executed": len(executed),
        "ground_escalated": len(escalated),
        "avg_risk_score": round(avg_risk, 4),
        "avg_miss_dist_km": round(avg_miss, 3),
        "evader_distribution": evader_dist,
        "autonomous_rate": round(len(executed) / max(len(high_risk), 1) * 100, 1),
    }
