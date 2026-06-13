"""
routers/simulate.py
POST /api/simulate/step   — advance simulation
GET  /api/simulate/status — fleet aggregates
GET  /api/simulate/cdm    — conjunction assessments
POST /api/simulate/reset  — reset to initial state
"""

import math
import time
from fastapi import APIRouter
from pydantic import BaseModel

from core.state import simulation
from core.cache import cache_set, cache_get
from p2p.gossip import gossip_round, compute_cdm_warnings

router = APIRouter(prefix="/api/simulate", tags=["simulate"])


class StepRequest(BaseModel):
    dt: float = 60.0  # seconds


@router.post("/step")
async def simulate_step(req: StepRequest):
    """Advance simulation by dt seconds. Runs gossip + CDM check."""
    simulation.advance(req.dt)

    # Compute CDM warnings
    cdm_warnings = compute_cdm_warnings(
        simulation.satellites,
        simulation.debris,
        threshold_km=10.0
    )

    # Run gossip round
    network_state = await gossip_round(simulation.satellites, cdm_warnings)

    # Run LSTM + negotiation for close pairs
    from core.state import simulation as sim
    negotiations = []

    # Import agents registry
    from main import agents
    pairs_checked = 0
    for sat_id, agent in agents.items():
        sat = sim.satellites[sat_id]
        # Check against 5 nearest satellites
        from p2p.gossip import find_neighbors
        neighbors = find_neighbors(sat_id, sim.satellites, k=5)
        for nb_id in neighbors:
            nb_sat = sim.satellites[nb_id]
            record = await agent.evaluate_and_negotiate(sat, nb_sat, agents)
            if record:
                negotiations.append(record.record_id)
            pairs_checked += 1

    stats = simulation.get_fleet_stats()
    stats["cdm_warnings"] = sum(len(v) for v in cdm_warnings.values())
    stats["negotiations_this_step"] = len(negotiations)
    stats["pairs_checked"] = pairs_checked

    return {
        "status": "ok",
        "step": simulation.step_count,
        "dt": req.dt,
        "fleet": stats,
        "network_edges": len(network_state["edges"]),
        "negotiations": negotiations,
    }


@router.get("/status")
async def fleet_status():
    """Fleet aggregates with CDM warning count."""
    cached = await cache_get("acm:fleet:status")
    if cached:
        return cached

    cdm = compute_cdm_warnings(simulation.satellites, simulation.debris)
    stats = simulation.get_fleet_stats()
    stats["cdm_warnings"] = sum(len(v) for v in cdm.values())

    await cache_set("acm:fleet:status", stats, ttl=10)
    return stats


@router.get("/cdm")
async def conjunction_data_messages():
    """Return all active conjunction warnings across the fleet."""
    cdm = compute_cdm_warnings(simulation.satellites, simulation.debris, threshold_km=20.0)

    results = []
    for sat_id, warnings in cdm.items():
        for w in warnings:
            results.append({
                "sat_id": sat_id,
                "debris_id": w["debris_id"],
                "distance_km": w["distance_km"],
                "severity": w["severity"],
            })

    results.sort(key=lambda x: x["distance_km"])
    return {"count": len(results), "conjunctions": results}


@router.post("/reset")
async def reset_simulation():
    """Reset simulation to initial state."""
    simulation.reset()
    # Reinitialize agents
    from main import init_agents
    await init_agents()
    return {"status": "reset", "message": "Simulation reset to initial state"}
