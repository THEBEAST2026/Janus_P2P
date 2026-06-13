"""
routers/threat.py
POST /api/threat/asteroid-alert — seed asteroid threat
GET  /api/threat/corridor       — corridor classification
GET  /api/threat/fleet-response — fleet evasion status
POST /api/threat/override       — ground override command
"""

import time
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from core.state import simulation
from core.cache import cache_get, cache_set, publish
from threat.asteroid import (
    ThreatVector, fetch_jpl_horizons, synthetic_threat,
    compute_corridor, execute_fleet_response
)

router = APIRouter(prefix="/api/threat", tags=["threat"])


class AsteroidAlertRequest(BaseModel):
    use_jpl: bool = False          # fetch real data from JPL Horizons
    object_id: str = "99942"       # Apophis default
    synthetic: bool = True         # use synthetic if JPL fails
    corridor_radius_km: float = 100.0


class OverrideRequest(BaseModel):
    command: str                   # HALT_ALL | RESUME | EVACUATE_CORRIDOR
    target_sat_ids: list[str] = [] # empty = all satellites
    reason: str = ""


@router.post("/asteroid-alert")
async def asteroid_alert(req: AsteroidAlertRequest):
    """
    Seed an asteroid threat into the simulation.
    Optionally fetches real ephemeris from NASA JPL Horizons.
    """
    threat = None

    if req.use_jpl:
        jpl_data = await fetch_jpl_horizons(req.object_id)
        if jpl_data:
            threat = ThreatVector(
                object_id=jpl_data["object_id"],
                name=f"JPL-{req.object_id}",
                x=jpl_data["x"], y=jpl_data["y"], z=jpl_data["z"],
                vx=jpl_data["vx"], vy=jpl_data["vy"], vz=jpl_data["vz"],
                tca_unix=time.time() + 7200,
                corridor_radius_km=req.corridor_radius_km,
                source="jpl_horizons",
            )

    if threat is None:
        threat = synthetic_threat()
        threat.corridor_radius_km = req.corridor_radius_km

    # Store in simulation state
    simulation.asteroid_threat = {
        "object_id": threat.object_id,
        "name": threat.name,
        "x": threat.x, "y": threat.y, "z": threat.z,
        "vx": threat.vx, "vy": threat.vy, "vz": threat.vz,
        "tca_unix": threat.tca_unix,
        "corridor_radius_km": threat.corridor_radius_km,
        "source": threat.source,
        "seeded_at": time.time(),
    }

    # Add as special debris object
    from core.state import DebrisObject
    simulation.debris["ASTEROID-001"] = DebrisObject(
        debris_id="ASTEROID-001",
        x=threat.x, y=threat.y, z=threat.z,
        vx=threat.vx, vy=threat.vy, vz=threat.vz,
        is_asteroid=True,
        name=threat.name,
    )

    # Compute corridor
    response = compute_corridor(threat, simulation.satellites)

    return {
        "status": "threat_seeded",
        "threat": simulation.asteroid_threat,
        "corridor_summary": {
            "in_corridor": len(response.in_corridor),
            "adjacent": len(response.adjacent),
            "safe": len(response.safe),
        }
    }


@router.get("/corridor")
async def corridor_status():
    """Which satellites are in the current threat corridor."""
    if not simulation.asteroid_threat:
        return {"status": "no_active_threat", "in_corridor": [], "adjacent": [], "safe": []}

    t = simulation.asteroid_threat
    threat = ThreatVector(
        object_id=t["object_id"],
        name=t["name"],
        x=t["x"], y=t["y"], z=t["z"],
        vx=t["vx"], vy=t["vy"], vz=t["vz"],
        tca_unix=t["tca_unix"],
        corridor_radius_km=t["corridor_radius_km"],
        source=t["source"],
    )

    response = compute_corridor(threat, simulation.satellites)

    return {
        "status": "active_threat",
        "threat_name": threat.name,
        "tca_unix": threat.tca_unix,
        "corridor_radius_km": threat.corridor_radius_km,
        "in_corridor": response.in_corridor,
        "adjacent": response.adjacent,
        "safe": response.safe,
        "in_corridor_count": len(response.in_corridor),
    }


@router.get("/fleet-response")
async def fleet_response_status():
    """Status of ongoing fleet evasion maneuvers."""
    cached = await cache_get("acm:threat:fleet_response")
    if cached:
        return cached
    return {"status": "no_active_response"}


@router.post("/execute-response")
async def execute_response():
    """Trigger coordinated fleet evasion for current threat."""
    if not simulation.asteroid_threat:
        raise HTTPException(status_code=400, detail="No active asteroid threat")

    from main import agents
    t = simulation.asteroid_threat
    threat = ThreatVector(
        object_id=t["object_id"], name=t["name"],
        x=t["x"], y=t["y"], z=t["z"],
        vx=t["vx"], vy=t["vy"], vz=t["vz"],
        tca_unix=t["tca_unix"],
        corridor_radius_km=t["corridor_radius_km"],
        source=t["source"],
    )

    response = compute_corridor(threat, simulation.satellites)
    result = await execute_fleet_response(response, simulation.satellites, agents)

    return {
        "status": result.status,
        "burns_scheduled": len(result.burns_scheduled),
        "in_corridor_cleared": result.in_corridor,
        "burns": result.burns_scheduled,
    }


@router.post("/override")
async def ground_override(req: OverrideRequest):
    """
    Ground station override command — immediate fleet-wide authority.
    """
    targets = req.target_sat_ids or list(simulation.satellites.keys())

    affected = []
    for sat_id in targets:
        sat = simulation.satellites.get(sat_id)
        if not sat:
            continue

        if req.command == "HALT_ALL":
            sat.status = "NOMINAL"
        elif req.command == "EVACUATE_CORRIDOR":
            sat.status = "MANEUVERING"
        elif req.command == "RESUME":
            sat.status = "NOMINAL"

        affected.append(sat_id)

    # Broadcast override to all agents
    await publish("acm:fleet:emergency", {
        "type": "GROUND_OVERRIDE",
        "command": req.command,
        "targets": targets,
        "reason": req.reason,
        "timestamp": time.time(),
    })

    return {
        "status": "override_executed",
        "command": req.command,
        "affected_satellites": len(affected),
        "reason": req.reason,
        "timestamp": time.time(),
    }
