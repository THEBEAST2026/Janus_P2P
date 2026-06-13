"""
routers/maneuver.py
POST /api/maneuver/burn       — schedule manual delta-V burn
POST /api/maneuver/auto-evade — ground-commanded auto evasion
GET  /api/maneuver/timeline   — maneuver schedule
"""

import time
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from core.state import simulation
from core.cache import cache_lpush, cache_lrange

router = APIRouter(prefix="/api/maneuver", tags=["maneuver"])

_scheduled_burns: list[dict] = []


class BurnRequest(BaseModel):
    sat_id: str
    delta_v_km_s: float
    scheduled_at: float | None = None
    reason: str = "manual"


class AutoEvadeRequest(BaseModel):
    sat_id: str
    debris_id: str
    target_miss_km: float = 5.0


@router.post("/burn")
async def schedule_burn(req: BurnRequest):
    sat = simulation.satellites.get(req.sat_id)
    if not sat:
        raise HTTPException(404, f"Satellite {req.sat_id} not found")

    actual_dv = sat.apply_burn(req.delta_v_km_s)
    record = {
        "sat_id": req.sat_id,
        "requested_dv": req.delta_v_km_s,
        "actual_dv": round(actual_dv, 6),
        "fuel_after_kg": round(sat.fuel_kg, 2),
        "reason": req.reason,
        "executed_at": time.time(),
        "source": "ground_manual",
    }
    await cache_lpush("acm:maneuver:timeline", record, maxlen=500)
    _scheduled_burns.append(record)
    return {"status": "executed", **record}


@router.post("/auto-evade")
async def auto_evade(req: AutoEvadeRequest):
    sat = simulation.satellites.get(req.sat_id)
    if not sat:
        raise HTTPException(404, f"Satellite {req.sat_id} not found")

    deb = simulation.debris.get(req.debris_id)
    if not deb:
        raise HTTPException(404, f"Debris {req.debris_id} not found")

    import math
    dist = math.sqrt((sat.x-deb.x)**2 + (sat.y-deb.y)**2 + (sat.z-deb.z)**2)
    from core.physics import delta_v_for_miss_distance
    dv = delta_v_for_miss_distance(req.target_miss_km, dist, sat.alt_km)

    if dv <= 0:
        return {"status": "no_burn_needed", "current_miss_km": round(dist, 3)}

    actual_dv = sat.apply_burn(dv)
    sat.status = "NOMINAL"

    record = {
        "sat_id": req.sat_id,
        "debris_id": req.debris_id,
        "pre_miss_km": round(dist, 3),
        "target_miss_km": req.target_miss_km,
        "delta_v": round(actual_dv, 6),
        "fuel_after_kg": round(sat.fuel_kg, 2),
        "executed_at": time.time(),
        "source": "ground_auto_evade",
    }
    await cache_lpush("acm:maneuver:timeline", record, maxlen=500)
    return {"status": "executed", **record}


@router.get("/timeline")
async def maneuver_timeline(limit: int = 50):
    cached = await cache_lrange("acm:maneuver:timeline", 0, limit - 1)
    all_burns = {r.get("executed_at", 0): r for r in cached}
    for b in _scheduled_burns[-limit:]:
        all_burns[b.get("executed_at", 0)] = b
    sorted_burns = sorted(all_burns.values(), key=lambda x: x.get("executed_at", 0), reverse=True)
    return {"count": len(sorted_burns), "burns": sorted_burns[:limit]}
