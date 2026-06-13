"""
routers/telemetry.py
GET  /api/telemetry         — current telemetry for all satellites
POST /api/telemetry         — inject telemetry packet
GET  /api/telemetry/history — last 1000 packets
"""

import time
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from core.state import simulation
from core.cache import cache_lpush, cache_lrange

router = APIRouter(prefix="/api/telemetry", tags=["telemetry"])


class TelemetryPacket(BaseModel):
    sat_id: str
    fuel_kg: float | None = None
    status: str | None = None
    lat: float | None = None
    lon: float | None = None
    alt_km: float | None = None


@router.get("")
async def get_telemetry():
    sats = []
    for sat in simulation.satellites.values():
        sats.append({
            "sat_id": sat.sat_id,
            "name": sat.name,
            "lat": round(sat.lat, 4),
            "lon": round(sat.lon, 4),
            "alt_km": round(sat.alt_km, 2),
            "fuel_pct": sat.fuel_pct,
            "fuel_kg": round(sat.fuel_kg, 2),
            "status": sat.status,
            "timestamp": time.time(),
        })
    return {"count": len(sats), "satellites": sats}


@router.post("")
async def inject_telemetry(packet: TelemetryPacket):
    sat = simulation.satellites.get(packet.sat_id)
    if not sat:
        raise HTTPException(404, f"Satellite {packet.sat_id} not found")

    if packet.fuel_kg is not None:
        sat.fuel_kg = packet.fuel_kg
    if packet.status is not None:
        sat.status = packet.status
    if packet.lat is not None:
        sat.lat = packet.lat
    if packet.lon is not None:
        sat.lon = packet.lon
    if packet.alt_km is not None:
        sat.alt_km = packet.alt_km

    record = {
        "sat_id": packet.sat_id,
        "injected": packet.dict(exclude_none=True),
        "timestamp": time.time(),
    }
    await cache_lpush("acm:telemetry:history", record, maxlen=1000)
    return {"status": "injected", "sat_id": packet.sat_id}


@router.get("/history")
async def telemetry_history(limit: int = 100):
    records = await cache_lrange("acm:telemetry:history", 0, limit - 1)
    return {"count": len(records), "history": records}
