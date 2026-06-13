"""
routers/visualization.py
GET /api/visualization/snapshot     — all satellite positions (cached 30s)
GET /api/visualization/fuel-heatmap — fuel levels for heatmap
GET /api/visualization/trajectory/:id — orbital trajectory for one satellite
"""

import math
import time
from fastapi import APIRouter, HTTPException
from core.state import simulation
from core.cache import cache_get, cache_set

router = APIRouter(prefix="/api/visualization", tags=["visualization"])


@router.get("/snapshot")
async def snapshot():
    """
    Returns all satellite + debris positions for Three.js globe.
    Redis cached for 30s.
    """
    cached = await cache_get("acm:snapshot")
    if cached:
        return cached

    satellites = []
    for sat in simulation.satellites.values():
        satellites.append({
            "sat_id": sat.sat_id,
            "name": sat.name,
            "lat": round(sat.lat, 4),
            "lon": round(sat.lon, 4),
            "alt_km": round(sat.alt_km, 2),
            "x": round(sat.x, 3),
            "y": round(sat.y, 3),
            "z": round(sat.z, 3),
            "fuel_pct": sat.fuel_pct,
            "status": sat.status,
            "is_isro": sat.is_isro,
        })

    debris_list = []
    for deb in simulation.debris.values():
        debris_list.append({
            "debris_id": deb.debris_id,
            "name": deb.name,
            "lat": round(deb.lat, 4),
            "lon": round(deb.lon, 4),
            "alt_km": round(deb.alt_km, 2),
            "x": round(deb.x, 3),
            "y": round(deb.y, 3),
            "z": round(deb.z, 3),
            "is_asteroid": deb.is_asteroid,
        })

    result = {
        "timestamp": time.time(),
        "step": simulation.step_count,
        "satellites": satellites,
        "debris": debris_list,
        "total_objects": len(satellites) + len(debris_list),
    }

    await cache_set("acm:snapshot", result, ttl=30)
    return result


@router.get("/fuel-heatmap")
async def fuel_heatmap():
    """Fuel level data for heatmap visualization."""
    data = []
    for sat in simulation.satellites.values():
        data.append({
            "sat_id": sat.sat_id,
            "lat": round(sat.lat, 4),
            "lon": round(sat.lon, 4),
            "fuel_pct": sat.fuel_pct,
            "fuel_kg": round(sat.fuel_kg, 2),
            "status": sat.status,
            "intensity": 1.0 - sat.fuel_pct / 100.0,  # high intensity = low fuel
        })
    data.sort(key=lambda x: x["fuel_pct"])
    return {"satellites": data, "timestamp": time.time()}


@router.get("/trajectory/{sat_id}")
async def satellite_trajectory(sat_id: str, steps: int = 60):
    """
    Generate future orbital trajectory for one satellite.
    Returns list of lat/lon/alt points over next N steps.
    """
    sat = simulation.satellites.get(sat_id)
    if not sat:
        raise HTTPException(status_code=404, detail=f"Satellite {sat_id} not found")

    # Simulate future positions using simple Keplerian propagation
    import copy
    from core.state import Satellite

    # Create a temporary copy for forward propagation
    temp_sat = copy.copy(sat)
    points = []

    mu = 398600.4418
    r = temp_sat.semi_major_axis
    omega = math.sqrt(mu / r**3)
    dt = simulation.dt

    for i in range(steps):
        temp_sat.mean_anomaly = (temp_sat.mean_anomaly + math.degrees(omega * dt)) % 360
        inc_rad = math.radians(temp_sat.inclination)
        raan_rad = math.radians(temp_sat.raan)
        ma_rad = math.radians(temp_sat.mean_anomaly)

        rx = r * math.cos(ma_rad)
        ry = r * math.sin(ma_rad)
        x = rx * math.cos(raan_rad) - ry * math.cos(inc_rad) * math.sin(raan_rad)
        y = rx * math.sin(raan_rad) + ry * math.cos(inc_rad) * math.cos(raan_rad)
        z = ry * math.sin(inc_rad)

        from core.physics import eci_to_lla
        lat, lon, alt = eci_to_lla(x, y, z)
        points.append({
            "step": i + 1,
            "lat": round(lat, 4),
            "lon": round(lon, 4),
            "alt_km": round(alt, 2),
        })

    return {
        "sat_id": sat_id,
        "name": sat.name,
        "current": {"lat": sat.lat, "lon": sat.lon, "alt_km": sat.alt_km},
        "trajectory": points,
        "dt_seconds": dt,
    }
