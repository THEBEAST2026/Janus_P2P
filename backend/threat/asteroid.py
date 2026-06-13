"""
threat/asteroid.py
NASA JPL Horizons API integration for asteroid ephemeris.
Fleet-wide threat corridor computation and coordinated response.
"""

import math
import time
import httpx
import logging
from dataclasses import dataclass, field

from core.cache import cache_set, cache_get, publish

logger = logging.getLogger("janus.threat")

JPL_HORIZONS_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"

# Apophis 2029 object ID
APOPHIS_ID = "99942"


@dataclass
class ThreatVector:
    object_id: str
    name: str
    x: float        # ECI km
    y: float
    z: float
    vx: float       # km/s
    vy: float
    vz: float
    tca_unix: float # Unix timestamp of closest approach
    corridor_radius_km: float = 50.0
    source: str = "manual"


@dataclass
class FleetThreatResponse:
    threat: ThreatVector
    in_corridor: list[str] = field(default_factory=list)
    adjacent: list[str] = field(default_factory=list)
    safe: list[str] = field(default_factory=list)
    burns_scheduled: list[dict] = field(default_factory=list)
    status: str = "COMPUTED"  # COMPUTED | EXECUTING | COMPLETE


async def fetch_jpl_horizons(object_id: str = APOPHIS_ID) -> dict | None:
    """
    Fetch asteroid state vector from NASA JPL Horizons REST API.
    No credentials needed. Caches for 1 hour.
    """
    cache_key = f"acm:jpl:{object_id}"
    cached = await cache_get(cache_key)
    if cached:
        return cached

    params = {
        "format": "json",
        "COMMAND": f"'{object_id}'",
        "OBJ_DATA": "NO",
        "MAKE_EPHEM": "YES",
        "EPHEM_TYPE": "VECTORS",
        "CENTER": "500@399",   # Geocenter
        "STEP_SIZE": "1d",
        "QUANTITIES": "2",     # State vectors
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(JPL_HORIZONS_URL, params=params)
            data = resp.json()
            result = _parse_horizons_response(data, object_id)
            if result:
                await cache_set(cache_key, result, ttl=3600)
            return result
    except Exception as e:
        logger.warning(f"JPL Horizons fetch failed: {e} — using synthetic threat")
        return None


def _parse_horizons_response(data: dict, object_id: str) -> dict | None:
    """Parse JPL Horizons JSON response to extract state vector."""
    try:
        result_text = data.get("result", "")
        # Find $$SOE ... $$EOE block
        soe = result_text.find("$$SOE")
        eoe = result_text.find("$$EOE")
        if soe == -1 or eoe == -1:
            return None

        block = result_text[soe + 5:eoe].strip()
        lines = [l.strip() for l in block.split("\n") if l.strip()]

        if len(lines) < 3:
            return None

        # Line 2: X Y Z, Line 3: VX VY VZ
        parts2 = lines[1].split()
        parts3 = lines[2].split()

        x = float(parts2[2])  # km
        y = float(parts2[4])
        z = float(parts2[6])
        vx = float(parts3[1])  # km/s
        vy = float(parts3[3])
        vz = float(parts3[5])

        return {
            "object_id": object_id,
            "x": x, "y": y, "z": z,
            "vx": vx, "vy": vy, "vz": vz,
            "fetched_at": time.time(),
        }
    except Exception as e:
        logger.warning(f"Horizons parse error: {e}")
        return None


def synthetic_threat(name: str = "SYNTHETIC-APOPHIS") -> ThreatVector:
    """Generate a synthetic asteroid threat for demo/testing."""
    import random
    rng = random.Random(42)
    r = 7500 + rng.uniform(-500, 500)
    return ThreatVector(
        object_id="SYNTH-001",
        name=name,
        x=r * 0.6,
        y=r * 0.8,
        z=r * 0.1,
        vx=-1.2,
        vy=0.8,
        vz=0.3,
        tca_unix=time.time() + 3600,
        corridor_radius_km=100.0,
        source="synthetic",
    )


def compute_corridor(threat: ThreatVector, satellites: dict) -> FleetThreatResponse:
    """
    Classify each satellite as IN_CORRIDOR, ADJACENT, or SAFE.
    Corridor = cylinder around asteroid trajectory.
    """
    response = FleetThreatResponse(threat=threat)

    # Threat trajectory direction unit vector
    v_mag = math.sqrt(threat.vx**2 + threat.vy**2 + threat.vz**2)
    if v_mag < 1e-10:
        return response

    uv = (threat.vx / v_mag, threat.vy / v_mag, threat.vz / v_mag)

    for sat_id, sat in satellites.items():
        # Vector from threat position to satellite
        dx = sat.x - threat.x
        dy = sat.y - threat.y
        dz = sat.z - threat.z

        # Project onto threat velocity direction
        proj = dx * uv[0] + dy * uv[1] + dz * uv[2]

        # Perpendicular distance to corridor centerline
        perp_x = dx - proj * uv[0]
        perp_y = dy - proj * uv[1]
        perp_z = dz - proj * uv[2]
        perp_dist = math.sqrt(perp_x**2 + perp_y**2 + perp_z**2)

        if perp_dist < threat.corridor_radius_km:
            response.in_corridor.append(sat_id)
        elif perp_dist < threat.corridor_radius_km * 2:
            response.adjacent.append(sat_id)
        else:
            response.safe.append(sat_id)

    return response


async def execute_fleet_response(
    response: FleetThreatResponse,
    satellites: dict,
    agents: dict,
) -> FleetThreatResponse:
    """
    Coordinate fleet maneuvers to clear the threat corridor.
    Phase burns to avoid creating new conjunctions.
    """
    logger.info(
        f"Fleet response: {len(response.in_corridor)} in corridor, "
        f"{len(response.adjacent)} adjacent, {len(response.safe)} safe"
    )

    # Broadcast fleet alert
    await publish("acm:fleet:emergency", {
        "type": "ASTEROID_ALERT",
        "threat_id": response.threat.object_id,
        "threat_name": response.threat.name,
        "tca_unix": response.threat.tca_unix,
        "corridor_radius_km": response.threat.corridor_radius_km,
        "in_corridor": response.in_corridor,
        "adjacent": response.adjacent,
    })

    # Phase burns — satellites maneuver in waves to avoid mutual conjunctions
    wave_size = 5
    for wave_idx, i in enumerate(range(0, len(response.in_corridor), wave_size)):
        wave = response.in_corridor[i:i + wave_size]
        for sat_id in wave:
            sat = satellites.get(sat_id)
            if not sat:
                continue

            # Determine altitude change direction (up if below threat, down if above)
            dv = min(0.005 + wave_idx * 0.001, 0.015)
            actual_dv = sat.apply_burn(dv)
            sat.status = "MANEUVERING"

            burn_record = {
                "sat_id": sat_id,
                "wave": wave_idx + 1,
                "dv_km_s": round(actual_dv, 6),
                "fuel_after_kg": round(sat.fuel_kg, 2),
                "timestamp": time.time(),
            }
            response.burns_scheduled.append(burn_record)

    response.status = "EXECUTING"

    # Cache response for frontend
    await cache_set("acm:threat:fleet_response", {
        "status": response.status,
        "threat_name": response.threat.name,
        "in_corridor": response.in_corridor,
        "adjacent": response.adjacent,
        "safe": response.safe,
        "burns_scheduled": response.burns_scheduled,
        "timestamp": time.time(),
    }, ttl=300)

    return response
