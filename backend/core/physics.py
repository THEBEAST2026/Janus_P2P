"""
core/physics.py
SGP4 propagation + ECI → LLA coordinate conversion.
"""

import math
import time
from dataclasses import dataclass
from sgp4.api import Satrec, jday


# ─── Constants ────────────────────────────────────────────────────────────────
EARTH_RADIUS_KM = 6378.137
EARTH_FLATTENING = 1 / 298.257223563
MU = 398600.4418  # km³/s²


@dataclass
class StateVector:
    sat_id: str
    x: float   # ECI km
    y: float
    z: float
    vx: float  # ECI km/s
    vy: float
    vz: float
    lat: float
    lon: float
    alt_km: float
    timestamp: float  # unix epoch


def eci_to_lla(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Convert ECI (km) to geodetic lat/lon/alt."""
    # Greenwich Sidereal Time approximation
    t = time.time()
    gst = (t / 86400.0) * 360.9856473  # degrees
    gst_rad = math.radians(gst % 360)

    # Rotate ECI → ECEF
    cos_gst = math.cos(gst_rad)
    sin_gst = math.sin(gst_rad)
    x_ecef = x * cos_gst + y * sin_gst
    y_ecef = -x * sin_gst + y * cos_gst
    z_ecef = z

    # ECEF → geodetic (Bowring iterative)
    lon = math.degrees(math.atan2(y_ecef, x_ecef))
    p = math.sqrt(x_ecef**2 + y_ecef**2)
    e2 = 2 * EARTH_FLATTENING - EARTH_FLATTENING**2
    lat = math.degrees(math.atan2(z_ecef, p * (1 - e2)))

    for _ in range(5):
        lat_rad = math.radians(lat)
        N = EARTH_RADIUS_KM / math.sqrt(1 - e2 * math.sin(lat_rad)**2)
        lat = math.degrees(math.atan2(
            z_ecef + e2 * N * math.sin(lat_rad), p
        ))

    lat_rad = math.radians(lat)
    N = EARTH_RADIUS_KM / math.sqrt(1 - e2 * math.sin(lat_rad)**2)
    alt = p / math.cos(lat_rad) - N if abs(lat) < 89 else abs(z_ecef) / math.sin(lat_rad) - N * (1 - e2)

    return lat, lon, alt


def propagate_satrec(satrec: Satrec, epoch_unix: float) -> StateVector | None:
    """Propagate a sgp4 Satrec object to a given unix timestamp."""
    t = epoch_unix
    jd, fr = jday(
        *time.gmtime(t)[:6]
    )
    e, r, v = satrec.sgp4(jd, fr)
    if e != 0:
        return None

    lat, lon, alt = eci_to_lla(*r)
    return StateVector(
        sat_id=satrec.satnum,
        x=r[0], y=r[1], z=r[2],
        vx=v[0], vy=v[1], vz=v[2],
        lat=lat, lon=lon, alt_km=alt,
        timestamp=t,
    )


def miss_distance(sv_a: StateVector, sv_b: StateVector) -> float:
    """Euclidean distance between two ECI state vectors in km."""
    return math.sqrt(
        (sv_a.x - sv_b.x)**2 +
        (sv_a.y - sv_b.y)**2 +
        (sv_a.z - sv_b.z)**2
    )


def relative_velocity(sv_a: StateVector, sv_b: StateVector) -> tuple[float, float, float]:
    """Relative velocity vector (km/s) A relative to B."""
    return (sv_a.vx - sv_b.vx, sv_a.vy - sv_b.vy, sv_a.vz - sv_b.vz)


def time_to_closest_approach(sv_a: StateVector, sv_b: StateVector) -> float:
    """Simple linear TCA estimate in seconds."""
    dx = sv_a.x - sv_b.x
    dy = sv_a.y - sv_b.y
    dz = sv_a.z - sv_b.z
    dvx, dvy, dvz = relative_velocity(sv_a, sv_b)

    dv2 = dvx**2 + dvy**2 + dvz**2
    if dv2 < 1e-10:
        return 0.0

    tca = -(dx * dvx + dy * dvy + dz * dvz) / dv2
    return max(0.0, tca)


def delta_v_for_miss_distance(
    target_miss_km: float,
    current_miss_km: float,
    altitude_km: float
) -> float:
    """
    Estimate delta-V (km/s) needed to achieve target miss distance.
    Simple Hohmann-style approximation for LEO.
    """
    if current_miss_km >= target_miss_km:
        return 0.0
    gap = target_miss_km - current_miss_km
    r = EARTH_RADIUS_KM + altitude_km
    v_circ = math.sqrt(MU / r)
    # Approximate: small dv for small altitude change
    dv = (gap / r) * v_circ * 0.5
    return round(dv, 6)
