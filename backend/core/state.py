"""
core/state.py
SimulationState — manages 50 satellite objects, debris, and sim clock.
"""

import math
import random
import time
from dataclasses import dataclass, field
from typing import Literal
from sgp4.api import Satrec


SatStatus = Literal["NOMINAL", "MANEUVERING", "CRITICAL", "FUEL_LOW"]


@dataclass
class Satellite:
    sat_id: str
    name: str
    # Orbital elements (Keplerian)
    semi_major_axis: float   # km
    eccentricity: float
    inclination: float       # degrees
    raan: float              # Right Ascension of Ascending Node, degrees
    arg_perigee: float       # degrees
    mean_anomaly: float      # degrees
    # State
    fuel_kg: float = 100.0
    fuel_max: float = 100.0
    status: SatStatus = "NOMINAL"
    # Live position (updated each sim step)
    lat: float = 0.0
    lon: float = 0.0
    alt_km: float = 550.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    # Satrec for SGP4
    satrec: Satrec | None = field(default=None, repr=False)
    is_isro: bool = False

    @property
    def fuel_pct(self) -> float:
        return round(self.fuel / self.fuel_max * 100, 1)

    @property
    def fuel(self) -> float:
        return self.fuel_kg

    def apply_burn(self, delta_v: float) -> float:
        """Apply delta-V burn, deduct fuel, return actual dv applied."""
        # Rough Tsiolkovsky: assume Isp=300s, dry_mass=500kg
        isp = 300.0
        g0 = 0.00981  # km/s²
        dry_mass = 500.0
        wet_mass = dry_mass + self.fuel_kg
        # Max dv with current fuel
        max_dv = isp * g0 * math.log(wet_mass / dry_mass)
        actual_dv = min(delta_v, max_dv)
        # Fuel consumed
        fuel_used = wet_mass * (1 - math.exp(-actual_dv / (isp * g0)))
        self.fuel_kg = max(0.0, self.fuel_kg - fuel_used)

        if self.fuel_kg < 10.0:
            self.status = "FUEL_LOW"
        elif self.fuel_kg < 5.0:
            self.status = "CRITICAL"

        # Apply burn as altitude change (simplified)
        self.alt_km += actual_dv * 1000  # rough: 1 m/s ≈ 1 km altitude
        self.semi_major_axis += actual_dv * 1000
        return actual_dv


@dataclass
class DebrisObject:
    debris_id: str
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    lat: float = 0.0
    lon: float = 0.0
    alt_km: float = 550.0
    is_asteroid: bool = False
    name: str = "DEBRIS"


class SimulationState:
    """
    Central state for all 50 satellites and debris objects.
    Thread-safe via asyncio (single-threaded event loop).
    """

    def __init__(self):
        self.satellites: dict[str, Satellite] = {}
        self.debris: dict[str, DebrisObject] = {}
        self.sim_time: float = time.time()
        self.dt: float = 60.0  # seconds per step
        self.step_count: int = 0
        self.start_time: float = time.time()
        self.asteroid_threat: dict | None = None
        self._seed_satellites()
        self._seed_debris()

    def _seed_satellites(self):
        """Seed 50 synthetic LEO satellites."""
        random.seed(42)
        for i in range(50):
            sat_id = f"SAT-{i+1:03d}"
            # Distribute across multiple orbital planes
            plane = i // 10
            sat_in_plane = i % 10
            self.satellites[sat_id] = Satellite(
                sat_id=sat_id,
                name=f"JANUS-{i+1:03d}",
                semi_major_axis=6928 + random.uniform(-50, 50),   # ~550km LEO
                eccentricity=random.uniform(0.0001, 0.005),
                inclination=53 + plane * 5 + random.uniform(-2, 2),
                raan=plane * 36 + random.uniform(-5, 5),
                arg_perigee=random.uniform(0, 360),
                mean_anomaly=sat_in_plane * 36 + random.uniform(-5, 5),
                fuel_kg=random.uniform(60, 100),
                fuel_max=100.0,
                alt_km=550 + random.uniform(-30, 30),
                lat=random.uniform(-70, 70),
                lon=random.uniform(-180, 180),
            )

    def _seed_debris(self):
        """Seed 20 debris objects."""
        random.seed(123)
        for i in range(20):
            d_id = f"DEB-{i+1:03d}"
            r = 6928 + random.uniform(-100, 200)
            theta = random.uniform(0, 2 * math.pi)
            phi = random.uniform(-math.pi/4, math.pi/4)
            v = math.sqrt(398600.4418 / r)
            self.debris[d_id] = DebrisObject(
                debris_id=d_id,
                x=r * math.cos(theta) * math.cos(phi),
                y=r * math.sin(theta) * math.cos(phi),
                z=r * math.sin(phi),
                vx=-v * math.sin(theta),
                vy=v * math.cos(theta),
                vz=0.0,
                alt_km=r - 6378.137,
                name=f"DEBRIS-{i+1:03d}",
            )

    def advance(self, dt: float | None = None):
        """Advance simulation clock."""
        step = dt or self.dt
        self.sim_time += step
        self.step_count += 1
        # Simple Keplerian position update for each satellite
        for sat in self.satellites.values():
            self._update_sat_position(sat, step)
        for deb in self.debris.values():
            self._update_debris_position(deb, step)

    def _update_sat_position(self, sat: Satellite, dt: float):
        """Simple circular orbit position update."""
        r = sat.semi_major_axis
        mu = 398600.4418
        omega = math.sqrt(mu / r**3)  # rad/s
        sat.mean_anomaly = (sat.mean_anomaly + math.degrees(omega * dt)) % 360
        # Convert to approximate ECI
        inc_rad = math.radians(sat.inclination)
        raan_rad = math.radians(sat.raan)
        ma_rad = math.radians(sat.mean_anomaly)

        # Position in orbital plane
        rx = r * math.cos(ma_rad)
        ry = r * math.sin(ma_rad)

        # Rotate to ECI
        sat.x = rx * (math.cos(raan_rad)) - ry * (math.cos(inc_rad) * math.sin(raan_rad))
        sat.y = rx * (math.sin(raan_rad)) + ry * (math.cos(inc_rad) * math.cos(raan_rad))
        sat.z = ry * math.sin(inc_rad)

        # Velocity (circular)
        v = math.sqrt(mu / r)
        sat.vx = -v * math.sin(ma_rad) * math.cos(raan_rad) - v * math.cos(ma_rad) * math.cos(inc_rad) * math.sin(raan_rad)
        sat.vy = -v * math.sin(ma_rad) * math.sin(raan_rad) + v * math.cos(ma_rad) * math.cos(inc_rad) * math.cos(raan_rad)
        sat.vz = v * math.cos(ma_rad) * math.sin(inc_rad)

        # Update LLA
        from core.physics import eci_to_lla
        sat.lat, sat.lon, sat.alt_km = eci_to_lla(sat.x, sat.y, sat.z)

    def _update_debris_position(self, deb: DebrisObject, dt: float):
        """Simple Euler integration for debris."""
        mu = 398600.4418
        r = math.sqrt(deb.x**2 + deb.y**2 + deb.z**2)
        if r < 1:
            return
        ax = -mu * deb.x / r**3
        ay = -mu * deb.y / r**3
        az = -mu * deb.z / r**3

        deb.vx += ax * dt
        deb.vy += ay * dt
        deb.vz += az * dt
        deb.x += deb.vx * dt
        deb.y += deb.vy * dt
        deb.z += deb.vz * dt
        deb.alt_km = r - 6378.137

        from core.physics import eci_to_lla
        deb.lat, deb.lon, deb.alt_km = eci_to_lla(deb.x, deb.y, deb.z)

    def get_fleet_stats(self) -> dict:
        sats = list(self.satellites.values())
        return {
            "total": len(sats),
            "nominal": sum(1 for s in sats if s.status == "NOMINAL"),
            "maneuvering": sum(1 for s in sats if s.status == "MANEUVERING"),
            "critical": sum(1 for s in sats if s.status in ("CRITICAL", "FUEL_LOW")),
            "debris_count": len(self.debris),
            "avg_fuel_pct": round(sum(s.fuel_pct for s in sats) / len(sats), 1),
            "uptime_seconds": round(time.time() - self.start_time),
            "step_count": self.step_count,
            "sim_time": self.sim_time,
        }

    def reset(self):
        self.satellites.clear()
        self.debris.clear()
        self.step_count = 0
        self.sim_time = time.time()
        self.start_time = time.time()
        self.asteroid_threat = None
        self._seed_satellites()
        self._seed_debris()


# Global singleton
simulation = SimulationState()
