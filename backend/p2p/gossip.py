"""
p2p/gossip.py
Gossip protocol — each satellite shares state with k=3 random neighbors.
Redis pub/sub = simulated ISL (Inter-Satellite Link).
"""

import math
import random
import time
import asyncio
import logging
from dataclasses import dataclass

from core.cache import cache_set, cache_get, publish

logger = logging.getLogger("janus.gossip")

ISL_RANGE_KM = 2000.0   # max ISL communication range
GOSSIP_K = 3             # neighbors per gossip round


@dataclass
class GossipMessage:
    sender_id: str
    timestamp: float
    x: float
    y: float
    z: float
    fuel_pct: float
    status: str
    cdm_warnings: list[str]


def isl_distance(sat_a, sat_b) -> float:
    """Euclidean ECI distance between two satellites."""
    return math.sqrt(
        (sat_a.x - sat_b.x)**2 +
        (sat_a.y - sat_b.y)**2 +
        (sat_a.z - sat_b.z)**2
    )


def find_neighbors(sat_id: str, satellites: dict, k: int = GOSSIP_K) -> list[str]:
    """
    Find k nearest neighbors within ISL range.
    Falls back to k random satellites if fewer in range.
    """
    my_sat = satellites.get(sat_id)
    if not my_sat:
        return []

    candidates = []
    for other_id, other in satellites.items():
        if other_id == sat_id:
            continue
        dist = isl_distance(my_sat, other)
        if dist <= ISL_RANGE_KM:
            candidates.append((dist, other_id))

    # Sort by distance, take k nearest
    candidates.sort(key=lambda x: x[0])
    neighbors = [c[1] for c in candidates[:k]]

    # If not enough in range, add random ones
    if len(neighbors) < k:
        remaining = [s for s in satellites if s != sat_id and s not in neighbors]
        random.shuffle(remaining)
        neighbors += remaining[:k - len(neighbors)]

    return neighbors


async def gossip_round(satellites: dict, cdm_warnings: dict[str, list]) -> dict:
    """
    Run one gossip round across all satellites.
    Returns network state: {edges, node_states}
    """
    edges = []
    node_states = {}

    for sat_id, sat in satellites.items():
        neighbors = find_neighbors(sat_id, satellites)
        msg = GossipMessage(
            sender_id=sat_id,
            timestamp=time.time(),
            x=sat.x, y=sat.y, z=sat.z,
            fuel_pct=sat.fuel_pct,
            status=sat.status,
            cdm_warnings=cdm_warnings.get(sat_id, []),
        )

        # Publish to each neighbor's ISL channel
        for neighbor_id in neighbors:
            await publish(f"acm:isl:{neighbor_id}", {
                "type": "GOSSIP",
                "from": sat_id,
                "to": neighbor_id,
                "state": {
                    "x": msg.x, "y": msg.y, "z": msg.z,
                    "fuel_pct": msg.fuel_pct,
                    "status": msg.status,
                    "warnings": msg.cdm_warnings,
                }
            })
            edges.append({
                "from": sat_id,
                "to": neighbor_id,
                "distance_km": round(isl_distance(sat, satellites[neighbor_id]), 1),
            })

        node_states[sat_id] = {
            "sat_id": sat_id,
            "lat": round(sat.lat, 4),
            "lon": round(sat.lon, 4),
            "alt_km": round(sat.alt_km, 2),
            "fuel_pct": sat.fuel_pct,
            "status": sat.status,
            "neighbors": neighbors,
        }

    # Cache network state for frontend
    network_state = {
        "timestamp": time.time(),
        "nodes": node_states,
        "edges": edges,
        "total_messages": len(edges),
    }
    await cache_set("acm:p2p:network", network_state, ttl=5)
    return network_state


def compute_cdm_warnings(satellites: dict, debris: dict, threshold_km: float = 5.0) -> dict[str, list]:
    """Check all satellite-debris pairs for close approaches."""
    warnings: dict[str, list] = {}

    for sat_id, sat in satellites.items():
        sat_warnings = []
        for deb_id, deb in debris.items():
            dist = math.sqrt(
                (sat.x - deb.x)**2 +
                (sat.y - deb.y)**2 +
                (sat.z - deb.z)**2
            )
            if dist < threshold_km:
                sat_warnings.append({
                    "debris_id": deb_id,
                    "distance_km": round(dist, 3),
                    "severity": "CRITICAL" if dist < 1.0 else "WARNING",
                })
        if sat_warnings:
            warnings[sat_id] = sat_warnings

    return warnings
