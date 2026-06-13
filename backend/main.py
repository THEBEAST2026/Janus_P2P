"""
main.py
Janus P2P — FastAPI entry point.
Startup sequence as per system design document.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-20s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("janus.main")

# ─── Global agent registry ────────────────────────────────────────────────────
from p2p.satellite_agent import SatelliteAgent
from ml.conjunction_lstm import load_predictor, LSTMPredictor

agents: dict[str, SatelliteAgent] = {}
predictor: LSTMPredictor | None = None


async def init_agents():
    """Initialize one SatelliteAgent per satellite, inject shared LSTM predictor."""
    global agents
    from core.state import simulation
    agents.clear()
    for sat_id in simulation.satellites:
        agents[sat_id] = SatelliteAgent(sat_id=sat_id, predictor=predictor)
    logger.info(f"✅ {len(agents)} satellite agents initialized")


# ─── Lifespan ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor

    # Step 1: Redis
    from core.cache import init_redis
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    await init_redis(redis_url)

    # Step 2: SimulationState (already initialized as singleton on import)
    from core.state import simulation
    logger.info(f"✅ SimulationState ready — {len(simulation.satellites)} satellites")

    # Step 3: Load LSTM model
    model_path = os.getenv("LSTM_MODEL_PATH", "ml/conjunction_lstm.pt")
    predictor = load_predictor(model_path if os.path.exists(model_path) else None)
    logger.info("✅ LSTM predictor ready")

    # Step 4: Initialize satellite agents
    await init_agents()

    # Step 5: Optionally seed real ISRO TLEs
    if os.getenv("ISRO_SEED", "0") == "1":
        asyncio.create_task(_seed_isro_tles())

    logger.info("🚀 Janus P2P backend is NOMINAL — ready to serve")
    yield
    logger.info("👋 Janus P2P shutting down")


async def _seed_isro_tles():
    """Background task: fetch real ISRO TLEs from CelesTrak."""
    try:
        import httpx
        from core.cache import cache_set
        from core.state import simulation

        CELESTRAK_URL = "https://celestrak.org/SOCRATES/query.php"
        ISRO_GROUPS = [
            "https://celestrak.org/SOCRATES/query.php?CATNR=NAVIC&FORMAT=JSON",
            "https://celestrak.org/SATCAT/search.php?INTLDES=2023-&FORMAT=JSON",
        ]

        logger.info("🛰  Fetching ISRO TLEs from CelesTrak...")
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                "https://celestrak.org/SATCAT/search.php",
                params={"CATNR": "NAVIC", "FORMAT": "JSON"}
            )
            if resp.status_code == 200:
                data = resp.json()
                await cache_set("acm:isro:catalogue", data, ttl=86400)
                logger.info(f"✅ Fetched {len(data)} ISRO satellite records")
    except Exception as e:
        logger.warning(f"ISRO TLE seed failed: {e}")


# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Janus P2P — Autonomous Constellation Manager",
    description=(
        "Satellites negotiate collision avoidance with each other "
        "using onboard ML — no ground operator command required for routine maneuvers."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Routers ──────────────────────────────────────────────────────────────────
from routers.simulate      import router as simulate_router
from routers.visualization import router as viz_router
from routers.p2p           import router as p2p_router
from routers.threat        import router as threat_router
from routers.telemetry     import router as telemetry_router
from routers.maneuver      import router as maneuver_router

app.include_router(simulate_router)
app.include_router(viz_router)
app.include_router(p2p_router)
app.include_router(threat_router)
app.include_router(telemetry_router)
app.include_router(maneuver_router)


# ─── Health ───────────────────────────────────────────────────────────────────
@app.get("/health", tags=["health"])
async def health():
    from core.cache import redis_health
    from core.state import simulation
    return {
        "status": "NOMINAL",
        "satellites": len(simulation.satellites),
        "debris": len(simulation.debris),
        "agents": len(agents),
        "step": simulation.step_count,
        "lstm_loaded": predictor is not None,
        "redis": await redis_health(),
    }


@app.get("/", tags=["health"])
async def root():
    return {
        "project": "Janus P2P",
        "tagline": "What if every ISRO satellite could protect itself without ever asking the ground?",
        "docs": "/docs",
        "health": "/health",
    }


# ─── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
