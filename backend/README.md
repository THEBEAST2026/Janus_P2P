# Janus P2P — Autonomous Constellation Manager

> What if every satellite could protect itself — without ever asking the ground?

Janus is a peer-to-peer autonomous collision-avoidance system for satellite constellations. Each satellite runs an onboard LSTM that predicts conjunction risk with nearby objects, then **negotiates directly with the neighboring satellite** over a simulated Inter-Satellite Link (ISL) to agree on evasive maneuvers — without waiting for ground station approval.

## Why

Traditional collision avoidance relies on ground operators reviewing Conjunction Data Messages (CDMs) and manually commanding burns. This is slow, doesn't scale to mega-constellations, and breaks down when satellites lose ground contact. Janus moves routine collision-avoidance decisions onto the satellites themselves, while keeping ground stations in control for high-risk escalations and fleet-wide threats (e.g. asteroid corridors).

## Architecture

```
┌─────────────────────────────────────────────┐
│              FastAPI Backend                  │
│                                                │
│  SimulationState (50 sats + 20 debris)        │
│         │                                     │
│         ├──► SatelliteAgent (×50)             │
│         │      └─► LSTM Predictor             │
│         │           └─► Negotiation Protocol  │
│         │                                     │
│         ├──► Gossip Protocol (ISL mesh)       │
│         │                                     │
│         └──► Threat Module (JPL Horizons)     │
└─────────────────────────────────────────────┘
```

## Key components

- **`core/state.py`** — Simulation state: 50 satellites on Keplerian orbits + 20 debris objects
- **`core/physics.py`** — SGP4 propagation, ECI ↔ geodetic conversion, delta-V estimation
- **`ml/conjunction_lstm.py`** — PyTorch LSTM predicting collision risk, miss distance, time-to-closest-approach, and which satellite should evade
- **`p2p/satellite_agent.py`** — Per-satellite agent: runs LSTM inference, negotiates burns with neighbors, executes maneuvers
- **`p2p/gossip.py`** — Gossip protocol simulating Inter-Satellite Links (k-nearest neighbor mesh)
- **`threat/asteroid.py`** — NASA JPL Horizons integration for real asteroid ephemeris + fleet-wide evasion coordination

## How the negotiation works

1. Each satellite maintains a rolling 20-step feature window with its nearest neighbors (relative position, velocity, fuel, altitude)
2. The onboard LSTM predicts: `risk_prob`, `miss_distance`, `time_to_closest_approach`, `who_should_evade`
3. If `risk_prob ≥ 0.75`, the satellite proposes a burn to its neighbor over the ISL
4. The neighbor checks its fuel reserves and accepts, rejects, or counter-proposes
5. If `risk_prob ≥ 0.95`, the maneuver is escalated to ground for approval instead of executing autonomously
6. All negotiations are logged and broadcast to the fleet via gossip

## Running locally

```bash
pip install -r requirements.txt

# (optional) generate training data and train the LSTM
python ml/generate_training.py
python ml/train.py

# start the server
python main.py
```

Visit `http://localhost:8000/docs` for the full interactive API documentation.

Redis is optional — the backend falls back to an in-memory store automatically if Redis isn't running.

## API overview

| Endpoint | Purpose |
|---|---|
| `POST /api/simulate/step` | Advance the simulation, run gossip + LSTM negotiations |
| `GET /api/simulate/status` | Fleet aggregates (fuel, status counts, CDM warnings) |
| `GET /api/visualization/snapshot` | All satellite + debris positions for 3D rendering |
| `GET /api/p2p/network-state` | ISL mesh graph |
| `GET /api/p2p/negotiation-log` | Autonomous maneuver history |
| `GET /api/p2p/lstm-accuracy` | Live model metrics |
| `POST /api/threat/asteroid-alert` | Seed an asteroid threat (optionally from JPL Horizons) |
| `POST /api/threat/execute-response` | Trigger fleet-wide coordinated evasion |
| `POST /api/threat/override` | Ground station override command |

## Status

Built for **FAR AWAY 2026** (Space & Aerospace track). Backend is functional with in-memory simulation; LSTM is trained on synthetic conjunction scenarios generated via simplified Keplerian propagation.

## Roadmap

- [ ] Replace synthetic LSTM training data with real CDM datasets
- [ ] WebRTC-based ISL instead of Redis pub/sub for true P2P
- [ ] Real TLE ingestion from CelesTrak/Space-Track
- [ ] Frontend: Three.js mission control dashboard
