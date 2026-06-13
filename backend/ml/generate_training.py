"""
ml/generate_training.py
Generates 10,000 synthetic LEO conjunction scenarios.
Run: python generate_training.py
Output: ml/training_data.npy

Real world upgrade path:
- Replace rng.uniform(20,100) with telemetry_api.get_fuel(sat_id)
- Replace synthetic orbits with real TLE data from CelesTrak
- Replace angle_diff with actual SpaceTrack CDM data
"""

import math
import random
import numpy as np


def generate_scenario(seed: int) -> tuple[list[list[float]], list[float]]:
    """Generate one conjunction scenario: 20 feature steps + labels."""
    rng = random.Random(seed)

    # Two random LEO satellites
    r_a = 6928 + rng.uniform(-100, 100)
    r_b = 6928 + rng.uniform(-100, 100)
    inc_a = math.radians(rng.uniform(45, 65))
    inc_b = math.radians(rng.uniform(45, 65))

    # Relative angle — small angle = potential conjunction
    angle_diff = math.radians(rng.uniform(0, 10))

    # Fuel levels — used for who_evades label
    fuel_a = rng.uniform(20, 100)
    fuel_b = rng.uniform(20, 100)

    steps = []
    mu = 398600.4418

    for step in range(20):
        # Simple circular orbit positions
        ma_a = math.radians(step * 5 + rng.uniform(-2, 2))
        ma_b = ma_a + angle_diff + math.radians(step * 0.1)

        x_a = r_a * math.cos(ma_a)
        y_a = r_a * math.sin(ma_a) * math.cos(inc_a)
        z_a = r_a * math.sin(ma_a) * math.sin(inc_a)

        x_b = r_b * math.cos(ma_b)
        y_b = r_b * math.sin(ma_b) * math.cos(inc_b)
        z_b = r_b * math.sin(ma_b) * math.sin(inc_b)

        miss_dist = math.sqrt((x_a-x_b)**2 + (y_a-y_b)**2 + (z_a-z_b)**2)

        v_a = math.sqrt(mu / r_a)
        v_b = math.sqrt(mu / r_b)

        dvx = -v_a * math.sin(ma_a) - (-v_b * math.sin(ma_b))
        dvy = v_a * math.cos(ma_a) * math.cos(inc_a) - v_b * math.cos(ma_b) * math.cos(inc_b)
        dvz = v_a * math.cos(ma_a) * math.sin(inc_a) - v_b * math.cos(ma_b) * math.sin(inc_b)

        # TCA estimate
        dx, dy, dz = x_a-x_b, y_a-y_b, z_a-z_b
        dv2 = dvx**2 + dvy**2 + dvz**2
        tca = max(0, -(dx*dvx + dy*dvy + dz*dvz) / dv2) if dv2 > 1e-10 else 0.0

        steps.append([
            miss_dist, dvx, dvy, dvz, tca,
            r_a - 6378.137, r_b - 6378.137,
            fuel_a, fuel_b
        ])

    # ── Labels ──────────────────────────────────────────────────────────────
    min_dist = min(s[0] for s in steps)
    risk_prob = 1.0 if min_dist < 1.0 else max(0.0, 1.0 - min_dist / 10.0)
    predicted_tca = steps[-1][4]

    # Who evades: based on fuel difference
    # diff > +20 → A has clearly more fuel → A evades
    # diff < -20 → B has clearly more fuel → B evades
    # -20 to +20 → similar fuel → both split the burn
    diff = fuel_a - fuel_b
    if diff > 20:
        who_evades = [1.0, 0.0, 0.0]   # A evades
    elif diff < -20:
        who_evades = [0.0, 1.0, 0.0]   # B evades
    else:
        who_evades = [0.0, 0.0, 1.0]   # SPLIT

    labels = [risk_prob, min_dist, predicted_tca] + who_evades
    return steps, labels


def generate_dataset(n: int = 10000, output_path: str = "ml/training_data.npy"):
    print(f"Generating {n} conjunction scenarios...")
    X = []
    Y = []

    for i in range(n):
        if i % 1000 == 0:
            print(f"  {i}/{n}...")
        steps, labels = generate_scenario(seed=i)
        X.append(steps)
        Y.append(labels)

    X = np.array(X, dtype=np.float32)  # (n, 20, 9)
    Y = np.array(Y, dtype=np.float32)  # (n, 6)

    np.save(output_path, {"X": X, "Y": Y})
    print(f"✅ Saved to {output_path} — X:{X.shape}, Y:{Y.shape}")
    return X, Y


if __name__ == "__main__":
    generate_dataset(10000)
