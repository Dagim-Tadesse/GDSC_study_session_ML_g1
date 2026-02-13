import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def generate_dataset(rows: int, seed: int = 42) -> pd.DataFrame:
    rng = _make_rng(seed)

    traffic_levels = np.array(["low", "medium", "high"])
    weathers = np.array(["clear", "rainy"])
    times_of_day = np.array(["morning", "evening", "night"])
    demand_levels = np.array(["low", "high"])

    distance_km = rng.uniform(0.8, 28.0, size=rows).round(2)

    traffic_level = rng.choice(traffic_levels, size=rows, p=[0.45, 0.35, 0.20])
    weather = rng.choice(weathers, size=rows, p=[0.78, 0.22])
    time_of_day = rng.choice(times_of_day, size=rows, p=[0.40, 0.40, 0.20])
    demand_level = rng.choice(demand_levels, size=rows, p=[0.65, 0.35])
    passengers = rng.integers(1, 5, size=rows)

    # Duration model (minutes)
    # Realism target:
    #   15–20 km with normal traffic often falls around 30–40 minutes.
    traffic_time_add = np.select(
        [traffic_level == "low", traffic_level ==
            "medium", traffic_level == "high"],
        [0.0, 5.0, 12.0],
        default=0.0,
    )
    minutes_per_km = rng.normal(2.0, 0.25, size=rows)  # ~30 km/h average
    duration_min = distance_km * minutes_per_km + \
        traffic_time_add + rng.normal(0, 2.0, size=rows)
    duration_min = np.clip(duration_min, 3.0, None).round(1)

    # Inject a few outliers BEFORE computing price so features and target stay consistent.
    outlier_count = max(2, rows // 60)
    outlier_idx = rng.choice(rows, size=outlier_count, replace=False)
    distance_km[outlier_idx] = distance_km[outlier_idx] * \
        rng.uniform(1.8, 2.6, size=outlier_count)
    duration_min[outlier_idx] = duration_min[outlier_idx] * \
        rng.uniform(1.7, 2.4, size=outlier_count)
    distance_km = distance_km.round(2)
    duration_min = duration_min.round(1)

    # Price model (Birr-like scale)
    # Realism target:
    #   15–20 km, 30–40 min, normal traffic should typically land around 400–700.
    base_fare = 80.0
    per_km = 20.0
    per_min = 6.0

    passengers_add = np.select(
        [passengers <= 2, passengers == 3, passengers >= 4],
        [0.0, 60.0, 100.0],
        default=0.0,
    )

    traffic_mult = np.select(
        [traffic_level == "low", traffic_level ==
            "medium", traffic_level == "high"],
        [1.00, 1.15, 1.35],
        default=1.00,
    )
    weather_mult = np.where(weather == "rainy", 1.10, 1.00)
    time_mult = np.select(
        [time_of_day == "morning", time_of_day ==
            "evening", time_of_day == "night"],
        [1.00, 1.10, 1.20],
        default=1.00,
    )
    demand_mult = np.where(demand_level == "high", 1.25, 1.00)

    core = base_fare + distance_km * per_km + \
        duration_min * per_min + passengers_add
    multiplier = traffic_mult * weather_mult * time_mult * demand_mult
    noise = rng.normal(0, 25.0, size=rows)
    ride_price = core * multiplier + noise
    ride_price = np.clip(ride_price, 120.0, None).round(2)

    df = pd.DataFrame(
        {
            "distance_km": distance_km,
            "duration_min": duration_min,
            "traffic_level": traffic_level,
            "weather": weather,
            "time_of_day": time_of_day,
            "demand_level": demand_level,
            "passengers": passengers,
            "ride_price": ride_price,
        }
    )

    # Inject a little missingness so the notebook can demonstrate cleaning.
    # Keep it small so models still train easily.
    missing_frac = 0.03
    for col in ["traffic_level", "weather", "time_of_day", "demand_level", "distance_km", "duration_min"]:
        mask = rng.random(rows) < missing_frac
        df.loc[mask, col] = np.nan

    return df


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic ride price dataset.")
    parser.add_argument("--rows", type=int, default=200,
                        help="Number of rows to generate (150–250 recommended).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path("data") / "ride_prices_synthetic.csv"),
        help="Output CSV path.",
    )

    args = parser.parse_args()

    if args.rows < 50:
        raise SystemExit("Please generate at least 50 rows.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate_dataset(rows=args.rows, seed=args.seed)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
