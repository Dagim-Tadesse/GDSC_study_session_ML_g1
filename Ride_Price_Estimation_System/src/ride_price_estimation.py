from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    accuracy_score,
    confusion_matrix,
)


def clip_iqr(series: pd.Series, k: float = 1.5) -> pd.Series:
    """Clip outliers using the IQR rule.

    Why:
    - Extreme values (outliers) can overly influence simple linear models.
    - Clipping is a simple, explainable way to reduce instability.
    """

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    low = q1 - k * iqr
    high = q3 + k * iqr
    return series.clip(low, high)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Ride Price Estimation: regression (LinearRegression) + "
            "classification (LogisticRegression) pipeline."
        )
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=str(Path(__file__).resolve(
        ).parents[1] / "data" / "ride_prices_synthetic.csv"),
        help="Path to ride_prices_synthetic.csv",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable matplotlib plots.",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------
    # 1) Problem framing (as comments)
    # ------------------------------------------------------------
    # Problem:
    #   Estimate ride price based on trip and contextual features.
    #
    # Why ML instead of fixed rules?
    #   Pricing depends on many interacting factors (distance, traffic, demand, weather, time-of-day).
    #   ML can learn those interactions from data.
    #
    # What is the model learning?
    #   The relationship between trip conditions and the final ride price.

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # ------------------------------------------------------------
    # 2) EDA (as code + a couple of plots)
    # ------------------------------------------------------------
    # We check:
    # - schema / dtypes
    # - summary stats
    # - missing values
    # - at least one visualization

    print("\n--- Dataset Preview ---")
    print(df.head())
    print("\n--- Info ---")
    print(df.info())
    print("\n--- Describe ---")
    print(df.describe(include="all").T)
    print("\n--- Missing values ---")
    print(df.isna().sum().sort_values(ascending=False))

    if not args.no_plots:
        sns.set_context("notebook")
        plt.style.use("seaborn-v0_8")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].scatter(df["distance_km"], df["ride_price"], alpha=0.7)
        axes[0].set_title("Distance vs Ride Price")
        axes[0].set_xlabel("distance_km")
        axes[0].set_ylabel("ride_price")

        sns.boxplot(data=df, x="traffic_level", y="ride_price", ax=axes[1])
        axes[1].set_title("Ride Price by Traffic Level")
        axes[1].set_xlabel("traffic_level")
        axes[1].set_ylabel("ride_price")

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------
    # 3) Data cleaning & preprocessing
    # ------------------------------------------------------------
    # Why:
    #   Poor data causes biased models and unstable predictions.
    #
    # Required steps:
    # - handle missing values
    # - encode categoricals (get_dummies)
    # - scale numeric features (StandardScaler)
    # - handle outliers (IQR clipping)

    df_clean = df.copy()

    numeric_cols = ["distance_km", "duration_min", "passengers"]
    categorical_cols = ["traffic_level",
                        "weather", "time_of_day", "demand_level"]

    # Missing values: numeric -> median, categorical -> mode
    for col in numeric_cols:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    for col in categorical_cols:
        mode_val = df_clean[col].mode(dropna=True)
        fill_val = mode_val.iloc[0] if len(mode_val) else "unknown"
        df_clean[col] = df_clean[col].fillna(fill_val)

    # Outliers: clip only continuous features (not the target)
    for col in ["distance_km", "duration_min"]:
        df_clean[col] = clip_iqr(df_clean[col])

    # Encode categoricals using get_dummies(drop_first=True)
    df_model = pd.get_dummies(
        df_clean, columns=categorical_cols, drop_first=True)

    X = df_model.drop(columns=["ride_price"])
    y = df_model["ride_price"]

    # Scale numeric columns only (dummies are already 0/1)
    preprocess = ColumnTransformer(
        transformers=[("num", StandardScaler(), numeric_cols)],
        remainder="passthrough",
    )

    # ------------------------------------------------------------
    # 4) Regression model
    # ------------------------------------------------------------
    # Goal:
    #   predict the exact ride_price
    #
    # Evaluate using:
    # - MAE
    # - R^2

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    reg_model = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", LinearRegression()),
        ]
    )
    reg_model.fit(X_train, y_train)

    y_pred = reg_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n--- Regression Results ---")
    print(f"MAE: {mae:.2f}")
    print(f"R^2: {r2:.3f}")

    if not args.no_plots:
        plt.figure(figsize=(5, 5))
        plt.scatter(y_test, y_pred, alpha=0.7)
        plt.xlabel("Actual ride_price")
        plt.ylabel("Predicted ride_price")
        plt.title("Linear Regression: Actual vs Predicted")

        min_v = float(min(y_test.min(), y_pred.min()))
        max_v = float(max(y_test.max(), y_pred.max()))
        plt.plot([min_v, max_v], [min_v, max_v], "r--", linewidth=1)
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------
    # 5) Classification model
    # ------------------------------------------------------------
    # Create a binary target:
    #   high_cost = 1 if ride_price > median else 0
    #
    # Why:
    #   Sometimes you only need a decision (expensive vs cheap), not an exact number.
    #
    # Logistic regression outputs probability; if p > 0.5 -> class 1.

    median_price = float(y.median())
    y_class = (y > median_price).astype(int)

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X,
        y_class,
        test_size=0.2,
        random_state=42,
        stratify=y_class,
    )

    clf_model = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )
    clf_model.fit(X_train_c, y_train_c)

    y_pred_c = clf_model.predict(X_test_c)
    acc = accuracy_score(y_test_c, y_pred_c)
    cm = confusion_matrix(y_test_c, y_pred_c)

    print("\n--- Classification Results ---")
    print(f"Accuracy: {acc:.3f}")
    print("Confusion matrix:\n", cm)

    if not args.no_plots:
        plt.figure(figsize=(4.5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Logistic Regression: Confusion Matrix")
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------
    # 6) Compare models
    # ------------------------------------------------------------
    # Regression:
    #   more detailed (exact price)
    # Classification:
    #   easier for decision-making (expensive vs cheap)

    # ------------------------------------------------------------
    # 7) Ethical reflection
    # ------------------------------------------------------------
    # - Unfair pricing: surge-like effects can disproportionately affect low-income riders.
    # - Real-world risk: model errors could overcharge customers.
    # - Limitation: synthetic data may not represent real-world behavior.

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
