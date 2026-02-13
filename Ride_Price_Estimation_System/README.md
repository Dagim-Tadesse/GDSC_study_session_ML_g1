# Ride Price Estimation System (ML)

- Google colab link - **https://colab.research.google.com/drive/1Ba2xjkcjiBGldoxuXo62cVXT3E9gHCvq#scrollTo=36c0c11b**

## Overview

This project demonstrates an end-to-end ML workflow to **estimate a ride price** from trip + context features using:

- **Regression** (predict exact `ride_price`)
- **Classification** (predict `high_cost` vs `low_cost`)

The goal is not to replicate Uber’s production pricing, but to show correct ML framing, preprocessing, modeling, evaluation, and reflection.

---

## Dataset

A **synthetic dataset** (150–250 rows) is generated with these features:

| Feature         | Type                                  | Why it affects price                        |
| --------------- | ------------------------------------- | ------------------------------------------- |
| `distance_km`   | numeric                               | longer distance → higher price              |
| `duration_min`  | numeric                               | longer time → higher price                  |
| `traffic_level` | categorical (`low/medium/high`)       | heavy traffic → higher price                |
| `weather`       | categorical (`clear/rainy`)           | bad weather → higher price                  |
| `time_of_day`   | categorical (`morning/evening/night`) | surge-like effects                          |
| `demand_level`  | categorical (`low/high`)              | surge                                       |
| `passengers`    | numeric                               | more passengers → larger car → higher price |

Target:

- `ride_price` (continuous)

Data file:

- `data/ride_prices_synthetic.csv`

---

## Project Structure

- `data/` — generated CSV dataset
- `src/` — runnable Python script (EDA + preprocessing + regression + classification)
- `notebooks/` — the main analysis notebook

---

## How to Run

### 1) Install dependencies

From this folder:

```bash
pip install -r requirements.txt
```

### 2) Run the notebook (main)

Open and run:

- `notebooks/ride_price_estimation.ipynb`

### 3) Run the script (VS Code)

```bash
python src/ride_price_estimation.py
```

To run without plots:

```bash
python src/ride_price_estimation.py --no-plots
```

---

## Key Findings (Fill after running)

- Regression MAE: ~95.90
- Regression $R^2$: ~0.922
- Classification Accuracy: ~0.977
- Biggest price drivers observed: `distance_km`, `duration_min`, `demand_level`, `traffic_level`, `time_of_day` (plus a smaller effect from `weather` and `passengers`).

---

## Ethics & Limitations

- Surge-like pricing can disproportionately impact low-income riders.
- Model errors can overcharge or undercharge customers.
- Synthetic data may not capture real-world geographic and behavioral patterns.
