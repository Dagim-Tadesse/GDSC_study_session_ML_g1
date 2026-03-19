import pandas as pd
import numpy as np

np.random.seed(42)

n=200

data = pd.DataFrame({
    "Creator_Name": [f"Creator_{i}" for i in range(n)],
    "Platf"