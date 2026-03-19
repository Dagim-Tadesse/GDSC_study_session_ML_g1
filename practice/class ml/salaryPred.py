from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Resolve CSV path relative to this script so it works from any current directory.
csv_path = Path(__file__).resolve().parents[1] / "csv" / "salary_data.csv"
df = pd.read_csv(csv_path)

feature_columns = [
    "YearsExperience",
    "Age",
    "EducationLevel",
    "CityTier",
    "SkillScore",
]
target_column = "Salary"

X = df[feature_columns]
y = df[target_column]

numeric_features = ["YearsExperience", "Age", "SkillScore"]
categorical_features = ["EducationLevel", "CityTier"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(drop="first",
         handle_unknown="ignore"), categorical_features),
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression()),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Model evaluation:")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")

sample_employee = pd.DataFrame(
    {
        "YearsExperience": [6.2],
        "Age": [32],
        "EducationLevel": ["Master"],
        "CityTier": ["Tier1"],
        "SkillScore": [85],
    }
)

predicted_salary = model.predict(sample_employee)[0]
print(f"Predicted salary for sample employee: {predicted_salary:.0f}")
