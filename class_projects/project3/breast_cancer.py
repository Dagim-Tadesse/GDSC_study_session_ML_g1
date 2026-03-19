from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Load and clean the breast cancer dataset."""
    data_path = Path(__file__).resolve().parent.parent / \
        "csv" / "breast_cancer_bd.csv"
    df = pd.read_csv(data_path, na_values=["?"])

    df = df.dropna().copy()
    df["Class"] = df["Class"].astype(int).map({2: 0, 4: 1})

    X = df.drop(columns=["Class", "Sample code number"])
    y = df["Class"]
    return X, y


def train_and_evaluate(X: pd.DataFrame, y: pd.Series) -> None:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("Logistic Regression for Breast Cancer Detection")
    print("------------------------------------------------")
    print(f"Accuracy: {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall: {rec:.3f}")
    print(f"F1-score: {f1:.3f}\n")

    print("Confusion Matrix (rows=true, cols=pred):")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
          target_names=["Benign", "Malignant"]))

    feature_importance = pd.Series(
        model.coef_[0], index=X.columns).sort_values(key=abs, ascending=False)
    print("Top weighted features:")
    print(feature_importance.head(5))


def main() -> None:
    X, y = load_dataset()
    train_and_evaluate(X, y)


if __name__ == "__main__":
    main()
