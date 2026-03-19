"""Fake News Detection using a Neural Network.

Behavior:
- If `data/fake_news_dataset.csv` does not exist, this script generates a synthetic dataset.
- Uses TensorFlow/Keras when available.
- Falls back to sklearn MLPClassifier (still a neural network) when TensorFlow is not installed.
"""

from __future__ import annotations

import os
import re
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neural_network import MLPClassifier

# Try TensorFlow first. If not present, fallback to sklearn MLP.
HAS_TENSORFLOW = True
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.optimizers import Adam
except ModuleNotFoundError:
    HAS_TENSORFLOW = False


SEED = 42
np.random.seed(SEED)
if HAS_TENSORFLOW:
    tf.random.set_seed(SEED)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATASET_PATH = DATA_DIR / "fake_news_dataset.csv"
EPOCHS = 12
BATCH_SIZE = 32


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _scramble_words(sentence: str, rng: np.random.Generator) -> str:
    words = sentence.split()
    if len(words) <= 4:
        return sentence
    middle = words[1:-1]
    rng.shuffle(middle)
    return " ".join([words[0], *middle, words[-1]])


def generate_synthetic_fake_news_dataset(dataset_path: Path, n_samples: int = 3000) -> None:
    """Generate a balanced, varied fake-news dataset with clear and scrambled text."""
    rng = np.random.default_rng(SEED)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    entities = [
        "NASA", "WHO", "Oxford University", "UN", "local government", "researchers",
        "doctors", "economists", "climate scientists", "engineers", "police department",
        "education board", "health ministry", "space agency", "independent analysts",
    ]
    places = [
        "New York", "London", "Tokyo", "Delhi", "Nairobi", "Berlin", "Sao Paulo",
        "Sydney", "Toronto", "Cairo", "Lagos", "Seoul", "Dubai", "Jakarta", "Paris",
    ]
    topics = [
        "public health", "renewable energy", "transport", "education", "space exploration",
        "water quality", "food safety", "employment", "housing", "internet safety",
    ]
    metrics = [
        "3%", "7%", "12%", "18%", "25%", "34%", "41%", "52%", "63%", "78%",
    ]

    real_templates = [
        "{entity} published a verified report on {topic} improvements in {place}.",
        "Officials in {place} confirmed new data showing {metric} growth in {topic}.",
        "Researchers from {entity} announced peer-reviewed findings on {topic}.",
        "The health department in {place} released a statement about {topic} policy updates.",
        "A national survey reported measurable progress in {topic} across major cities.",
    ]

    fake_templates = [
        "Secret project in {place} makes people invisible overnight, says unnamed source.",
        "Aliens partnered with {entity} to control weather and minds worldwide.",
        "Miracle pill from {place} instantly cures all diseases in 24 hours.",
        "Hidden lab under {place} creates time machine for politicians.",
        "Government confirms humans can live without sleep after new experiment.",
    ]

    noise_phrases = [
        "SHOCKING truth revealed", "you won't believe this", "exclusive leak",
        "share before deleted", "suppressed by mainstream media", "urgent update",
    ]

    rows: list[dict[str, Any]] = []
    half = n_samples // 2

    # Real headlines (label 0)
    for _ in range(half):
        t = rng.choice(real_templates)
        text = t.format(
            entity=rng.choice(entities),
            topic=rng.choice(topics),
            place=rng.choice(places),
            metric=rng.choice(metrics),
        )
        if rng.random() < 0.15:
            text += f" According to official records from {rng.integers(2018, 2026)}."
        rows.append({"text": text, "label": 0})

    # Fake headlines (label 1), some intentionally scrambled/noisy
    for _ in range(half):
        t = rng.choice(fake_templates)
        text = t.format(
            entity=rng.choice(entities),
            place=rng.choice(places),
        )
        if rng.random() < 0.55:
            text = _scramble_words(text, rng)
        if rng.random() < 0.6:
            text += f" | {rng.choice(noise_phrases)} !!!"
        rows.append({"text": text, "label": 1})

    rng.shuffle(rows)
    df = pd.DataFrame(rows, columns=["text", "label"])
    df.to_csv(dataset_path, index=False)
    print(f"Generated synthetic dataset at: {dataset_path}")
    print(f"Dataset shape: {df.shape}")


def ensure_dataset_exists(dataset_path: Path) -> None:
    if dataset_path.exists():
        return
    generate_synthetic_fake_news_dataset(dataset_path)


def load_data(dataset_path: Path):
    ensure_dataset_exists(dataset_path)

    df = pd.read_csv(dataset_path)
    required_cols = {"text", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print("\nDataset loaded from:", dataset_path)
    print("Dataset shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nClass distribution (0=Real, 1=Fake):")
    print(df["label"].value_counts().sort_index())

    df = df[["text", "label"]].dropna().copy()
    df["text"] = df["text"].apply(clean_text)

    x_text = df["text"].values
    y = df["label"].astype(int).values

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    x = vectorizer.fit_transform(x_text).toarray()
    print("\nVectorized feature shape:", x.shape)

    return x, y, vectorizer


def build_model(input_dim: int):
    if HAS_TENSORFLOW:
        model = Sequential(
            [
                Dense(128, activation="relu", input_shape=(input_dim,)),
                Dropout(0.3),
                Dense(64, activation="relu"),
                Dropout(0.2),
                Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        return model

    # Fallback neural network implementation from sklearn.
    return MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        max_iter=EPOCHS,
        batch_size=BATCH_SIZE,
        random_state=SEED,
    )


def plot_training_history(history) -> None:
    if history is None:
        print("Skipping training curves: TensorFlow history not available in sklearn fallback mode.")
        return

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Loss vs Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Accuracy vs Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()


def predict_proba(model, x_input: np.ndarray) -> np.ndarray:
    if HAS_TENSORFLOW:
        return model.predict(x_input, verbose=0).flatten()
    return model.predict_proba(x_input)[:, 1]


def predict_headline(model, vectorizer: TfidfVectorizer, headline: str):
    cleaned = clean_text(headline)
    vec = vectorizer.transform([cleaned]).toarray()
    prob_fake = float(predict_proba(model, vec)[0])
    label_name = "Fake News" if prob_fake >= 0.5 else "Real News"
    return label_name, prob_fake


def main() -> None:
    print("Backend:", "TensorFlow/Keras" if HAS_TENSORFLOW else "sklearn MLP fallback")

    x, y, vectorizer = load_data(DATASET_PATH)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=SEED, stratify=y
    )
    print("\nTrain shape:", x_train.shape, y_train.shape)
    print("Test shape:", x_test.shape, y_test.shape)

    model = build_model(input_dim=x_train.shape[1])

    history = None
    if HAS_TENSORFLOW:
        model.summary()
        history = model.fit(
            x_train,
            y_train,
            validation_split=0.1,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=1,
        )
    else:
        model.fit(x_train, y_train)

    plot_training_history(history)

    y_prob = predict_proba(model, x_test)
    y_pred = (y_prob >= 0.5).astype(int)

    test_acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {test_acc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
          target_names=["Real (0)", "Fake (1)"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    examples = [
        "Scientists confirm water on Mars.",
        "Secret government project creates invisible humans.",
        "Scientists discover a new planet similar to Earth.",
        "Aliens have landed in New York and taken control of the city.",
    ]
    print("\nManual headline predictions:")
    for text in examples:
        label, prob = predict_headline(model, vectorizer, text)
        print(f"- {text}")
        print(f"  Prediction: {label} (fake_probability={prob:.4f})")

    if HAS_TENSORFLOW:
        model_path = BASE_DIR / "fake_news_nn_model.keras"
        model.save(model_path)
    else:
        model_path = BASE_DIR / "fake_news_nn_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

    vectorizer_path = BASE_DIR / "tfidf_vectorizer.pkl"
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)

    print(f"\nModel saved to: {model_path}")
    print(f"Vectorizer saved to: {vectorizer_path}")
    print(f"Dataset location: {DATASET_PATH}")


if __name__ == "__main__":
    main()
