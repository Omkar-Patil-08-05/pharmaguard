"""
Common utilities for model training.
"""

import time
import joblib
import pandas as pd

from pathlib import Path

from ml.utils.helpers import setup_logger

from ml.utils.config import (
    PROCESSED_DATASET,
    MODEL_DIR
)

logger = setup_logger("MODEL")


def load_dataset():

    logger.info("Loading processed dataset...")

    df = pd.read_csv(PROCESSED_DATASET)

    logger.info(
        f"Dataset Shape : {df.shape}"
    )

    return df


def train_model(model, X):

    logger.info(
        f"Training {model.__class__.__name__}"
    )

    start = time.perf_counter()

    model.fit(X)

    end = time.perf_counter()

    training_time = end - start

    logger.info(
        f"Training Time : {training_time:.3f} sec"
    )

    return model, training_time


def save_model(model, filename):

    path = MODEL_DIR / filename

    joblib.dump(model, path)

    logger.info(
        f"Saved : {path}"
    )


def predict(model, X):

    logger.info("Generating Predictions")

    predictions = model.predict(X)

    scores = model.decision_function(X)

    return predictions, scores