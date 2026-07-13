"""
Autoencoder Training
PharmaGuard
"""

import time
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.layers import (
    Input,
    Dense
)

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import (
    EarlyStopping
)

from ml.models.common import (
    load_dataset,
    save_model
)

from ml.utils.helpers import (
    setup_logger,
    initialize_project_directories
)

from ml.utils.config import (
    AUTOENCODER_MODEL,
    AUTOENCODER_RESULTS,
    AUTOENCODER_SUMMARY
)

logger = setup_logger("Autoencoder")

def build_autoencoder(input_dim):

    inputs = Input(shape=(input_dim,))

    encoder = Dense(
        16,
        activation="relu"
    )(inputs)

    encoder = Dense(
        8,
        activation="relu"
    )(encoder)

    bottleneck = Dense(
        4,
        activation="relu"
    )(encoder)

    decoder = Dense(
        8,
        activation="relu"
    )(bottleneck)

    decoder = Dense(
        16,
        activation="relu"
    )(decoder)

    outputs = Dense(
        input_dim,
        activation="linear"
    )(decoder)

    autoencoder = Model(
        inputs,
        outputs
    )

    autoencoder.compile(

        optimizer="adam",

        loss="mse"
    )

    return autoencoder

def train():

    initialize_project_directories()

    logger.info("Loading Dataset")

    X = load_dataset()

    logger.info(
        "Initializing Autoencoder"
    )

    model = build_autoencoder(
        X.shape[1]
    )

    callback = EarlyStopping(

        monitor="val_loss",

        patience=5,

        restore_best_weights=True
    )

    logger.info(
        "Training Autoencoder"
    )

    start = time.perf_counter()

    history = model.fit(

        X,

        X,

        epochs=50,

        batch_size=256,

        validation_split=0.2,

        shuffle=True,

        callbacks=[callback],

        verbose=1
    )

    training_time = (
        time.perf_counter() - start
    )

    logger.info(

        f"Training Time : {training_time:.3f} sec"
    )

    logger.info(
        "Generating Predictions"
    )

    start = time.perf_counter()

    reconstructed = model.predict(
        X,
        verbose=0
    )

    inference_time = (
        time.perf_counter() - start
    )

    logger.info(

        f"Inference Time : {inference_time:.3f} sec"
    )

    reconstruction_error = np.mean(

        np.square(
            X.to_numpy() - reconstructed
        ),

        axis=1
    )

    return (

        model,

        X,

        reconstruction_error,

        training_time,

        inference_time
    )

def generate_risk_score(reconstruction_error):
    """
    Convert reconstruction error
    into a 0-100 risk score.
    """

    reconstruction_error = np.asarray(reconstruction_error)

    scaler = MinMaxScaler(
        feature_range=(0, 100)
    )

    risk_scores = scaler.fit_transform(
        reconstruction_error.reshape(-1, 1)
    )

    return risk_scores.flatten()

def assign_risk_level(risk_scores):

    return pd.cut(

        risk_scores,

        bins=[0, 30, 70, 100],

        labels=[
            "Low",
            "Medium",
            "High"
        ],

        include_lowest=True
    )

def save_results(

    X,

    predictions,

    reconstruction_error,

    risk_scores,

    training_time,

    inference_time
):

    df = X.copy()

    df["prediction"] = predictions

    df["anomaly_score"] = reconstruction_error

    df["risk_score"] = risk_scores

    df["risk_level"] = assign_risk_level(
        risk_scores
    )

    df.to_csv(
        AUTOENCODER_RESULTS,
        index=False
    )

    anomaly_count = int(
        (predictions == -1).sum()
    )

    normal_count = int(
        (predictions == 1).sum()
    )

    with open(

        AUTOENCODER_SUMMARY,

        "w"

    ) as report:

        report.write(
            "# Autoencoder Report\n\n"
        )

        report.write(
            "## Dataset\n\n"
        )

        report.write(
            f"- Shape : {X.shape}\n\n"
        )

        report.write(
            "## Performance\n\n"
        )

        report.write(
            f"- Training Time : {training_time:.3f} sec\n"
        )

        report.write(
            f"- Inference Time : {inference_time:.3f} sec\n\n"
        )

        report.write(
            "## Prediction Summary\n\n"
        )

        report.write(
            f"- Normal Shipments : {normal_count}\n"
        )

        report.write(
            f"- Anomalies : {anomaly_count}\n"
        )

        report.write(
            f"- Anomaly Percentage : "
            f"{(anomaly_count/len(X))*100:.2f}%\n\n"
        )

        report.write(
            "## Risk Score Statistics\n\n"
        )

        report.write(
            f"- Minimum : {risk_scores.min():.2f}\n"
        )

        report.write(
            f"- Maximum : {risk_scores.max():.2f}\n"
        )

        report.write(
            f"- Average : {risk_scores.mean():.2f}\n\n"
        )

        report.write(
            "## Risk Distribution\n\n"
        )

        risk_distribution = (

            pd.Series(

                assign_risk_level(
                    risk_scores
                )

            )

            .value_counts()

            .sort_index()
        )

        for level, count in risk_distribution.items():

            report.write(
                f"- {level}: {count}\n"
            )

    logger.info(
        "Results Saved"
    )

def main():

    (
        model,
        X,
        reconstruction_error,
        training_time,
        inference_time
    ) = train()

    risk_scores = generate_risk_score(
        reconstruction_error
    )

    # Top 2% reconstruction errors are anomalies
    threshold = np.percentile(
        reconstruction_error,
        98
    )

    predictions = np.where(
        reconstruction_error >= threshold,
        -1,
        1
    )

    model.save(
        AUTOENCODER_MODEL
    )

    logger.info(
        f"Saved : {AUTOENCODER_MODEL}"
    )

    save_results(

        X,

        predictions,

        reconstruction_error,

        risk_scores,

        training_time,

        inference_time

    )

    logger.info(
        "Autoencoder Training Completed"
    )


if __name__ == "__main__":
    main()