"""
One-Class SVM Training
PharmaGuard
"""

import time
import numpy as np
import pandas as pd

from sklearn.svm import OneClassSVM
from sklearn.preprocessing import MinMaxScaler

from ml.models.common import (
    load_dataset,
    train_model,
    save_model,
    predict
)

from ml.utils.helpers import (
    setup_logger,
    initialize_project_directories
)

from ml.utils.config import (
    ONE_CLASS_SVM_MODEL,
    ONE_CLASS_SVM_RESULTS,
    ONE_CLASS_SVM_SUMMARY
)

logger = setup_logger("OneClassSVM")


def train():

    initialize_project_directories()

    logger.info("Loading Dataset")

    X = load_dataset()

    logger.info("Initializing One-Class SVM")

    model = OneClassSVM(
        kernel="rbf",
        gamma="scale",
        nu=0.02
    )

    model, training_time = train_model(model, X)

    logger.info("Generating Predictions")

    start = time.perf_counter()

    predictions, scores = predict(model, X)

    inference_time = time.perf_counter() - start

    logger.info(
        f"Inference Time : {inference_time:.3f} sec"
    )

    return (
        model,
        X,
        predictions,
        scores,
        training_time,
        inference_time
    )


def generate_risk_score(scores):
    """
    Convert One-Class SVM decision scores
    into a 0-100 risk score.

    Higher score = Higher Risk
    """

    scaler = MinMaxScaler(feature_range=(0, 100))

    risk_scores = scaler.fit_transform(
        (-scores).reshape(-1, 1)
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
    scores,
    risk_scores,
    training_time,
    inference_time
):

    df = X.copy()

    df["prediction"] = predictions

    df["anomaly_score"] = scores

    df["risk_score"] = risk_scores

    df["risk_level"] = assign_risk_level(risk_scores)

    df.to_csv(
        ONE_CLASS_SVM_RESULTS,
        index=False
    )

    anomaly_count = int((predictions == -1).sum())

    normal_count = int((predictions == 1).sum())

    with open(
        ONE_CLASS_SVM_SUMMARY,
        "w"
    ) as report:

        report.write("# One-Class SVM Report\n\n")

        report.write("## Dataset\n\n")

        report.write(f"- Shape : {X.shape}\n\n")

        report.write("## Performance\n\n")

        report.write(
            f"- Training Time : {training_time:.3f} sec\n"
        )

        report.write(
            f"- Inference Time : {inference_time:.3f} sec\n\n"
        )

        report.write("## Prediction Summary\n\n")

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

        report.write("## Risk Score Statistics\n\n")

        report.write(
            f"- Minimum : {risk_scores.min():.2f}\n"
        )

        report.write(
            f"- Maximum : {risk_scores.max():.2f}\n"
        )

        report.write(
            f"- Average : {risk_scores.mean():.2f}\n\n"
        )

        report.write("## Risk Distribution\n\n")

        risk_distribution = (
            pd.Series(assign_risk_level(risk_scores))
            .value_counts()
            .sort_index()
        )

        for level, count in risk_distribution.items():

            report.write(
                f"- {level}: {count}\n"
            )

    logger.info("Results Saved")


def main():

    (
        model,
        X,
        predictions,
        scores,
        training_time,
        inference_time
    ) = train()

    risk_scores = generate_risk_score(scores)

    save_model(
        model,
        ONE_CLASS_SVM_MODEL.name
    )

    save_results(
        X,
        predictions,
        scores,
        risk_scores,
        training_time,
        inference_time
    )

    logger.info(
        "One-Class SVM Training Completed"
    )


if __name__ == "__main__":
    main()