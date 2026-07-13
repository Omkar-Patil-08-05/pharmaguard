"""
Data Preprocessing Pipeline
PharmaGuard
"""

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler

from ml.utils.helpers import (
    setup_logger,
    initialize_project_directories
)

from ml.utils.config import (
    FINAL_DATASET,
    PROCESSED_DATASET,
    PREPROCESSOR_PIPELINE
)

logger = setup_logger("PREPROCESSING")

def load_dataset():

    logger.info("Loading dataset...")

    df = pd.read_csv(FINAL_DATASET)

    logger.info("Dataset Loaded")

    return df

def remove_identifier_columns(df):

    logger.info("Removing Identifier Columns")

    columns = [
        "batch_id",
        "shipment_id"
    ]

    return df.drop(columns=columns)

def convert_dates(df):

    logger.info("Converting Date Columns")

    df["shipment_date"] = pd.to_datetime(df["shipment_date"])

    df["delivery_date"] = pd.to_datetime(df["delivery_date"])

    df["shipment_month"] = df["shipment_date"].dt.month

    df["shipment_day"] = df["shipment_date"].dt.day

    df["shipment_weekday"] = df["shipment_date"].dt.weekday

    df["delivery_month"] = df["delivery_date"].dt.month

    df["delivery_day"] = df["delivery_date"].dt.day

    df["delivery_weekday"] = df["delivery_date"].dt.weekday

    df.drop(
        columns=[
            "shipment_date",
            "delivery_date"
        ],
        inplace=True
    )

    return df

def build_preprocessing_pipeline(df):

    logger.info("Building Preprocessing Pipeline")

    categorical_columns = [
        "shipping_mode",
        "delivery_status",
        "order_status"
    ]

    numerical_columns = [
        "transit_time",
        "scheduled_transit_time",
        "distance",
        "temperature",
        "warehouse_stops",
        "quantity",
        "late_delivery_risk"
    ]

    categorical_pipeline = Pipeline(
        steps=[
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1
                )
            )
        ]
    )

    numerical_pipeline = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    pipeline = ColumnTransformer(
        transformers=[
            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            ),
            (
                "numerical",
                numerical_pipeline,
                numerical_columns
            )
        ]
    )

    return pipeline

def preprocess(df):

    logger.info("Selecting ML Features")

    selected_columns = [
        "transit_time",
        "scheduled_transit_time",
        "distance",
        "temperature",
        "warehouse_stops",
        "quantity",
        "late_delivery_risk",
        "shipping_mode",
        "delivery_status",
        "order_status"
    ]

    df = df[selected_columns]

    pipeline = build_preprocessing_pipeline(df)

    processed = pipeline.fit_transform(df)

    columns = selected_columns

    processed_df = pd.DataFrame(
        processed,
        columns=columns
    )

    processed_df.to_csv(
        PROCESSED_DATASET,
        index=False
    )

    joblib.dump(
        pipeline,
        PREPROCESSOR_PIPELINE
    )

    logger.info("Processed Dataset Saved")

    logger.info("Preprocessing Pipeline Saved")

def main():

    initialize_project_directories()

    logger.info("Starting Preprocessing")

    df = load_dataset()

    df = remove_identifier_columns(df)

    df = convert_dates(df)

    preprocess(df)

    logger.info("Preprocessing Completed")

if __name__ == "__main__":
    main()