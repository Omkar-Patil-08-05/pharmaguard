"""
Exploratory Data Analysis
PharmaGuard
"""

import pandas as pd
from ml.eda.visualization import save_histogram

from ml.utils.config import (
    FINAL_DATASET,
    SUMMARY_STATISTICS,
    DATA_QUALITY_REPORT,
    MISSING_VALUES_REPORT,
    DUPLICATE_REPORT,
    UNIQUE_VALUES_REPORT,
    CONSTANT_COLUMNS_REPORT,
    DATA_TYPES_REPORT
)

from ml.utils.helpers import (
    setup_logger,
    initialize_project_directories
)

logger = setup_logger("EDA")


def load_dataset():

    logger.info("Loading dataset...")

    df = pd.read_csv(FINAL_DATASET)

    logger.info(f"Dataset Loaded Successfully")

    return df


def dataset_information(df):

    logger.info("Displaying Dataset Information")

    print("\n")
    print("=" * 70)
    print("DATASET INFORMATION")
    print("=" * 70)

    print(f"Rows      : {df.shape[0]}")
    print(f"Columns   : {df.shape[1]}")

    print("\nColumns")

    for column in df.columns:
        print(f"- {column}")

    print("\nData Types")

    print(df.dtypes)


def summary_statistics(df):

    logger.info("Generating Summary Statistics")

    summary = df.describe(include="all").transpose()

    summary.to_csv(SUMMARY_STATISTICS)

    logger.info("Summary Statistics Saved")

def missing_values_analysis(df):

    logger.info("Checking Missing Values")

    missing = df.isnull().sum()

    missing_df = pd.DataFrame({
        "Column": missing.index,
        "Missing Values": missing.values,
        "Percentage": (
            missing.values / len(df) * 100
        ).round(2)
    })

    missing_df.to_csv(MISSING_VALUES_REPORT, index=False)

    return missing_df


def duplicate_analysis(df):

    logger.info("Checking Duplicate Records")

    duplicates = df.duplicated().sum()

    duplicate_df = pd.DataFrame({
        "Metric": ["Duplicate Rows"],
        "Count": [duplicates]
    })

    duplicate_df.to_csv(DUPLICATE_REPORT, index=False)

    return duplicates


def unique_value_analysis(df):

    logger.info("Calculating Unique Values")

    unique_df = pd.DataFrame({
        "Column": df.columns,
        "Unique Values": [df[col].nunique() for col in df.columns]
    })

    unique_df.to_csv(UNIQUE_VALUES_REPORT, index=False)

    return unique_df


def constant_column_analysis(df):

    logger.info("Checking Constant Columns")

    constant_columns = []

    for col in df.columns:

        if df[col].nunique() == 1:
            constant_columns.append(col)

    constant_df = pd.DataFrame({
        "Constant Columns": constant_columns
    })

    constant_df.to_csv(CONSTANT_COLUMNS_REPORT, index=False)

    return constant_columns


def data_type_analysis(df):

    logger.info("Saving Data Types")

    dtype_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values
    })

    dtype_df.to_csv(DATA_TYPES_REPORT, index=False)

def generate_data_quality_report(
        df,
        missing_df,
        duplicates,
        constant_columns
):

    logger.info("Generating Data Quality Report")

    with open(DATA_QUALITY_REPORT, "w") as report:

        report.write("# PharmaGuard Data Quality Report\n\n")

        report.write("## Dataset Overview\n\n")

        report.write(f"- Rows: {df.shape[0]}\n")
        report.write(f"- Columns: {df.shape[1]}\n\n")

        report.write("## Missing Values\n\n")

        total_missing = missing_df["Missing Values"].sum()

        report.write(f"Total Missing Values: {total_missing}\n\n")

        report.write("## Duplicate Records\n\n")

        report.write(f"Duplicate Rows: {duplicates}\n\n")

        report.write("## Constant Columns\n\n")

        if len(constant_columns) == 0:
            report.write("No constant columns found.\n\n")
        else:
            for col in constant_columns:
                report.write(f"- {col}\n")

        report.write("\nReport Generated Successfully.\n")

def generate_numerical_plots(df):

    logger.info("Generating Numerical Feature Plots")

    numerical_columns = df.select_dtypes(
        include=["int64", "float64"]
    ).columns

    for column in numerical_columns:

        save_histogram(df, column)

    logger.info("Numerical Plots Generated")


def main():

    initialize_project_directories()

    logger.info("Starting EDA")

    df = load_dataset()

    dataset_information(df)

    summary_statistics(df)

    # ----------------------------------------
    # Data Quality Analysis
    # ----------------------------------------

    missing_df = missing_values_analysis(df)

    duplicates = duplicate_analysis(df)

    unique_value_analysis(df)

    constant_columns = constant_column_analysis(df)

    data_type_analysis(df)

    generate_data_quality_report(
        df,
        missing_df,
        duplicates,
        constant_columns
    )
    generate_numerical_plots(df)
    logger.info("EDA Completed Successfully")


if __name__ == "__main__":
    main()