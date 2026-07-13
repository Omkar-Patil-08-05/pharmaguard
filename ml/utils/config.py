"""
Project Configuration
PharmaGuard ML Module
"""

from pathlib import Path

# ==========================================
# Project Directories
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = PROJECT_ROOT / "dataset"

ML_DIR = PROJECT_ROOT / "ml"

EDA_DIR = ML_DIR / "eda"

PLOTS_DIR = EDA_DIR / "plots"

REPORTS_DIR = EDA_DIR / "reports"

MODEL_DIR = ML_DIR / "saved_models"

# ==========================================
# Dataset Paths
# ==========================================

RAW_DATASET = DATASET_DIR / "DataCoSupplyChainDataset.csv"

CLEANED_DATASET = DATASET_DIR / "cleaned_data.csv"

FINAL_DATASET = DATASET_DIR / "pharma_supply_chain.csv"

# ==========================================
# Output Files
# ==========================================

SUMMARY_STATISTICS = REPORTS_DIR / "summary_statistics.csv"

MISSING_VALUES = REPORTS_DIR / "missing_values.csv"

CORRELATION_MATRIX = REPORTS_DIR / "correlation_matrix.csv"

EDA_REPORT = REPORTS_DIR / "eda_report.md"

# ==========================================
# Plot Directories
# ==========================================

NUMERICAL_PLOTS_DIR = PLOTS_DIR / "numerical"

CATEGORICAL_PLOTS_DIR = PLOTS_DIR / "categorical"

CORRELATION_PLOTS_DIR = PLOTS_DIR / "correlation"

# ==========================================
# EDA Reports
# ==========================================

DATA_QUALITY_REPORT = REPORTS_DIR / "data_quality_report.md"

MISSING_VALUES_REPORT = REPORTS_DIR / "missing_values.csv"

DUPLICATE_REPORT = REPORTS_DIR / "duplicate_report.csv"

UNIQUE_VALUES_REPORT = REPORTS_DIR / "unique_values.csv"

CONSTANT_COLUMNS_REPORT = REPORTS_DIR / "constant_columns.csv"

DATA_TYPES_REPORT = REPORTS_DIR / "data_types.csv"

# ==========================================
# Random Seed
# ==========================================

RANDOM_STATE = 42

# ==========================================
# Preprocessing Outputs
# ==========================================

PROCESSED_DATASET = ML_DIR / "data" / "processed_dataset.csv"

PREPROCESSOR_PIPELINE = ML_DIR / "saved_models" / "preprocessing_pipeline.pkl"

ORDINAL_ENCODER = ML_DIR / "saved_models" / "ordinal_encoder.pkl"

STANDARD_SCALER = ML_DIR / "saved_models" / "standard_scaler.pkl"

# ==========================================
# Model Paths
# ==========================================

ISOLATION_FOREST_MODEL = MODEL_DIR / "isolation_forest.pkl"

ISOLATION_FOREST_RESULTS = (
    PROJECT_ROOT
    / "docs"
    / "reports"
    / "isolation_forest_predictions.csv"
)

ISOLATION_FOREST_SUMMARY = (
    PROJECT_ROOT
    / "docs"
    / "reports"
    / "isolation_forest_summary.md"
)