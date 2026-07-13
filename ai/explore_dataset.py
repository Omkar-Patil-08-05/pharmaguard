# explore_dataset.py
import pandas as pd
import os

# -----------------------------
# Create reports directory
# -----------------------------
os.makedirs("docs/reports", exist_ok=True)

# -----------------------------
# Load Dataset
# -----------------------------
file_path = "dataset/DataCoSupplyChainDataset.csv"

df = pd.read_csv(file_path, encoding="latin1")

# -----------------------------
# Dataset Shape
# -----------------------------
print("="*60)
print("DATASET SHAPE")
print("="*60)
print(df.shape)

# -----------------------------
# Column Names
# -----------------------------
print("\n" + "="*60)
print("COLUMN NAMES")
print("="*60)
print(df.columns.tolist())

# -----------------------------
# First Five Rows
# -----------------------------
print("\n" + "="*60)
print("FIRST FIVE ROWS")
print("="*60)
print(df.head())

# -----------------------------
# Dataset Information
# -----------------------------
print("\n" + "="*60)
print("DATA TYPES")
print("="*60)
print(df.dtypes)

# Save dataset information
with open("docs/reports/dataset_info.txt", "w") as f:
    df.info(buf=f)

# -----------------------------
# Missing Values
# -----------------------------
print("\n" + "="*60)
print("MISSING VALUES")
print("="*60)
print(df.isnull().sum())

# -----------------------------
# Duplicate Rows
# -----------------------------
print("\n" + "="*60)
print("DUPLICATE ROWS")
print("="*60)
print(df.duplicated().sum())

# -----------------------------
# Summary Statistics
# -----------------------------
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
print(df.describe(include="all"))

# Save summary statistics
df.describe(include="all").to_csv("docs/reports/summary_statistics.csv")

print("\nDataset exploration completed successfully.")