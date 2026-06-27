import pandas as pd
import os

# ==========================================================
# Load Engineered Dataset
# ==========================================================

df = pd.read_csv("dataset/engineered_data.csv")

os.makedirs("docs/reports", exist_ok=True)

print("=" * 60)
print("DATASET VALIDATION REPORT")
print("=" * 60)

# ==========================================================
# Dataset Shape
# ==========================================================

print("\nDataset Shape:")
print(df.shape)

# ==========================================================
# Missing Values
# ==========================================================

print("\nMissing Values:")
print(df.isnull().sum())

# ==========================================================
# Duplicate IDs
# ==========================================================

print("\nDuplicate Batch IDs:")
print(df["batch_id"].duplicated().sum())

print("\nDuplicate Shipment IDs:")
print(df["shipment_id"].duplicated().sum())

# ==========================================================
# Temperature Statistics
# ==========================================================

print("\nTemperature Statistics:")
print(df["temperature"].describe())

# ==========================================================
# Distance Statistics
# ==========================================================

print("\nDistance Statistics:")
print(df["distance"].describe())

# ==========================================================
# Warehouse Stops
# ==========================================================

print("\nWarehouse Stops Distribution:")
print(df["warehouse_stops"].value_counts().sort_index())

# ==========================================================
# Shipping Mode
# ==========================================================

print("\nShipping Mode Distribution:")
print(df["shipping_mode"].value_counts())

# ==========================================================
# Manufacturer Distribution
# ==========================================================

print("\nManufacturer Distribution:")
print(df["manufacturer"].value_counts())

# ==========================================================
# Delivery Status
# ==========================================================

print("\nDelivery Status Distribution:")
print(df["delivery_status"].value_counts())

# ==========================================================
# Save Validation Report
# ==========================================================

with open("docs/reports/validation_report.txt", "w") as report:

    report.write("PHARMAGUARD DATASET VALIDATION REPORT\n")
    report.write("=" * 60 + "\n\n")

    report.write(f"Dataset Shape: {df.shape}\n\n")

    report.write("Missing Values\n")
    report.write(str(df.isnull().sum()))
    report.write("\n\n")

    report.write(f"Duplicate Batch IDs: {df['batch_id'].duplicated().sum()}\n")
    report.write(f"Duplicate Shipment IDs: {df['shipment_id'].duplicated().sum()}\n")

print("\nValidation report saved successfully!")